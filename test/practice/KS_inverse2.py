import pyro
from pyro.distributions import LogNormal, Uniform
import pyro.distributions as dist
import tqdm
import arviz
import pandas as pd
import torch
from stochproc import timeseries as ts, distributions as dists
from torch.utils.data import DataLoader, TensorDataset
from modulus.models.fno import FNO
from modulus.models.mlp.fully_connected import FullyConnected
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
import numpy as np
import torch.nn as nn

# class KuramotoSivashinsky(nn.Module):
#     '''Credit: https://scicomp.stackexchange.com/questions/37336/solving-numerically-the-1d-kuramoto-sivashinsky-equation-using-spectral-methods'''

#     def __init__(self, nu=1, L=100, nx=1024, dt=0.05):
#         super().__init__()
#         self.nu = nu
#         self.L = L
#         self.nx = nx
#         self.dt = dt
        
#         # Wave number mesh
#         self.k = torch.arange(-nx/2, nx/2, 1)
        
#         # Fourier Transform of the linear operator
#         self.FL = (((2 * np.pi) / L) * self.k) ** 2 - nu * (((2 * np.pi) / L) * self.k) ** 4
        
#         # Fourier Transform of the non-linear operator
#         self.FN = - (1 / 2) * ((1j) * ((2 * np.pi) / L) * self.k)

#     def forward(self, nt, u0):
#         # Initialize solution meshes
#         u = torch.ones((self.nx, nt))
#         u_hat = torch.ones((self.nx, nt), dtype=torch.complex64)
#         u_hat2 = torch.ones((self.nx, nt), dtype=torch.complex64)

#         # Set initial conditions
#         u[:, 0] = u0
#         u_hat[:, 0] = (1 / self.nx) * torch.fft.fftshift(torch.fft.fft(u0))
#         u_hat2[:, 0] = (1 / self.nx) * torch.fft.fftshift(torch.fft.fft(u0**2))

#         # Time-stepping
#         for j in range(nt-1):
#             uhat_current = u_hat[:, j]
#             uhat_current2 = u_hat2[:, j]
            
#             if j == 0:
#                 uhat_last = u_hat[:, 0]
#                 uhat_last2 = u_hat2[:, 0]
#             else:
#                 uhat_last = u_hat[:, j-1]
#                 uhat_last2 = u_hat2[:, j-1]
            
#             # Crank-Nicholson + Adam scheme
#             u_hat[:, j+1] = (1 / (1 - (self.dt / 2) * self.FL)) * (
#                 (1 + (self.dt / 2) * self.FL) * uhat_current + 
#                 (((3 / 2) * self.FN) * uhat_current2 - ((1 / 2) * self.FN) * uhat_last2) * self.dt
#             )
            
#             # Go back to real space
#             u[:, j+1] = torch.real(self.nx * torch.fft.ifft(torch.fft.ifftshift(u_hat[:, j+1])))
#             u_hat2[:, j+1] = (1 / self.nx) * torch.fft.fftshift(torch.fft.fft(u[:, j+1]**2))

#         return u.T

def kuramoto_sivashinsky_step(u_t, nu=1, L=100, nx=1024, dt=0.05):
    '''Solve one time step of the Kuramoto-Sivashinsky equation.'''
    
    # Wave number mesh
    k = torch.arange(-nx/2, nx/2, 1)
    
    # Fourier Transform of the linear operator
    FL = (((2 * np.pi) / L) * k) ** 2 - nu * (((2 * np.pi) / L) * k) ** 4
    
    # Fourier Transform of the non-linear operator
    FN = - (1 / 2) * ((1j) * ((2 * np.pi) / L) * k)
    
    # Fourier Transform of current state
    u_hat = (1 / nx) * torch.fft.fftshift(torch.fft.fft(u_t))
    u_hat2 = (1 / nx) * torch.fft.fftshift(torch.fft.fft(u_t**2))
    
    # Crank-Nicholson + Adam scheme
    u_hat_next = (1 / (1 - (dt / 2) * FL)) * (
        (1 + (dt / 2) * FL) * u_hat + 
        (((3 / 2) * FN) * u_hat2 - ((1 / 2) * FN) * u_hat2) * dt
    )
    
    # Go back to real space
    u_t_next = torch.real(nx * torch.fft.ifft(torch.fft.ifftshift(u_hat_next)))

    return u_t_next

# Initialize parameters
dim = 3  # Adjust dimension as needed
dt = 0.05
T = 30
nx = 1024
nt = int(T / dt)
init_point = torch.randn(nx)

# Define the Kuramoto-Sivashinsky model
ks_model = KuramotoSivashinsky(nu=1, L=100, nx=nx, dt=dt)

# Generate synthetic data using the KS model
synthetic_data = ks_model(nt, init_point)
print("shape", synthetic_data.shape)

# Convert the data to the appropriate shape for the neural network
learned_traj = synthetic_data
print("learned traj", learned_traj.shape)

def ks_probabilistic(data, verbose=False):
    nu = pyro.sample("nu", Uniform(low=0.1, high=10.0))
    L = pyro.sample("L", Uniform(low=50.0, high=150.0))
    
    ks_model = KuramotoSivashinsky(nu=nu, L=L, nx=nx, dt=dt)
    generated_data = ks_model(nt, init_point)
    # print("generated shape", generated_data.shape) #[1024, 600]
    
    pyro.sample("obs", dist.Normal(generated_data, 0.1).to_event(2), obs=data)

    return


# Define the neural network model
loss_type = "JAC"
model = FNO(
        in_channels=1,
        out_channels=1,
        num_fno_modes=25,
        padding=4,
        dimension=1,
        latent_channels=128
    ).to('cuda')
FNO_path = "../test_result/best_model_FNO_KS_"+str(loss_type)+".pth"
model.load_state_dict(torch.load(FNO_path))
model.eval()

# Initialize trajectory with initial condition
torch.cuda.empty_cache()
learned_traj_nn = torch.zeros(nt, nx)
print("init", init_point.shape, learned_traj_nn.shape)
learned_traj_nn[0] = init_point.cpu()

for i in range(1, len(learned_traj_nn)):
    out = model(learned_traj_nn[i - 1].reshape(1, 1, nx).cuda())
    print("out", out.shape)
    learned_traj_nn[i] = out.squeeze().detach().cpu()

# Define guide and SVI
guide = pyro.infer.autoguide.AutoDiagonalNormal(ks_probabilistic)
optim = pyro.optim.Adam({"lr": 0.01})
niter = 10000
smoothing = 0.99

svi = pyro.infer.SVI(ks_probabilistic, guide, optim, loss=pyro.infer.Trace_ELBO())
running_average = 0.0
bar = tqdm.tqdm(range(niter))
for n in bar:
    loss = svi.step(learned_traj_nn)    
    running_average = smoothing * running_average + (1 - smoothing) * loss
    bar.set_description(f"Loss: {running_average:,.2f}")

# Posterior sampling
num_samples = 10000
posterior_predictive = pyro.infer.Predictive(
    ks_probabilistic,
    guide=guide,
    num_samples=num_samples
)
posterior_draws = posterior_predictive(learned_traj_nn)

# Compute means
nu_mean = posterior_draws['nu'].mean()
L_mean = posterior_draws['L'].mean()

print(f"JAC Mean of nu: {nu_mean}")
print(f"JAC Mean of L: {L_mean}")

# Convert posterior draws to numpy arrays for plotting
nu_samples = posterior_draws['nu'].cpu().numpy().flatten()
L_samples = posterior_draws['L'].cpu().numpy().flatten()

# Global font size
plt.rcParams.update({'font.size': 14})
bin_num = 70
levels = [0.5, 0.9]

# Create distribution plots
true_nu = 1.0  # Adjust true values as needed
true_L = 100.0

fig, axs = plt.subplots(2, 1, figsize=(8, 12))

# Plot for nu
sns.histplot(nu_samples, stat="probability", kde=False, ax=axs[0], element="step", fill=False, bins=bin_num)
axs[0].set_title(rf"Posterior dist of $\nu$, mean = {nu_mean:.4f}")
axs[0].set_xlabel(r"$\nu$")
axs[0].set_ylabel('Density')
axs[0].axvline(true_nu, color='b', linestyle='--', linewidth=2, label=rf'True $\nu$ = {true_nu:.4f}')
axs[0].legend()

# Plot for L
sns.histplot(L_samples, stat="probability", kde=False, ax=axs[1], element="step", fill=False, bins=bin_num)
axs[1].set_title(rf"Posterior dist of $L$, mean = {L_mean:.4f}")
axs[1].set_xlabel(r"$L$")
axs[1].set_ylabel('Density')
axs[1].axvline(true_L, color='b', linestyle='--', linewidth=2, label=rf'True $L$ = {true_L:.4f}')
axs[1].legend()

plt.tight_layout()
plt.savefig(f"../test_result/KS_inv_dist_JAC.png")
plt.close()

# Create contour plots for parameters
fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(x=nu_samples, y=L_samples, ax=ax, color='blue', alpha=0.3)
sns.scatterplot(x=[true_nu], y=[true_L], color='orange', marker='o', s=60, label='True Parameter Value', ax=ax)

for level in levels:
    sns.kdeplot(x=nu_samples, y=L_samples, ax=ax, levels=[level], color='black', linewidths=1.5, linestyle='--', label=f"Credible Region: {str(level*100)}%")

ax.set_title(r"$\nu$ vs $L$")
ax.set_xlabel(r"$\nu$")
ax.set_ylabel(r"$L$")
plt.savefig(f"../test_result/KS_{str(loss_type)}_bc.png")
plt.close()
