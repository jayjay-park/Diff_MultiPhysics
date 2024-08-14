import torch
import numpy as np
import pyro
import pyro.distributions as dist
import pyro.infer
import pyro.optim
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.distributions import Uniform
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm
from modulus.models.fno import FNO

torch.autograd.set_detect_anomaly(True)

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

# Generate synthetic data using the KS model
synthetic_data = torch.zeros((nt, nx))
synthetic_data[0] = init_point

for t in range(1, nt):
    synthetic_data[t] = kuramoto_sivashinsky_step(synthetic_data[t-1], nu=1, L=100, nx=nx, dt=dt)

print("shape", synthetic_data.shape)

# Convert the data to the appropriate shape for the neural network
learned_traj = synthetic_data
print("learned traj", learned_traj.shape)

def ks_probabilistic(data, verbose=False):
    nu = pyro.sample("nu", Uniform(low=0.1, high=10.0))
    L = pyro.sample("L", Uniform(low=50.0, high=150.0))
    
    # Ensure the data is enclosed in a plate
    with pyro.plate("data_plate", size=data.size(0), dim=-1):
        next_data = kuramoto_sivashinsky_step(init_point, nu=nu, L=L, nx=nx, dt=dt)
        pyro.sample("obs", dist.Normal(next_data, 0.1).to_event(1), obs=data)

    return next_data

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
learned_traj_nn = torch.zeros(nt, nx).cpu()
print("init", init_point.shape, learned_traj_nn.shape)
learned_traj_nn[0] = init_point.cpu()

for i in range(1, len(learned_traj_nn)):
    out = model(learned_traj_nn[i - 1].reshape(1, 1, nx).cuda())
    print("out", out.shape)
    learned_traj_nn[i] = out.squeeze().detach().cpu()

# Define guide and SVI
guide = AutoDiagonalNormal(ks_probabilistic)
optim = pyro.optim.Adam({"lr": 0.01})
niter = 10000
smoothing = 0.99

svi = SVI(ks_probabilistic, guide, optim, loss=Trace_ELBO())
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
