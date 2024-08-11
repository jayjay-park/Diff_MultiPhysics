import pyro
import pyro.distributions as dist
import torch
import torch.nn as nn
import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class KuramotoSivashinsky(nn.Module):
    def __init__(self, nu=1, L=100, nx=1024, dt=0.05):
        super().__init__()
        self.nu = nu
        self.L = L
        self.nx = nx
        self.dt = dt
        
        self.k = torch.arange(-nx/2, nx/2, 1)
        self.FL = (((2 * np.pi) / L) * self.k) ** 2 - nu * (((2 * np.pi) / L) * self.k) ** 4
        self.FN = - (1 / 2) * ((1j) * ((2 * np.pi) / L) * self.k)

    def forward(self, nt, u0):
        u = torch.ones((self.nx, nt))
        u_hat = torch.ones((self.nx, nt), dtype=torch.complex64)
        u_hat2 = torch.ones((self.nx, nt), dtype=torch.complex64)

        u[:, 0] = u0
        u_hat[:, 0] = (1 / self.nx) * torch.fft.fftshift(torch.fft.fft(u0))
        u_hat2[:, 0] = (1 / self.nx) * torch.fft.fftshift(torch.fft.fft(u0**2))

        for j in range(nt-1):
            uhat_current = u_hat[:, j]
            uhat_current2 = u_hat2[:, j]
            
            if j == 0:
                uhat_last = u_hat[:, 0]
                uhat_last2 = u_hat2[:, 0]
            else:
                uhat_last = u_hat[:, j-1]
                uhat_last2 = u_hat2[:, j-1]
            
            u_hat[:, j+1] = (1 / (1 - (self.dt / 2) * self.FL)) * (
                (1 + (self.dt / 2) * self.FL) * uhat_current + 
                (((3 / 2) * self.FN) * uhat_current2 - ((1 / 2) * self.FN) * uhat_last2) * self.dt
            )
            
            u[:, j+1] = torch.real(self.nx * torch.fft.ifft(torch.fft.ifftshift(u_hat[:, j+1])))
            u_hat2[:, j+1] = (1 / self.nx) * torch.fft.fftshift(torch.fft.fft(u[:, j+1]**2))

        return u

def ks_probabilistic(data, verbose=False):
    nu = pyro.sample("nu", dist.Uniform(0.5, 2.0))
    L = pyro.sample("L", dist.Uniform(50.0, 150.0))
    
    model = KuramotoSivashinsky(nu=nu, L=L)
    u0 = data[:, 0]
    nt = data.shape[1]
    
    predicted = model(nt, u0)
    
    noise_scale = pyro.param("noise_scale", torch.tensor(0.1), constraint=dist.constraints.positive)
    pyro.sample("obs", dist.Normal(predicted, noise_scale).to_event(2), obs=data)
    
    return predicted


# inference
torch.autograd.set_detect_anomaly(True)
# Initialize
nx = 1024
dt = 0.05
T = 30
nt = int(T / dt)
init_point = torch.randn(nx)

# Generate synthetic data (replace this with your actual data)
true_model = KuramotoSivashinsky(nu=1, L=100)
true_data = true_model(nt, init_point)

# Define guide
guide = pyro.infer.autoguide.AutoDiagonalNormal(ks_probabilistic)

# Setup SVI
optim = pyro.optim.Adam({"lr": 0.01})
svi = pyro.infer.SVI(ks_probabilistic, guide, optim, loss=pyro.infer.Trace_ELBO())

# Training loop
num_iterations = 10000
for i in tqdm.tqdm(range(num_iterations)):
    loss = svi.step(true_data)
    if i % 100 == 0:
        print(f"Iteration {i}: loss = {loss}")

# Posterior sampling
num_samples = 10000
posterior_predictive = pyro.infer.Predictive(ks_probabilistic, guide=guide, num_samples=num_samples)
posterior_samples = posterior_predictive(true_data)

# Extract samples
nu_samples = posterior_samples['nu'].cpu().numpy()
L_samples = posterior_samples['L'].cpu().numpy()

# Compute means
nu_mean = nu_samples.mean()
L_mean = L_samples.mean()

print(f"Mean of nu: {nu_mean}")
print(f"Mean of L: {L_mean}")

# plot
# Plotting
plt.figure(figsize=(12, 8))

# Plot for nu
plt.subplot(2, 1, 1)
sns.histplot(nu_samples, kde=True)
plt.axvline(1.0, color='r', linestyle='--', label='True Value')
plt.title(f"Posterior Distribution of nu, mean = {nu_mean:.4f}")
plt.xlabel("nu")
plt.legend()

# Plot for L
plt.subplot(2, 1, 2)
sns.histplot(L_samples, kde=True)
plt.axvline(100.0, color='r', linestyle='--', label='True Value')
plt.title(f"Posterior Distribution of L, mean = {L_mean:.4f}")
plt.xlabel("L")
plt.legend()

plt.tight_layout()
plt.savefig("KS_parameter_distributions.png")
plt.close()

# Contour plot
plt.figure(figsize=(10, 8))
sns.kdeplot(x=nu_samples, y=L_samples, cmap="YlGnBu", shade=True, cbar=True)
plt.scatter(1.0, 100.0, color='r', s=100, label='True Value')
plt.title("Joint Posterior Distribution of nu and L")
plt.xlabel("nu")
plt.ylabel("L")
plt.legend()
plt.savefig("KS_joint_distribution.png")
plt.close()