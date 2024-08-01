import pyro
from pyro.distributions import LogNormal, Uniform
import tqdm
import arviz
import pandas as pd
import torch
from stochproc import timeseries as ts, distributions as dists
from torch.utils.data import DataLoader, TensorDataset
from modulus.models.fno import FNO
from modulus.models.mlp.fully_connected import FullyConnected
import matplotlib.pyplot as plt  # Import matplotlib
import seaborn as sns
from scipy.stats import gaussian_kde


# initialize
dim=3
dt = 0.01
T = 30
init_point = torch.randn(dim)
# loss_type = "MSE"

def f(x, s, r, b):
    x1 = s * (x.value[..., 1] - x.value[..., 0])
    x2 = r * x.value[..., 0] - x.value[..., 1] - x.value[..., 0] * x.value[..., 2]
    x3 = - b * x.value[..., 2] + x.value[..., 0] * x.value[..., 1]

    return torch.stack((x1, x2, x3), dim=-1)

def lorenz_probabilistic(data, verbose=False):
    s = pyro.sample("sigma", Uniform(low=5.0, high=40.0))
    r = pyro.sample("rho", Uniform(low=10.0, high=50.0))
    b = pyro.sample("beta", Uniform(low=1.0, high=20.0))
    
    m = ts.RungeKutta(f, (s, r, b), init_point, dt=dt, event_dim=1, tuning_std=0.01)
    m.do_sample_pyro(pyro, data.shape[0], obs=data)
    return

# call model
model = FNO(
        in_channels=3,
        out_channels=3,
        num_fno_modes=3,
        padding=4,
        dimension=1,
        latent_channels=64).to('cuda')
FNO_path = "../test_result/best_model_FNO_Lorenz_JAC.pth"
model.load_state_dict(torch.load(FNO_path))
model.eval()

# call model
# loss_type = "MSE"
MSE_model = FNO(
        in_channels=3,
        out_channels=3,
        num_fno_modes=3,
        padding=4,
        dimension=1,
        latent_channels=128).to('cuda')
MSE_FNO_path = "../test_result/best_model_FNO_Lorenz_MSE.pth"
MSE_model.load_state_dict(torch.load(MSE_FNO_path))
MSE_model.eval()

# Generate data
torch.cuda.empty_cache()
learned_traj = torch.zeros(T*int(1/dt), 3)
learned_traj[0] = init_point.cpu()
MSE_learned_traj = torch.zeros(T*int(1/dt), 3)
MSE_learned_traj[0] = init_point.cpu()

for i in range(1, len(learned_traj)):
    out = model(learned_traj[i-1].reshape(1, dim, 1).cuda()).reshape(dim,-1)
    learned_traj[i] = out.squeeze().detach().cpu()

for i in range(1, len(MSE_learned_traj)):
    MSE_out = MSE_model(MSE_learned_traj[i-1].reshape(1, dim, 1).cuda()).reshape(dim,-1)
    MSE_learned_traj[i] = MSE_out.squeeze().detach().cpu()

# Define separate guides for JAC and MSE models
JAC_guide = pyro.infer.autoguide.AutoDiagonalNormal(lorenz_probabilistic)
MSE_guide = pyro.infer.autoguide.AutoDiagonalNormal(lorenz_probabilistic)

optim = pyro.optim.Adam({"lr": 0.01})
niter = 10000
smoothing = 0.99

# For JAC model
pyro.clear_param_store()
JAC_svi = pyro.infer.SVI(lorenz_probabilistic, JAC_guide, optim, loss=pyro.infer.Trace_ELBO())
running_average = 0.0
bar = tqdm.tqdm(range(niter))
for n in bar:
    loss = JAC_svi.step(learned_traj)    
    running_average = smoothing * running_average + (1 - smoothing) * loss
    bar.set_description(f"JAC Loss: {running_average:,.2f}")
# Posterior sampling
num_samples = 10000

JAC_posterior_predictive = pyro.infer.Predictive(
    lorenz_probabilistic,
    guide=JAC_guide,
    num_samples=num_samples
)
JAC_posterior_draws = JAC_posterior_predictive(learned_traj)


# For MSE model
pyro.clear_param_store()
MSE_svi = pyro.infer.SVI(lorenz_probabilistic, MSE_guide, optim, loss=pyro.infer.Trace_ELBO())
running_average = 0.0
bar = tqdm.tqdm(range(niter))
for n in bar:
    loss = MSE_svi.step(MSE_learned_traj)    
    running_average = smoothing * running_average + (1 - smoothing) * loss
    bar.set_description(f"MSE Loss: {running_average:,.2f}")

MSE_posterior_predictive = pyro.infer.Predictive(
    lorenz_probabilistic,
    guide=MSE_guide,
    num_samples=num_samples
)
MSE_posterior_draws = MSE_posterior_predictive(MSE_learned_traj)

# Compute means
JAC_beta_mean = JAC_posterior_draws['beta'].mean()
JAC_sigma_mean = JAC_posterior_draws['sigma'].mean()
JAC_rho_mean = JAC_posterior_draws['rho'].mean()

MSE_beta_mean = MSE_posterior_draws['beta'].mean()
MSE_sigma_mean = MSE_posterior_draws['sigma'].mean()
MSE_rho_mean = MSE_posterior_draws['rho'].mean()

print(f"JAC Mean of beta: {JAC_beta_mean}")
print(f"JAC Mean of sigma: {JAC_sigma_mean}")
print(f"JAC Mean of rho: {JAC_rho_mean}")
print(f"MSE Mean of beta: {MSE_beta_mean}")
print(f"MSE Mean of sigma: {MSE_sigma_mean}")
print(f"MSE Mean of rho: {MSE_rho_mean}")

# Convert posterior draws to numpy arrays for plotting
JAC_beta_samples = JAC_posterior_draws['beta'].cpu().numpy().flatten()
JAC_sigma_samples = JAC_posterior_draws['sigma'].cpu().numpy().flatten()
JAC_rho_samples = JAC_posterior_draws['rho'].cpu().numpy().flatten()

MSE_beta_samples = MSE_posterior_draws['beta'].cpu().numpy().flatten()
MSE_sigma_samples = MSE_posterior_draws['sigma'].cpu().numpy().flatten()
MSE_rho_samples = MSE_posterior_draws['rho'].cpu().numpy().flatten()

# global font size
plt.rcParams.update({'font.size': 14})
bin_num =70
# Specify credible regions as contours
levels = [0.5, 0.9]  # Contour levels for 50% and 90% credible regions

# Create distribution plots for beta, sigma, and rho
true_rho = 28.0
true_sigma = 10.0
true_beta = 8 / 3

fig, axs = plt.subplots(3, 1, figsize=(8, 12))
# Plot for beta
sns.histplot(JAC_beta_samples, stat="probability", kde=False, ax=axs[0], element="step", fill=False, bins=bin_num)
axs[0].set_title(rf"Posterior dist of $\beta$, mean = {JAC_beta_mean:.4f}")
axs[0].set_xlabel(r"$\beta$")
axs[0].set_ylabel('Density')
axs[0].axvline(true_beta, color='b', linestyle='--', linewidth=2, label=rf'True $\beta$ = {true_beta:.4f}')
axs[0].legend()

# Plot for sigma
sns.histplot(JAC_sigma_samples, stat="probability", kde=False, ax=axs[1], element="step", fill=False, bins=bin_num)
axs[1].set_title(rf"Posterior dist of $\sigma$, mean = {JAC_sigma_mean:.4f}")
axs[1].set_xlabel(r"$\sigma$")
axs[1].set_ylabel('Density')
axs[1].axvline(true_sigma, color='b', linestyle='--', linewidth=2, label=rf'True $\sigma$ = {true_sigma:.4f}')
axs[1].legend()

# Plot for rho
sns.histplot(JAC_rho_samples, stat="probability", kde=False, ax=axs[2], element="step", fill=False, bins=bin_num)
axs[2].set_title(rf"Posterior dist of $\rho$, mean = {JAC_rho_mean:.4f}")
axs[2].set_xlabel(r"$\rho$")
axs[2].set_ylabel('Density')
axs[2].axvline(true_rho, color='b', linestyle='--', linewidth=2, label=rf'True $\rho$ = {true_rho:.4f}')
axs[2].legend()
plt.tight_layout()
plt.savefig(f"../test_result/Lorenz_inv_dist_JAC.png")
plt.close()

fig, axs = plt.subplots(3, 1, figsize=(8, 12))
# Plot for beta
sns.histplot(MSE_beta_samples, stat="probability", kde=False, ax=axs[0], element="step", fill=False, bins=bin_num)
axs[0].set_title(rf"Posterior dist of $\beta$, mean = {MSE_beta_mean:.4f}")
axs[0].set_xlabel(r"$\beta$")
axs[0].set_ylabel('Density')
axs[0].axvline(true_beta, color='b', linestyle='--', linewidth=2, label=rf'True $\beta$ = {true_beta:.4f}')
axs[0].legend()

# Plot for sigma
sns.histplot(MSE_sigma_samples, stat="probability", kde=False, ax=axs[1], element="step", fill=False, bins=bin_num)
axs[1].set_title(rf"Posterior dist of $\sigma$, mean = {MSE_sigma_mean:.4f}")
axs[1].set_xlabel(r"$\sigma$")
axs[1].set_ylabel('Density')
axs[1].axvline(true_sigma, color='b', linestyle='--', linewidth=2, label=rf'True $\sigma$ = {true_sigma:.4f}')
axs[1].legend()

# Plot for rho
sns.histplot(MSE_rho_samples, stat="probability", kde=False, ax=axs[2], element="step", fill=False, bins=bin_num)
axs[2].set_title(rf"Posterior dist of $\rho$, mean = {MSE_rho_mean:.4f}")
axs[2].set_xlabel(r"$\rho$")
axs[2].set_ylabel('Density')
axs[2].axvline(true_rho, color='b', linestyle='--', linewidth=2, label=rf'True $\rho$ = {true_rho:.4f}')
axs[2].legend()
plt.tight_layout()
plt.savefig(f"../test_result/Lorenz_inv_dist_MSE.png")
plt.close()

# Create contour plot for beta and sigma
fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(x=JAC_beta_samples, y=JAC_sigma_samples, ax=ax, color='blue', alpha=0.3)
sns.scatterplot(x=MSE_beta_samples, y=MSE_sigma_samples, ax=ax, color='red', alpha=0.3)
# if loss_type == "JAC":
sns.scatterplot(x=[2.66], y=[10.0], color='orange', marker='o', s=60, label='True Parameter Value', ax=ax)
# sns.kdeplot(x=beta_samples, y=sigma_samples, ax=ax, color='black')
# Plot credible regions as contours
for level in levels:
    sns.kdeplot(
        x=JAC_beta_samples, y=JAC_sigma_samples, ax=ax, 
        levels=[level], color='black', linewidths=1.5, linestyle='--', label=f"Credible Region: {str(level*100)}%"
    )
for level in levels:
    sns.kdeplot(
        x=MSE_beta_samples, y=MSE_sigma_samples, ax=ax, 
        levels=[level], color='gray', linewidths=1.5, linestyle='--', label=f"Credible Region: {str(level*100)}%"
    )
ax.set_title(r"$\beta$ vs $\sigma$")
ax.set_xlabel(r"$\beta$")
ax.set_ylabel(r"$\sigma$")
plt.savefig(f"../test_result/Lorenz_all_betasigma.png")
plt.close()

# Create contour plot for rho and sigma
fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(x=JAC_rho_samples, y=JAC_sigma_samples, ax=ax, color='blue', alpha=0.3)
sns.scatterplot(x=MSE_rho_samples, y=MSE_sigma_samples, ax=ax, color='red', alpha=0.3)
# if loss_type == "JAC":
sns.scatterplot(x=[28.], y=[10.0], color='orange', marker='o', s=50, label='True Parameter Value', ax=ax)
# sns.kdeplot(x=rho_samples, y=sigma_samples, ax=ax, color='black')
for l in levels:
    sns.kdeplot(
        x=JAC_rho_samples, y=JAC_sigma_samples, ax=ax, 
        levels=[l], color='black', linewidths=1.5, linestyle='--', label=f"Credible Region: {str(l*100)}%"
    )
for l in levels:
    sns.kdeplot(
        x=MSE_rho_samples, y=MSE_sigma_samples, ax=ax, 
        levels=[l], color='gray', linewidths=1.5, linestyle='--', label=f"Credible Region: {str(l*100)}%"
    )
ax.set_title(r"$\rho$ vs $\sigma$")
ax.set_xlabel(r"$\rho$")
ax.set_ylabel(r"$\sigma$")
plt.savefig(f"../test_result/Lorenz_all_rhosigma.png")
