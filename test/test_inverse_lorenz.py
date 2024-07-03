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
loss_type = "MSE"

def f(x, s, r, b):
    x1 = s * (x.value[..., 1] - x.value[..., 0])
    x2 = r * x.value[..., 0] - x.value[..., 1] - x.value[..., 0] * x.value[..., 2]
    x3 = - b * x.value[..., 2] + x.value[..., 0] * x.value[..., 1]

    return torch.stack((x1, x2, x3), dim=-1)

def lorenz_probabilistic(data, verbose=False):
    s = pyro.sample("sigma", Uniform(low=5.0, high=40.0))
    r = pyro.sample("rho", Uniform(low=10.0, high=50.0))
    b = pyro.sample("beta", Uniform(low=1.0, high=20.0))
    
    model = ts.RungeKutta(f, (s, r, b), init_point, dt=dt, event_dim=1, tuning_std=0.01)
    model.do_sample_pyro(pyro, data.shape[0], obs=data)
    return

# call model
model = FNO(
        in_channels=3,
        out_channels=3,
        num_fno_modes=3,
        padding=4,
        dimension=1,
        latent_channels=128).to('cuda')
FNO_path = "../test_result/best_model_FNO_Lorenz_"+str(loss_type)+".pth"
model.load_state_dict(torch.load(FNO_path))
model.eval()

# generate data
torch.cuda.empty_cache()
learned_traj = torch.zeros(T*int(1/dt), 3)
learned_traj[0] = init_point
print(learned_traj.shape)
for i in range(1, len(learned_traj)):
    out = model(learned_traj[i-1].reshape(1, dim, 1).cuda()).reshape(dim,-1)
    learned_traj[i] = out.squeeze().detach().cpu()

guide = pyro.infer.autoguide.AutoDiagonalNormal(lorenz_probabilistic)
optim = pyro.optim.Adam({"lr": 0.01})
svi = pyro.infer.SVI(lorenz_probabilistic, guide, optim, loss=pyro.infer.Trace_ELBO())

niter = 10000
pyro.clear_param_store()
running_average = 0.0
smoothing = 0.99
bar = tqdm.tqdm(range(niter))

for n in bar:
    loss = svi.step(learned_traj)    
    running_average = smoothing * running_average + (1 - smoothing) * loss
    bar.set_description(f"Loss: {running_average:,.2f}")

num_samples = 10000
posterior_predictive = pyro.infer.Predictive(
    lorenz_probabilistic,
    guide=guide,
    num_samples=num_samples
)

posterior_draws = {k: v.unsqueeze(0) for k, v in posterior_predictive(learned_traj).items()}
print("posterior draws", posterior_draws['beta'].shape)

# Compute the mean of beta, sigma, and rho
beta_mean = posterior_draws['beta'].mean()
sigma_mean = posterior_draws['sigma'].mean()
rho_mean = posterior_draws['rho'].mean()

print(f"Mean of beta: {beta_mean}")
print(f"Mean of sigma: {sigma_mean}")
print(f"Mean of rho: {rho_mean}")

# Convert posterior draws to numpy arrays for plotting
beta_samples = posterior_draws['beta'].cpu().numpy().flatten()
sigma_samples = posterior_draws['sigma'].cpu().numpy().flatten()
rho_samples = posterior_draws['rho'].cpu().numpy().flatten()

# global font size
plt.rcParams.update({'font.size': 14})
bin_num =70
# Specify credible regions as contours
levels = [0.5, 0.9]  # Contour levels for 50% and 90% credible regions

# Create distribution plots for beta, sigma, and rho
fig, axs = plt.subplots(3, 1, figsize=(8, 12))
sns.histplot(beta_samples, stat="probability", kde=False, ax=axs[0], element="step", fill=False, bins=bin_num)
axs[0].set_title(rf"Posterior dist of $\beta$, mean = {beta_mean:.4f}")
axs[0].set_xlabel(r"$\beta$")
axs[0].set_ylabel('Density')
sns.histplot(sigma_samples, stat="probability", kde=False, ax=axs[1], element="step", fill=False, bins=bin_num)
axs[1].set_title(rf"Posterior dist of $\sigma$, mean = {sigma_mean:.4f}")
axs[1].set_xlabel(r"$\sigma$")
axs[1].set_ylabel('Density')
sns.histplot(rho_samples, stat="probability", kde=False, ax=axs[2], element="step", fill=False, bins=bin_num)
axs[2].set_title(rf"Posterior dist of $\rho$, mean = {rho_mean:.4f}")
axs[2].set_xlabel(r"$\rho$")
axs[2].set_ylabel('Density')
plt.tight_layout()
plt.savefig(f"../test_result/Lorenz_inv_dist_{str(loss_type)}.png")

# Create contour plot for beta and sigma
fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(x=beta_samples, y=sigma_samples, ax=ax, color='blue', alpha=0.3)
if loss_type == "JAC":
    sns.scatterplot(x=[2.66], y=[10.0], color='orange', marker='o', s=50, label='True Parameter Value', ax=ax)
# sns.kdeplot(x=beta_samples, y=sigma_samples, ax=ax, color='black')
# Plot credible regions as contours
for level in levels:
    sns.kdeplot(
        x=beta_samples, y=sigma_samples, ax=ax, 
        levels=[level], color='black', linewidths=1.5, linestyle='--', label=f"Credible Region: {str(level*100)}%"
    )
ax.set_title(r"$\beta$ vs $\sigma$")
ax.set_xlabel(r"$\beta$")
ax.set_ylabel(r"$\sigma$")
plt.savefig(f"../test_result/Lorenz_{str(loss_type)}_betasigma.png")
plt.close()

# Create contour plot for rho and sigma
fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(x=rho_samples, y=sigma_samples, ax=ax, color='blue', alpha=0.3)
if loss_type == "JAC":
    sns.scatterplot(x=[28.], y=[10.0], color='orange', marker='o', s=50, label='True Parameter Value', ax=ax)
# sns.kdeplot(x=rho_samples, y=sigma_samples, ax=ax, color='black')
for l in levels:
    sns.kdeplot(
        x=rho_samples, y=sigma_samples, ax=ax, 
        levels=[l], color='black', linewidths=1.5, linestyle='--', label=f"Credible Region: {str(l*100)}%"
    )
ax.set_title(r"$\rho$ vs $\sigma$")
ax.set_xlabel(r"$\rho$")
ax.set_ylabel(r"$\sigma$")
plt.savefig(f"../test_result/Lorenz_{str(loss_type)}_rhosigma.png")
