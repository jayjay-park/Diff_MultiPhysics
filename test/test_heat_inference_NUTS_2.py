import torch
import pyro
import pyro.distributions as dist
from pyro.infer import NUTS, MCMC
import matplotlib.pyplot as plt
from modulus.models.fno import FNO
from test_heat import create_q_function, solve_heat_equation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Generate synthetic data
nx, ny = 60, 60
true_q = torch.randn(nx, ny, device=device)
true_k = torch.ones((nx, ny), device=device)
true_T = solve_heat_equation(true_k, true_q, nx, ny)

# Load trained FNO
trainedFNO = FNO(
    in_channels=1,
    out_channels=1,
    decoder_layer_size=128,
    num_fno_layers=6,
    num_fno_modes=24,
    padding=3,
    dimension=2,
    latent_channels=64
).to(device)

loss_type = "JAC"
FNO_path = f"../test_result/best_model_FNO_Heat_{loss_type}.pth"
trainedFNO.load_state_dict(torch.load(FNO_path))
trainedFNO.eval()

# Generate predicted temperature using FNO
pred_T = trainedFNO(true_q.unsqueeze(dim=0).unsqueeze(dim=1).float()).squeeze()

# Add noise to create observed data
noise_std = 0.1
observed_T = pred_T + noise_std * torch.randn_like(pred_T)

# Pyro model: probabilistic model
def model(observed=None):
    # Priors for freq_x and freq_y
    freq_x = pyro.sample("freq_x", dist.Normal(torch.tensor(5.0, device=device), torch.tensor(1.0, device=device)))
    freq_y = pyro.sample("freq_y", dist.Normal(torch.tensor(3.0, device=device), torch.tensor(1.0, device=device)))

    # Create the heat source q using the sampled frequencies
    x = torch.linspace(0, 1, nx, device=device)
    y = torch.linspace(0, 1, ny, device=device)
    X, Y = torch.meshgrid(x, y, indexing='ij')

    q = 3000 * (torch.sin(freq_x * X) * torch.sin(freq_y * Y) + torch.cos(freq_y * Y) * torch.cos(2 * X))
    
    # Forward model using the heat equation solver
    T = solve_heat_equation(true_k, q, nx, ny)
    
    # Likelihood
    pyro.sample("obs", dist.Normal(T, noise_std * torch.ones_like(T)).to_event(2), obs=observed)

    return T


# Set up NUTS sampler
nuts_kernel = NUTS(model, adapt_step_size=True)

# Run MCMC
num_samples = 1000
mcmc = MCMC(nuts_kernel, num_samples=num_samples, warmup_steps=200)
mcmc.run(observed_T)

# Get samples for freq_x and freq_y
posterior_freq_x = mcmc.get_samples()["freq_x"]
posterior_freq_y = mcmc.get_samples()["freq_y"]

# Compute mean and standard deviation
freq_x_mean = posterior_freq_x.mean()
freq_x_std = posterior_freq_x.std()
freq_y_mean = posterior_freq_y.mean()
freq_y_std = posterior_freq_y.std()

print(f"Estimated freq_x: Mean = {freq_x_mean.item()}, Std = {freq_x_std.item()}")
print(f"Estimated freq_y: Mean = {freq_y_mean.item()}, Std = {freq_y_std.item()}")


# Plot histograms for freq_x and freq_y
plt.figure(figsize=(12, 6))

# Histogram for freq_x
plt.subplot(1, 2, 1)
plt.hist(posterior_freq_x.detach().cpu().numpy(), bins=30, density=True, color='blue', alpha=0.6)
plt.axvline(freq_x_mean.detach().cpu().numpy(), color='red', linestyle='--', label=f'Mean: {freq_x_mean.item():.2f}')
plt.title("Posterior Distribution of freq_x")
plt.xlabel("freq_x")
plt.ylabel("Density")
plt.legend()

# Histogram for freq_y
plt.subplot(1, 2, 2)
plt.hist(posterior_freq_y.detach().cpu().numpy(), bins=30, density=True, color='green', alpha=0.6)
plt.axvline(freq_y_mean.detach().cpu().numpy(), color='red', linestyle='--', label=f'Mean: {freq_y_mean.item():.2f}')
plt.title("Posterior Distribution of freq_y")
plt.xlabel("freq_y")
plt.ylabel("Density")
plt.legend()

plt.tight_layout()
plt.savefig(f"../test_result/Posterior_Distributions_freq_x_freq_y.png")

# Compute 95% credible intervals
freq_x_lower, freq_x_upper = torch.quantile(posterior_freq_x, torch.tensor([0.025, 0.975]).cuda())
freq_y_lower, freq_y_upper = torch.quantile(posterior_freq_y, torch.tensor([0.025, 0.975]).cuda())

# Plot credible intervals
plt.figure(figsize=(12, 6))

# Credible interval for freq_x
plt.subplot(1, 2, 1)
plt.errorbar(1, freq_x_mean.detach().cpu().numpy(), yerr=[[freq_x_mean.detach().cpu().numpy() - freq_x_lower.detach().cpu().numpy()], [freq_x_upper.detach().cpu().numpy() - freq_x_mean.detach().cpu().numpy()]], fmt='o', color='blue', capsize=5, label='95% CI')
plt.title("95% Credible Interval for freq_x")
plt.xlabel("freq_x")
plt.ylabel("Value")
plt.legend()
plt.xticks([])

# Credible interval for freq_y
plt.subplot(1, 2, 2)
plt.errorbar(1, freq_y_mean.detach().cpu().numpy(), yerr=[[freq_y_mean.detach().cpu().numpy() - freq_y_lower.detach().cpu().numpy()], [freq_y_upper.detach().cpu().numpy() - freq_y_mean.detach().cpu().numpy()]], fmt='o', color='green', capsize=5, label='95% CI')
plt.title("95% Credible Interval for freq_y")
plt.xlabel("freq_y")
plt.ylabel("Value")
plt.legend()
plt.xticks([])

plt.tight_layout()
plt.savefig(f"../test_result/Credible_Intervals_freq_x_freq_y.png")

# Generate q samples based on inferred freq_x and freq_y
q_samples = []
for fx, fy in zip(posterior_freq_x.detach().cpu(), posterior_freq_y.detach().cpu()):
    q_samples.append(create_q_function(nx, ny, fx, fy, noise_level=0.0))
q_samples = torch.stack(q_samples)

# Compute mean and variance of q
q_mean = q_samples.mean(dim=0)
q_std = q_samples.std(dim=0)

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Mean of q
im0 = axes[0].imshow(q_mean.detach().cpu().numpy(), cmap='inferno')
axes[0].set_title("Mean of Estimated q")
fig.colorbar(im0, ax=axes[0])

# Variance of q
im1 = axes[1].imshow(q_std.detach().cpu().numpy(), cmap='inferno')
axes[1].set_title("Variance of Estimated q")
fig.colorbar(im1, ax=axes[1])

# Error between true q and mean q
error = (true_q - q_mean).abs()
im2 = axes[2].imshow(error.detach().cpu().numpy(), cmap='inferno')
axes[2].set_title("Absolute Error in q")
fig.colorbar(im2, ax=axes[2])

plt.tight_layout()
plt.savefig(f"../plot/Heat_plot/q_mean_variance.png")

# # Set up NUTS sampler
# nuts_kernel = NUTS(model, adapt_step_size=True)

# # Run MCMC
# num_samples = 1000
# mcmc = MCMC(nuts_kernel, num_samples=num_samples, warmup_steps=200)
# mcmc.run(observed_T)

# # Get samples
# posterior_samples = mcmc.get_samples()["q"]

# # Compute statistics
# q_mean = posterior_samples.mean(dim=0)
# q_std = posterior_samples.std(dim=0)

# # Compute 95% credible interval
# q_lower, q_upper = torch.quantile(posterior_samples, torch.tensor([0.025, 0.975]).cuda(), dim=0)

# fig, axes = plt.subplots(2, 2, figsize=(16, 16))
# plt.rcParams.update({'font.size': 14})

# # True q
# im0 = axes[0, 0].imshow(true_q.cpu().numpy(), cmap='inferno')
# axes[0, 0].set_title(r"True Heat Source $q$")
# fig.colorbar(im0, ax=axes[0, 0], fraction=0.045, pad=0.06)

# # Inferred q (mean)
# im1 = axes[0, 1].imshow(q_mean.detach().cpu().numpy(), cmap='inferno')
# axes[0, 1].set_title(r"Inferred Heat Source $q$ (Mean)")
# fig.colorbar(im1, ax=axes[0, 1], fraction=0.045, pad=0.06)

# # Uncertainty (standard deviation)
# im2 = axes[1, 0].imshow(q_std.cpu().detach().numpy(), cmap='viridis')
# axes[1, 0].set_title(r"Uncertainty in $q$ (Std Dev)")
# fig.colorbar(im2, ax=axes[1, 0], fraction=0.045, pad=0.06)

# # Error (difference between true and inferred)
# error = (true_q - q_mean).abs()
# im3 = axes[1, 1].imshow(error.detach().cpu().numpy(), cmap='viridis')
# axes[1, 1].set_title("Absolute Error in $q$")
# fig.colorbar(im3, ax=axes[1, 1], fraction=0.045, pad=0.06)

# plt.tight_layout()
# plt.savefig(f"../test_result/Heat_{loss_type}_posterior_NUTS.png")

# # Plot a slice with credible interval
# slice_idx = nx // 2
# plt.figure(figsize=(12, 6))
# x = torch.arange(ny)
# plt.fill_between(x.cpu().numpy(), 
#                  q_lower[slice_idx].detach().cpu().numpy(), 
#                  q_upper[slice_idx].detach().cpu().numpy(), 
#                  alpha=0.3, label='95% CI')
# plt.plot(x.cpu().numpy(), q_mean[slice_idx].detach().cpu().numpy(), label='Mean', color='r')
# plt.plot(x.cpu().numpy(), true_q[slice_idx].detach().cpu().numpy(), label='True', color='k', linestyle='--')
# plt.title(f'Slice of q at y = {slice_idx}')
# plt.xlabel('x')
# plt.ylabel('q')
# plt.legend()
# plt.savefig(f"../test_result/Heat_{loss_type}_slice_NUTS.png")