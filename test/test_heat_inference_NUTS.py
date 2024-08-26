import torch
import pyro
import pyro.distributions as dist
from pyro.infer import NUTS, MCMC
import matplotlib.pyplot as plt
from modulus.models.fno import FNO

# TODO: implement MCMC with sparse observation, 7 x 7 grid

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Simulation function (unchanged)
def solve_heat_equation(k, q, nx=50, ny=50, num_iterations=1000):
    dx = dy = 1.0 / (nx - 1)
    T = torch.zeros((nx, ny), device=k.device)  # Initialize with boundary temperature
    T[0, :] = T[-1, :] = T[:, 0] = T[:, -1] = 0
    
    for _ in range(num_iterations):
        T_old = T.clone()
        T[1:-1, 1:-1] = (
            k[1:-1, 1:-1] * (T_old[2:, 1:-1] / k[2:, 1:-1] + T_old[:-2, 1:-1] / k[:-2, 1:-1] + 
                             T_old[1:-1, 2:] / k[1:-1, 2:] + T_old[1:-1, :-2] / k[1:-1, :-2])
            + dx * dy * q[1:-1, 1:-1]  # Changed sign to positive
        ) / (k[1:-1, 1:-1] * (1/k[2:, 1:-1] + 1/k[:-2, 1:-1] + 1/k[1:-1, 2:] + 1/k[1:-1, :-2]))
        
        # Boundary conditions (Dirichlet)
        T[0, :] = T[-1, :] = T[:, 0] = T[:, -1] = 0
    
    return T

# Generate synthetic data
nx, ny = 50, 50
true_q = torch.randn(nx, ny, device=device)
true_k = torch.ones((nx, ny), device=device)
true_T = solve_heat_equation(true_k, true_q)

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
FNO_path = f"../test_result/best_model_FNO_Heat_full epoch_{loss_type}_{nx}.pth"
trainedFNO.load_state_dict(torch.load(FNO_path))
trainedFNO.eval()

# Generate predicted temperature using FNO
pred_T = trainedFNO(true_q.unsqueeze(dim=0).unsqueeze(dim=1).float()).squeeze()

# Add noise to create observed data
noise_std = 0.1
observed_T = pred_T + noise_std * torch.randn_like(pred_T)

# Pyro model: probabilistic model
def model(observed=None):
    # Prior for q
    q = pyro.sample("q", dist.Normal(torch.zeros(nx, ny, device=device), torch.ones(nx, ny, device=device)).to_event(2))
    
    # Forward model using FNO
    T = trainedFNO(q.unsqueeze(dim=0).unsqueeze(dim=1).float()).squeeze()
    
    # Likelihood
    pyro.sample("obs", dist.Normal(T, noise_std * torch.ones_like(T)).to_event(2), obs=observed)
    
    return T

# Set up NUTS sampler
nuts_kernel = NUTS(model, adapt_step_size=True)

# Run MCMC
num_samples = 1000
mcmc = MCMC(nuts_kernel, num_samples=num_samples, warmup_steps=200)
mcmc.run(observed_T)

# Get samples
posterior_samples = mcmc.get_samples()["q"]

# Compute statistics
q_mean = posterior_samples.mean(dim=0)
q_std = posterior_samples.std(dim=0)

# Compute 95% credible interval
q_lower, q_upper = torch.quantile(posterior_samples, torch.tensor([0.025, 0.975]).cuda(), dim=0)

fig, axes = plt.subplots(2, 2, figsize=(16, 16))
plt.rcParams.update({'font.size': 14})

# True q
im0 = axes[0, 0].imshow(true_q.cpu().numpy(), cmap='inferno')
axes[0, 0].set_title(r"True Heat Source $q$")
fig.colorbar(im0, ax=axes[0, 0], fraction=0.045, pad=0.06)

# Inferred q (mean)
im1 = axes[0, 1].imshow(q_mean.detach().cpu().numpy(), cmap='inferno')
axes[0, 1].set_title(r"Inferred Heat Source $q$ (Mean)")
fig.colorbar(im1, ax=axes[0, 1], fraction=0.045, pad=0.06)

# Uncertainty (standard deviation)
im2 = axes[1, 0].imshow(q_std.cpu().detach().numpy(), cmap='viridis')
axes[1, 0].set_title(r"Uncertainty in $q$ (Std Dev)")
fig.colorbar(im2, ax=axes[1, 0], fraction=0.045, pad=0.06)

# Error (difference between true and inferred)
error = (true_q - q_mean).abs()
im3 = axes[1, 1].imshow(error.detach().cpu().numpy(), cmap='viridis')
axes[1, 1].set_title("Absolute Error in $q$")
fig.colorbar(im3, ax=axes[1, 1], fraction=0.045, pad=0.06)

plt.tight_layout()
plt.savefig(f"../test_result/Heat_{loss_type}_posterior_NUTS.png")

# Plot a slice with credible interval
slice_idx = nx // 2
plt.figure(figsize=(12, 6))
x = torch.arange(ny)
plt.fill_between(x.cpu().numpy(), 
                 q_lower[slice_idx].detach().cpu().numpy(), 
                 q_upper[slice_idx].detach().cpu().numpy(), 
                 alpha=0.3, label='95% CI')
plt.plot(x.cpu().numpy(), q_mean[slice_idx].detach().cpu().numpy(), label='Mean', color='r')
plt.plot(x.cpu().numpy(), true_q[slice_idx].detach().cpu().numpy(), label='True', color='k', linestyle='--')
plt.title(f'Slice of q at y = {slice_idx}')
plt.xlabel('x')
plt.ylabel('q')
plt.legend()
plt.savefig(f"../test_result/Heat_{loss_type}_slice_NUTS.png")