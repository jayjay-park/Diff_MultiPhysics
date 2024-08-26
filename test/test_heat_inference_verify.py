import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Simulation function
def solve_heat_equation(k, q, nx=50, ny=50, num_iterations=1000):
    dx = dy = 1.0 / (nx - 1)
    T = torch.zeros((nx, ny), device=k.device)  # Initialize with boundary temperature
    T[0, :] = T[-1, :] = T[:, 0] = T[:, -1] = 0
    
    for _ in range(num_iterations):
        T_old = T.clone()
        T[1:-1, 1:-1] = (
            k[1:-1, 1:-1] * (T_old[2:, 1:-1] / k[2:, 1:-1] + T_old[:-2, 1:-1] / k[:-2:, 1:-1] + 
                             T_old[1:-1, 2:] / k[1:-1, 2:] + T_old[1:-1, :-2] / k[1:-1, :-2])
            + dx * dy * q[1:-1, 1:-1]
        ) / (k[1:-1, 1:-1] * (1/k[2:, 1:-1] + 1/k[:-2:, 1:-1] + 1/k[1:-1, 2:] + 1/k[1:-1, :-2]))
        
        # Boundary conditions (Dirichlet)
        T[0, :] = T[-1, :] = T[:, 0] = T[:, -1] = 0
    
    return T

# Generate synthetic data
nx, ny = 50, 50
true_q = torch.randn(nx, ny, device=device)
true_k = torch.ones((nx, ny), device=device)
true_T = solve_heat_equation(true_k, true_q)

# Add noise to create observed data
noise_std = 0.01
observed_T = true_T + noise_std * torch.randn_like(true_T)

# Pyro model: probabilistic model
def model(observed=None):
    # Prior for q
    q = pyro.sample("q", dist.Normal(torch.zeros(nx, ny, device=device), torch.ones(nx, ny, device=device)).to_event(2))
    
    # Forward model using the true function
    T = solve_heat_equation(true_k, q)
    
    # Likelihood
    pyro.sample("obs", dist.Normal(T, noise_std * torch.ones_like(T)).to_event(2), obs=observed)
    
    return T

# Pyro guide (variational distribution)
def guide(observed=None):
    q_loc = pyro.param("q_loc", torch.zeros(nx, ny, device=device))
    q_scale = pyro.param("q_scale", torch.ones(nx, ny, device=device),
                         constraint=dist.constraints.positive)
    return pyro.sample("q", dist.Normal(q_loc, q_scale).to_event(2))

# Set up the variational inference
pyro.clear_param_store()
adam = pyro.optim.Adam({"lr": 0.01})
svi = SVI(model, guide, adam, loss=Trace_ELBO(retain_graph=True))

# Run inference
num_iterations = 1000
for i in range(num_iterations):
    loss = svi.step(observed_T)
    if (i+1) % 100 == 0:
        print(f"Iteration {i+1}/{num_iterations} - Loss: {loss}")

# Generate multiple samples from the posterior
num_samples = 1000
posterior_samples = []
for _ in range(num_samples):
    sample = guide(observed_T)
    posterior_samples.append(sample)

posterior_samples = torch.stack(posterior_samples)

# Get the inferred q
inferred_q_loc = pyro.param("q_loc").detach()
inferred_q = inferred_q_loc

# Compute mean and standard deviation
q_mean = posterior_samples.mean(dim=0)
q_std = posterior_samples.std(dim=0)

# Compute 95% credible interval
q_lower, q_upper = torch.quantile(posterior_samples, torch.tensor([0.025, 0.975], device=device), dim=0)
loss_type = "Verify"
fig, axes = plt.subplots(2, 2, figsize=(16, 16))
plt.rcParams.update({'font.size': 14})

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
plt.savefig(f"../test_result/Heat_{loss_type}_posterior.png")

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
plt.savefig(f"../test_result/Heat_{loss_type}_slice.png")

# Posterior vs. Prior verification
prior_q_mean = torch.zeros_like(q_mean)
prior_q_std = torch.ones_like(q_std)

# Compute difference between posterior and prior
diff_mean = (q_mean - prior_q_mean).abs().mean().item()
diff_std = (q_std - prior_q_std).abs().mean().item()

print(f"Mean difference between posterior and prior: {diff_mean}")
print(f"Std deviation difference between posterior and prior: {diff_std}")
