import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import matplotlib.pyplot as plt
from modulus.models.fno import FNO
from test_heat import create_q_function, solve_heat_equation, create_q_function_inference

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Generate synthetic data
nx, ny = 60, 60
x = torch.linspace(0, 1, nx, device=device)
y = torch.linspace(0, 1, ny, device=device)
X, Y = torch.meshgrid(x, y, indexing='ij')

true_q = 3000 * (torch.sin(5. * X) * torch.sin(3. * Y) + torch.cos(3. * Y) * torch.cos(2 * X))
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
if loss_type != "TRUE":
    FNO_path = f"../test_result/best_model_FNO_Heat_{loss_type}.pth"
    trainedFNO.load_state_dict(torch.load(FNO_path))
    trainedFNO.eval()
    # Generate predicted temperature using FNO
    pred_T = trainedFNO(true_q.unsqueeze(dim=0).unsqueeze(dim=1).float()).squeeze()
else:
    pred_T = true_T

# Add noise to create observed data
noise_std = 0.1
observed_T = pred_T + noise_std * torch.randn_like(pred_T)

# Pyro model: probabilistic model
def model(observed=None):
    freq_x = pyro.sample("freq_x", dist.Normal(torch.tensor(5.0, device=device), torch.tensor(1.0, device=device)))
    freq_y = pyro.sample("freq_y", dist.Normal(torch.tensor(3.0, device=device), torch.tensor(1.0, device=device)))

    x = torch.linspace(0, 1, nx, device=device)
    y = torch.linspace(0, 1, ny, device=device)
    X, Y = torch.meshgrid(x, y, indexing='ij')

    q = 3000 * (torch.sin(freq_x * X) * torch.sin(freq_y * Y) + torch.cos(freq_y * Y) * torch.cos(2 * X))
    
    T = solve_heat_equation(true_k, q, nx, ny)
    
    pyro.sample("obs", dist.Normal(T, noise_std * torch.ones_like(T)).to_event(2), obs=observed)

    return T

# Pyro guide: variational distribution
def guide(observed=None):
    freq_x_loc = pyro.param("freq_x_loc", torch.tensor(5.0, device=device))
    freq_x_scale = pyro.param("freq_x_scale", torch.tensor(1., device=device), constraint=dist.constraints.positive)
    freq_y_loc = pyro.param("freq_y_loc", torch.tensor(3.0, device=device))
    freq_y_scale = pyro.param("freq_y_scale", torch.tensor(1., device=device), constraint=dist.constraints.positive)

    pyro.sample("freq_x", dist.Normal(freq_x_loc, freq_x_scale))
    pyro.sample("freq_y", dist.Normal(freq_y_loc, freq_y_scale))

# Set up SVI
optimizer = Adam({"lr": 0.01})
svi = SVI(model, guide, optimizer, loss=Trace_ELBO(retain_graph=True))

# Run SVI
num_iterations = 1000
for i in range(num_iterations):
    loss = svi.step(observed_T)
    if i % 100 == 0:
        print(f"Iteration {i}/{num_iterations} : Loss = {loss}")

# Get the learned parameters
freq_x_loc = pyro.param("freq_x_loc").item()
freq_x_scale = pyro.param("freq_x_scale").item()
freq_y_loc = pyro.param("freq_y_loc").item()
freq_y_scale = pyro.param("freq_y_scale").item()

print(f"Estimated freq_x: Mean = {freq_x_loc}, Std = {freq_x_scale}")
print(f"Estimated freq_y: Mean = {freq_y_loc}, Std = {freq_y_scale}")
# JAC
# Estimated freq_x: Mean = 4.99452018737793, Std = 0.017078550532460213
# Estimated freq_y: Mean = 3.0442113876342773, Std = 0.019067300483584404
# Stiff (600 training data)
# Estimated freq_x: Mean = 4.107088565826416, Std = 0.028546450659632683
# Estimated freq_y: Mean = 2.643915891647339, Std = 0.024336788803339005

# True
# Estimated freq_x: Mean = 4.996736526489258, Std = 0.017203284427523613
# Estimated freq_y: Mean = 2.9999308586120605, Std = 0.015644129365682602
# Stiff
# Estimated freq_x: Mean = 5.007697582244873, Std = 0.014010228216648102
# Estimated freq_y: Mean = 3.0004279613494873, Std = 0.01581578701734543


# MSE
# Estimated freq_x: Mean = 4.97986364364624, Std = 0.016950080171227455
# Estimated freq_y: Mean = 3.141986131668091, Std = 0.01716192439198494
# Stiff (100 training data)
# Estimated freq_x: Mean = 3.8380191326141357, Std = 0.034099020063877106
# Estimated freq_y: Mean = 2.4467694759368896, Std = 0.027377523481845856
# Stiff (600 trainig data)
# Estimated freq_x: Mean = 4.073402404785156, Std = 0.028732210397720337
# Estimated freq_y: Mean = 2.5611820220947266, Std = 0.023386692628264427

# Generate samples from the learned variational distribution
num_samples = 10000
posterior_freq_x = dist.Normal(freq_x_loc, freq_x_scale).sample((num_samples,))
posterior_freq_y = dist.Normal(freq_y_loc, freq_y_scale).sample((num_samples,))

# Plot histograms for freq_x and freq_y
plt.figure(figsize=(12, 6))

# Histogram for freq_x
plt.subplot(1, 2, 1)
plt.hist(posterior_freq_x.detach().cpu().numpy(), bins=100, density=True, color='blue', alpha=0.6)
plt.axvline(freq_x_loc, color='red', linestyle='--', label=f'Mean: {freq_x_loc:.2f}')
plt.title("Posterior Distribution of freq_x")
plt.xlabel("freq_x")
plt.ylabel("Density")
plt.legend()

# Histogram for freq_y
plt.subplot(1, 2, 2)
plt.hist(posterior_freq_y.detach().cpu().numpy(), bins=100, density=True, color='green', alpha=0.6)
plt.axvline(freq_y_loc, color='red', linestyle='--', label=f'Mean: {freq_y_loc:.2f}')
plt.title("Posterior Distribution of freq_y")
plt.xlabel("freq_y")
plt.ylabel("Density")
plt.legend()

plt.tight_layout()
plt.savefig(f"../test_result/Posterior_Distributions_SVI_{loss_type}.png")



# Compute 95% credible intervals
freq_x_lower, freq_x_upper = torch.quantile(posterior_freq_x, torch.tensor([0.025, 0.975]))
freq_y_lower, freq_y_upper = torch.quantile(posterior_freq_y, torch.tensor([0.025, 0.975]))

# Plot credible intervals
plt.figure(figsize=(12, 6))

# Credible interval for freq_x
plt.subplot(1, 2, 1)
plt.errorbar(1, freq_x_loc, yerr=[[freq_x_loc - freq_x_lower.item()], [freq_x_upper.item() - freq_x_loc]], fmt='o', color='blue', capsize=5, label='95% CI')
plt.title("95% Credible Interval for freq_x")
plt.xlabel("freq_x")
plt.ylabel("Value")
plt.legend()
plt.xticks([])

# Credible interval for freq_y
plt.subplot(1, 2, 2)
plt.errorbar(1, freq_y_loc, yerr=[[freq_y_loc - freq_y_lower.item()], [freq_y_upper.item() - freq_y_loc]], fmt='o', color='green', capsize=5, label='95% CI')
plt.title("95% Credible Interval for freq_y")
plt.xlabel("freq_y")
plt.ylabel("Value")
plt.legend()
plt.xticks([])

plt.tight_layout()
plt.savefig(f"../test_result/Credible_Intervals_SVI_{loss_type}.png")



# Generate q samples based on inferred freq_x and freq_y
q_samples = []
for fx, fy in zip(posterior_freq_x.detach().cpu(), posterior_freq_y.detach().cpu()):
    q_samples.append(create_q_function_inference(nx, ny, fx, fy, noise_level=0.0))
q_samples = torch.stack(q_samples)

# Compute mean and variance of q
q_mean = q_samples.mean(dim=0)
q_std = q_samples.std(dim=0)

# Visualization
fig, axes = plt.subplots(1, 4, figsize=(18, 6))
plt.rcParams.update({'font.size': 16})

# True q
im0 = axes[0].imshow(true_q.detach().cpu().numpy(), cmap='inferno')
axes[0].set_title(r"True $q$")
fig.colorbar(im0, ax=axes[0], fraction=0.045, pad=0.06)

# Mean of q
im1 = axes[1].imshow(q_mean.detach().cpu().numpy(), cmap='inferno')
axes[1].set_title(r"Mean of Estimated $q$")
fig.colorbar(im1, ax=axes[1], fraction=0.045, pad=0.06)

# Variance of q
im2 = axes[2].imshow(q_std.detach().cpu().numpy(), cmap='PuRd')
axes[2].set_title(r"Variance of Estimated $q$")
fig.colorbar(im2, ax=axes[2], fraction=0.045, pad=0.06)

# Error between true q and mean q
error = true_q.detach().cpu() - q_mean.detach().cpu()
im3 = axes[3].imshow(error.numpy(), cmap='bwr')
axes[3].set_title(r"Error in $q$")
fig.colorbar(im3, ax=axes[3], fraction=0.045, pad=0.06)

plt.tight_layout()
plt.savefig(f"../plot/Heat_plot/q_mean_variance_SVI_{loss_type}.png")