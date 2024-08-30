import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import matplotlib.pyplot as plt
from modulus.models.fno import FNO
from generate_NS_org import *


# Set up the device and random seed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
print(f"Using device: {device}")

# Define simulation parameters
N = 64  # grid size
L = 1.0  # domain size
dt = 0.001  # time step
nu = 1e-3  # viscosity
n_steps = 1  # number of simulation steps
loss_type = "MSE"

# Initialize NavierStokesSimulator
simulator = NavierStokesSimulator(N, L, dt, nu).to(device)
xlin = torch.linspace(0, L, N, device='cuda')
xx, yy = torch.meshgrid(xlin, xlin, indexing='ij')
true_vx = -torch.sin(2.0 * torch.pi * yy)
true_vy = torch.sin(4.0 * torch.pi * xx)
true_wz = simulator.curl(true_vx, true_vy)


# Initialize FNO
fno = FNO(
    in_channels=2,  # vx and vy
    out_channels=2,  # predicted vx and vy
    decoder_layer_size=128,
    num_fno_layers=6,
    num_fno_modes=20,
    padding=3,
    dimension=2,
    latent_channels=64
).to(device)

# Generate synthetic data: one step simulator
def generate_synthetic_data():
    freq_x = torch.normal(mean=4.0, std=0.3, size=(1,), device=device).item()
    freq_y = torch.normal(mean=2.0, std=0.5, size=(1,), device=device).item()
    phase_x = torch.normal(mean=0.0, std=1., size=(1,), device=device).item()
    phase_y = torch.normal(mean=0.0, std=1., size=(1,), device=device).item()

    vx = -torch.sin(freq_y * torch.pi * simulator.yy + phase_y)
    vy = torch.sin(freq_x * torch.pi * simulator.xx + phase_x)
    
    # Use FNO for prediction
    if loss_type != "TRUE":
        input_field = torch.stack([vx, vy], dim=0).unsqueeze(0)
        output_field = fno(input_field).squeeze(0)
        vx_final, vy_final = output_field[0], output_field[1]
    else:
        with torch.no_grad():
          vx_final, vy_final = simulator(vx, vy)

    wz_output = simulator.curl(vx_final, vy_final).detach().cpu().numpy()
    
    
    return vx_final, vy_final, wz_output, freq_x, freq_y, phase_x, phase_y

# Load trained FNO model
if loss_type != "TRUE":
    FNO_path = f"../test_result/best_model_FNO_NS_{loss_type}.pth"
    fno.load_state_dict(torch.load(FNO_path))
    fno.eval()
    # Generate true data
    pred_vx, pred_vy, pred_wz, pred_freq_x, pred_freq_y, pred_phase_x, pred_phase_y = generate_synthetic_data()
else:
    pred_vx, pred_vy = true_vx, true_vy
    
true_freq_x = torch.tensor(4.0)
true_freq_y = torch.tensor(2.0)
true_phase_x = torch.tensor(0.)
true_phase_y = torch.tensor(0.)

# Add noise to create observed data
noise_std = 0.05
observed_vx = pred_vx + noise_std * torch.randn_like(pred_vx)
observed_vy = pred_vy + noise_std * torch.randn_like(pred_vy)

# Pyro model: probabilistic model
def model(observed_vx=None, observed_vy=None):
    freq_x = pyro.sample("freq_x", dist.Normal(4.0, 0.3))
    freq_y = pyro.sample("freq_y", dist.Normal(2.0, 0.5))
    phase_x = pyro.sample("phase_x", dist.Normal(0.0, 1.0))
    phase_y = pyro.sample("phase_y", dist.Normal(0.0, 1.0))
    
    vx = -torch.sin(freq_y * torch.pi * simulator.yy + phase_y)
    vy = torch.sin(freq_x * torch.pi * simulator.xx + phase_x)
    
    # Use FNO for prediction
    if loss_type != "TRUE":
        input_field = torch.stack([vx, vy]).unsqueeze(0)
        output_field = fno(input_field).squeeze(0)
        vx, vy = output_field[0], output_field[1]
    else:
        with torch.no_grad():
          vx, vy = simulator(vx, vy)
    
    pyro.sample("obs_vx", dist.Normal(vx, noise_std * torch.ones_like(vx)).to_event(2), obs=observed_vx)
    pyro.sample("obs_vy", dist.Normal(vy, noise_std * torch.ones_like(vy)).to_event(2), obs=observed_vy)
    
    return vx, vy

# Pyro guide: variational distribution
def guide(observed_vx=None, observed_vy=None):
    freq_x_loc = pyro.param("freq_x_loc", torch.tensor(4.0, device=device))
    freq_x_scale = pyro.param("freq_x_scale", torch.tensor(0.3, device=device), constraint=dist.constraints.positive)
    freq_y_loc = pyro.param("freq_y_loc", torch.tensor(2.0, device=device))
    freq_y_scale = pyro.param("freq_y_scale", torch.tensor(0.5, device=device), constraint=dist.constraints.positive)
    phase_x_loc = pyro.param("phase_x_loc", torch.tensor(0.0, device=device))
    phase_x_scale = pyro.param("phase_x_scale", torch.tensor(1.0, device=device), constraint=dist.constraints.positive)
    phase_y_loc = pyro.param("phase_y_loc", torch.tensor(0.0, device=device))
    phase_y_scale = pyro.param("phase_y_scale", torch.tensor(1.0, device=device), constraint=dist.constraints.positive)

    pyro.sample("freq_x", dist.Normal(freq_x_loc, freq_x_scale))
    pyro.sample("freq_y", dist.Normal(freq_y_loc, freq_y_scale))
    pyro.sample("phase_x", dist.Normal(phase_x_loc, phase_x_scale))
    pyro.sample("phase_y", dist.Normal(phase_y_loc, phase_y_scale))

# Set up SVI
optimizer = Adam({"lr": 0.01})
svi = SVI(model, guide, optimizer, loss=Trace_ELBO(retain_graph=True))

# Run SVI
num_iterations = 1000
for i in range(num_iterations):
    loss = svi.step(observed_vx, observed_vy)
    if i % 100 == 0:
        print(f"Iteration {i}/{num_iterations} : Loss = {loss}")

# Get the learned parameters
freq_x_loc = pyro.param("freq_x_loc").item()
freq_x_scale = pyro.param("freq_x_scale").item()
freq_y_loc = pyro.param("freq_y_loc").item()
freq_y_scale = pyro.param("freq_y_scale").item()
phase_x_loc = pyro.param("phase_x_loc").item()
phase_x_scale = pyro.param("phase_x_scale").item()
phase_y_loc = pyro.param("phase_y_loc").item()
phase_y_scale = pyro.param("phase_y_scale").item()

print(f"Estimated freq_x: Mean = {freq_x_loc}, Std = {freq_x_scale}")
print(f"Estimated freq_y: Mean = {freq_y_loc}, Std = {freq_y_scale}")
print(f"Estimated phase_x: Mean = {phase_x_loc}, Std = {phase_x_scale}")
print(f"Estimated phase_y: Mean = {phase_y_loc}, Std = {phase_y_scale}")

# Generate samples from the learned variational distribution
num_samples = 10000
posterior_freq_x = dist.Normal(freq_x_loc, freq_x_scale).sample((num_samples,))
posterior_freq_y = dist.Normal(freq_y_loc, freq_y_scale).sample((num_samples,))
posterior_phase_x = dist.Normal(phase_x_loc, phase_x_scale).sample((num_samples,))
posterior_phase_y = dist.Normal(phase_y_loc, phase_y_scale).sample((num_samples,))

# Plot histograms for all parameters
fig, axes = plt.subplots(2, 2, figsize=(16, 16))
plt.rcParams.update({'font.size': 16})

axes[0, 0].hist(posterior_freq_x.detach().cpu().numpy(), bins=100, density=True, alpha=0.6)
axes[0, 0].axvline(freq_x_loc, color='red', linestyle='--', label=f'Mean: {freq_x_loc:.2f}')
axes[0, 0].axvline(true_freq_x, color='green', linestyle='--', label=f'True: {true_freq_x:.2f}')
axes[0, 0].set_title("Posterior Distribution of freq_x")
axes[0, 0].set_xlabel("freq_x")
axes[0, 0].set_ylabel("Density")
axes[0, 0].legend()

axes[0, 1].hist(posterior_freq_y.detach().cpu().numpy(), bins=100, density=True, alpha=0.6)
axes[0, 1].axvline(freq_y_loc, color='red', linestyle='--', label=f'Mean: {freq_y_loc:.2f}')
axes[0, 1].axvline(true_freq_y, color='green', linestyle='--', label=f'True: {true_freq_y:.2f}')
axes[0, 1].set_title("Posterior Distribution of freq_y")
axes[0, 1].set_xlabel("freq_y")
axes[0, 1].set_ylabel("Density")
axes[0, 1].legend()

axes[1, 0].hist(posterior_phase_x.detach().cpu().numpy(), bins=100, density=True, alpha=0.6)
axes[1, 0].axvline(phase_x_loc, color='red', linestyle='--', label=f'Mean: {phase_x_loc:.2f}')
axes[1, 0].axvline(true_phase_x, color='green', linestyle='--', label=f'True: {true_phase_x:.2f}')
axes[1, 0].set_title("Posterior Distribution of phase_x")
axes[1, 0].set_xlabel("phase_x")
axes[1, 0].set_ylabel("Density")
axes[1, 0].legend()

axes[1, 1].hist(posterior_phase_y.detach().cpu().numpy(), bins=100, density=True, alpha=0.6)
axes[1, 1].axvline(phase_y_loc, color='red', linestyle='--', label=f'Mean: {phase_y_loc:.2f}')
axes[1, 1].axvline(true_phase_y, color='green', linestyle='--', label=f'True: {true_phase_y:.2f}')
axes[1, 1].set_title("Posterior Distribution of phase_y")
axes[1, 1].set_xlabel("phase_y")
axes[1, 1].set_ylabel("Density")
axes[1, 1].legend()

plt.tight_layout()
plt.savefig(f"../test_result/Posterior_Distributions_NS_{loss_type}.png")
plt.close()

# Compute 95% credible intervals
params = ['freq_x', 'freq_y', 'phase_x', 'phase_y']
posteriors = [posterior_freq_x, posterior_freq_y, posterior_phase_x, posterior_phase_y]
true_values = [true_freq_x, true_freq_y, true_phase_x, true_phase_y]
means = [freq_x_loc, freq_y_loc, phase_x_loc, phase_y_loc]

fig, ax = plt.subplots(figsize=(12, 6))

for i, (param, posterior, true_value, mean) in enumerate(zip(params, posteriors, true_values, means)):
    lower, upper = torch.quantile(posterior, torch.tensor([0.025, 0.975]))
    ax.errorbar(i, mean, yerr=[[mean - lower.item()], [upper.item() - mean]], fmt='o', capsize=5, label='95% CI')
    ax.plot(i, true_value, 'r*', markersize=10, label='True value' if i == 0 else '')

ax.set_xticks(range(len(params)))
ax.set_xticklabels(params)
ax.set_ylabel("Value")
ax.set_title("95% Credible Intervals for Navier-Stokes Parameters")
ax.legend()
plt.tight_layout()
plt.savefig(f"../test_result/Credible_Intervals_NS_{loss_type}.png")
plt.close()

# Generate vorticity samples based on inferred parameters
wz_samples = []
num_wz_samples = 2000
i = 0
for fx, fy, px, py in zip(posterior_freq_x[:num_wz_samples], posterior_freq_y[:num_wz_samples], posterior_phase_x[:num_wz_samples], posterior_phase_y[:num_wz_samples]):
    vx = -torch.sin(fy * torch.pi * simulator.yy + py)
    vy = torch.sin(fx * torch.pi * simulator.xx + px)
    
    # Use FNO for prediction
    # input_field = torch.stack([vx, vy]).unsqueeze(0)
    # output_field = fno(input_field).squeeze(0)
    # vx, vy = output_field[0], output_field[1]
    wz = simulator.curl(vx, vy)
    wz_samples.append(wz.detach().cpu())
    i += 1
wz_samples = torch.stack(wz_samples)
print("samples: ", wz_samples.shape)

# Compute mean and variance of vorticity
wz_mean = wz_samples.mean(dim=0)
wz_std = wz_samples.std(dim=0)

# Visualize the results
fig, axs = plt.subplots(2, 2, figsize=(15, 15))

# Plot true vorticity
im0 = axs[0, 0].imshow(true_wz.cpu().numpy(), cmap='RdBu_r')
axs[0, 0].set_title('True Vorticity')
plt.colorbar(im0, ax=axs[0, 0], fraction=0.045, pad=0.06)

# Plot mean predicted vorticity
im1 = axs[0, 1].imshow(wz_mean.cpu().numpy(), cmap='RdBu_r')
axs[0, 1].set_title('Mean Predicted Vorticity')
plt.colorbar(im1, ax=axs[0, 1], fraction=0.045, pad=0.06)

# Plot standard deviation of predicted vorticity
im2 = axs[1, 0].imshow(wz_std.cpu().numpy(), cmap='viridis')
axs[1, 0].set_title('Std Dev of Predicted Vorticity')
plt.colorbar(im2, ax=axs[1, 0], fraction=0.045, pad=0.06)

# Plot absolute difference between true and mean predicted vorticity
diff = torch.abs(true_wz.cpu() - wz_mean.cpu())
im3 = axs[1, 1].imshow(diff.cpu().numpy(), cmap='viridis')
axs[1, 1].set_title('Absolute Difference')
plt.colorbar(im3, ax=axs[1, 1], fraction=0.045, pad=0.06)

plt.tight_layout()
plt.tight_layout()
plt.savefig(f"../test_result/Posterior_Vorticity_{loss_type}.png")


# Compute and print error metrics
mae = torch.abs(true_wz.cpu() - wz_mean.cpu())
print(f"Sum of Absolute Error of Vorticity: {mae.sum().item():.4f}")