import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import matplotlib.pyplot as plt
from modulus.models.fno import FNO
from generate_NS_org import NavierStokesSimulator

# Step 1: Set up the environment
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
print(f"Using device: {device}")

# Step 2: Define simulation parameters
N = 64  # grid size
L = 1.0  # domain size
dt = 0.001  # time step
nu = 1e-3  # viscosity
n_steps = 20  # number of simulation steps
loss_type = "JAC"

# Step 3: Initialize NavierStokesSimulator and FNO model
simulator = NavierStokesSimulator(N, L, dt, nu).to(device)
xlin = torch.linspace(0, L, N, device=device)
xx, yy = torch.meshgrid(xlin, xlin, indexing='ij')

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

# Step 4: Load trained FNO model
FNO_path = f"../test_result/best_model_FNO_NS_{loss_type}.pth"
fno.load_state_dict(torch.load(FNO_path))
fno.eval()

# Step 5: Generate synthetic data for multiple time steps
def generate_synthetic_data(n_steps):
    freq_x = torch.normal(mean=4.0, std=0.3, size=(1,), device=device).item()
    freq_y = torch.normal(mean=2.0, std=0.5, size=(1,), device=device).item()
    phase_x = torch.normal(mean=0.0, std=1., size=(1,), device=device).item()
    phase_y = torch.normal(mean=0.0, std=1., size=(1,), device=device).item()

    vx = -torch.sin(freq_y * torch.pi * simulator.yy + phase_x)
    vy = torch.sin(freq_x * torch.pi * simulator.xx + phase_y)
    vx_true = vx.detach()
    vy_true = vy.detach()
    
    vx_steps = [vx]
    vy_steps = [vy]
    vx_true_steps = [vx_true]
    vy_true_steps = [vy_true]
    
    for _ in range(n_steps - 1):
        input_field = torch.stack([vx, vy], dim=0).unsqueeze(0)
        output_field = fno(input_field).squeeze(0)
        vx, vy = output_field[0], output_field[1]
        vx_true, vy_true = simulator(vx_true, vy_true)
        vx_steps.append(vx)
        vy_steps.append(vy)
        vx_true_steps.append(vx_true)
        vy_true_steps.append(vy_true)
    
    return torch.stack(vx_true_steps), torch.stack(vy_true_steps), torch.stack(vx_steps), torch.stack(vy_steps), freq_x, freq_y, phase_x, phase_y

# Generate true data
true_vx, true_vy, learned_vx, learned_vy, true_freq_x, true_freq_y, true_phase_x, true_phase_y = generate_synthetic_data(n_steps)

# Step 6: Add noise to create observed data
noise_std = 0.05
# observed_vx = true_vx + noise_std * torch.randn_like(true_vx)
# observed_vy = true_vy + noise_std * torch.randn_like(true_vy)
observed_vx = learned_vx
observed_vy = learned_vy

# Step 7: Define Pyro model (probabilistic model)
def model(observed_vx=None, observed_vy=None):
    freq_x = pyro.sample("freq_x", dist.Normal(4.0, 0.3))
    freq_y = pyro.sample("freq_y", dist.Normal(2.0, 0.5))
    phase_x = pyro.sample("phase_x", dist.Normal(0.0, 1.0))
    phase_y = pyro.sample("phase_y", dist.Normal(0.0, 1.0))
    
    vx = -torch.sin(freq_y * torch.pi * simulator.yy + phase_y)
    vy = torch.sin(freq_x * torch.pi * simulator.xx + phase_x)
    
    for t in range(n_steps):
        if t > 0:
            input_field = torch.stack([vx, vy]).unsqueeze(0)
            output_field = fno(input_field).squeeze(0)
            vx, vy = output_field[0], output_field[1]
        
        pyro.sample(f"obs_vx_{t}", dist.Normal(vx, noise_std * torch.ones_like(vx)).to_event(2), obs=observed_vx[t])
        pyro.sample(f"obs_vy_{t}", dist.Normal(vy, noise_std * torch.ones_like(vy)).to_event(2), obs=observed_vy[t])
    
    return vx, vy

# Step 8: Define Pyro guide (variational distribution)
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

# Step 9: Set up SVI
optimizer = Adam({"lr": 0.01})
svi = SVI(model, guide, optimizer, loss=Trace_ELBO(retain_graph=True))

# Step 10: Run SVI
num_iterations = 500
for i in range(num_iterations):
    loss = svi.step(observed_vx, observed_vy)
    if i % 100 == 0:
        print(f"Iteration {i}/{num_iterations} : Loss = {loss}")

# Step 11: Get the learned parameters
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

# Step 12: Generate samples from the learned variational distribution
num_samples = 1000
posterior_freq_x = dist.Normal(freq_x_loc, freq_x_scale).sample((num_samples,))
posterior_freq_y = dist.Normal(freq_y_loc, freq_y_scale).sample((num_samples,))
posterior_phase_x = dist.Normal(phase_x_loc, phase_x_scale).sample((num_samples,))
posterior_phase_y = dist.Normal(phase_y_loc, phase_y_scale).sample((num_samples,))

# Step 13: Generate velocity and vorticity samples based on inferred parameters for all time steps
vx_samples = []
vy_samples = []
wz_samples = []
for fx, fy, px, py in zip(posterior_freq_x, posterior_freq_y, posterior_phase_x, posterior_phase_y):
    vx = -torch.sin(fy * torch.pi * simulator.yy + py)
    vy = torch.sin(fx * torch.pi * simulator.xx + px)
    
    vx_steps = []
    vy_steps = []
    wz_steps = []
    for _ in range(n_steps):
        wz = simulator.curl(vx, vy)
        vx_steps.append(vx.detach().cpu())
        vy_steps.append(vy.detach().cpu())
        wz_steps.append(wz.detach().cpu())
        
        input_field = torch.stack([vx, vy]).unsqueeze(0)
        output_field = fno(input_field).squeeze(0)
        vx, vy = output_field[0], output_field[1]
    
    vx_samples.append(torch.stack(vx_steps))
    vy_samples.append(torch.stack(vy_steps))
    wz_samples.append(torch.stack(wz_steps))

vx_samples = torch.stack(vx_samples)
vy_samples = torch.stack(vy_samples)
wz_samples = torch.stack(wz_samples)
print("samples shape:", vx_samples.shape)

# Step 14: Compute mean and variance of velocities and vorticity for each time step
vx_mean = vx_samples.mean(dim=0)
vx_std = vx_samples.std(dim=0)
vy_mean = vy_samples.mean(dim=0)
vy_std = vy_samples.std(dim=0)
wz_mean = wz_samples.mean(dim=0)
wz_std = wz_samples.std(dim=0)

# Step 15: Visualize the results for selected time steps
time_steps_to_plot = [0, 9, 19]  # Plot initial, middle, and final time steps
fig, axs = plt.subplots(len(time_steps_to_plot), 7, figsize=(25, 7*len(time_steps_to_plot)))

for i, t in enumerate(time_steps_to_plot):
    true_wz = simulator.curl(true_vx[t], true_vy[t])
    
    # Plot true vorticity
    im0 = axs[i, 0].imshow(true_wz.detach().cpu().numpy(), cmap='RdBu_r')
    axs[i, 0].set_title(f'True Vorticity (t={t})')
    plt.colorbar(im0, ax=axs[i, 0], fraction=0.046, pad=0.04)

    # Plot mean predicted vorticity
    im1 = axs[i, 1].imshow(wz_mean[t].detach().cpu().numpy(), cmap='RdBu_r')
    axs[i, 1].set_title(f'Mean Predicted Vorticity (t={t})')
    plt.colorbar(im1, ax=axs[i, 1], fraction=0.046, pad=0.04)

    # Plot standard deviation of predicted vorticity
    im2 = axs[i, 2].imshow(wz_std[t].detach().cpu().numpy(), cmap='viridis')
    axs[i, 2].set_title(f'Std Dev of Predicted Vorticity (t={t})')
    plt.colorbar(im2, ax=axs[i, 2], fraction=0.046, pad=0.04)

    # Plot mean predicted x-velocity
    im3 = axs[i, 3].imshow(vx_mean[t].detach().cpu().numpy(), cmap='RdBu_r')
    axs[i, 3].set_title(f'Mean Predicted Vx (t={t})')
    plt.colorbar(im3, ax=axs[i, 3], fraction=0.046, pad=0.04)

    # Plot mean predicted y-velocity
    im4 = axs[i, 4].imshow(vy_mean[t].detach().cpu().numpy(), cmap='RdBu_r')
    axs[i, 4].set_title(f'Mean Predicted Vy (t={t})')
    plt.colorbar(im4, ax=axs[i, 4], fraction=0.046, pad=0.04)

    # Plot mean true x-velocity
    im5 = axs[i, 5].imshow(true_vx[t].detach().cpu().numpy(), cmap='RdBu_r')
    axs[i, 5].set_title(f'True Vx (t={t})')
    plt.colorbar(im5, ax=axs[i, 5], fraction=0.046, pad=0.04)

    # Plot mean true x-velocity
    im6 = axs[i, 6].imshow(true_vy[t].detach().cpu().numpy(), cmap='RdBu_r')
    axs[i, 6].set_title(f'True Vx (t={t})')
    plt.colorbar(im6, ax=axs[i, 6], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig(f"../test_result/Posterior_Velocities_Vorticity_MultiStep_{loss_type}_org.png")

# Step 16: Compute and print error metrics for each time step
for t in range(n_steps):
    true_wz = simulator.curl(true_vx[t], true_vy[t])
    mae_wz = torch.abs(true_wz.detach().cpu() - wz_mean[t].detach().cpu())
    mae_vx = torch.abs(true_vx[t].detach().cpu() - vx_mean[t].detach().cpu())
    mae_vy = torch.abs(true_vy[t].detach().cpu() - vy_mean[t].detach().cpu())
    print(f"Time step {t}:")
    print(f"  Sum of Absolute Error of Vorticity: {mae_wz.sum().item():.4f}")
    print(f"  Sum of Absolute Error of Vx: {mae_vx.sum().item():.4f}")
    print(f"  Sum of Absolute Error of Vy: {mae_vy.sum().item():.4f}")