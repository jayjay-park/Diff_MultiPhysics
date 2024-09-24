import torch
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS
import matplotlib.pyplot as plt
from modulus.models.fno import FNO
import seaborn as sns
import math
from PINO_NS import *

# Set up the device and random seed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
pyro.set_rng_seed(0)
print(f"Using device: {device}")

# Parameters
nx, ny = 100, 100  # Grid size
grid_size = 7  # Sparse observation grid size
T = 50  # Time at which vorticity is observed
noise_std = 0.05  # Noise standard deviation
L1, L2 = 2 * math.pi, 2 * math.pi  # Domain size
Re = 1000  # Reynolds number
time_step = 0.001
loss_type = "JAC"

# Define forcing function
t = torch.linspace(0, 1, nx + 1, device=device)[:-1]
X, Y = torch.meshgrid(t, t, indexing='ij')
forcing = 0.1 * (torch.sin(2 * math.pi * (X + Y)) + torch.cos(2 * math.pi * (X + Y)))

# Initialize FNO model
fno = FNO(
    in_channels=1,
    out_channels=1,
    decoder_layer_size=128,
    num_fno_layers=6,
    num_fno_modes=20,
    padding=3,
    dimension=2,
    latent_channels=64
).to(device)

# Load pre-trained FNO model
FNO_path = f"../test_result/best_model_FNO_NS_vort_{loss_type}.pth"
fno.load_state_dict(torch.load(FNO_path))
fno.eval()

# Initialize traditional Navier-Stokes solver
ns_solver = NavierStokes2d(nx, ny, L1=L1, L2=L2, device=device)

def simulate_vorticity(w_current, solver, num_steps=50):
    vorticity_data = []
    num_steps = num_steps * 10
    for _ in range(int(num_steps)):
        if solver == fno:
            w_current = solver(w_current.reshape(1, 1, nx, ny))
        else:
            w_current = solver(w_current, f=forcing, T=time_step, Re=Re)
        # vorticity_data.append(w_current.detach().cpu().squeeze())
    return w_current

# Generate true initial vorticity and observations
true_initial_vorticity = gaussian_random_field_2d((nx, ny), 20, random_seed=89).cuda().float()
true_vorticity_T50 = simulate_vorticity(true_initial_vorticity, ns_solver)
fno_vorticity_T50 = simulate_vorticity(true_initial_vorticity, fno)

print("after shape: ", true_vorticity_T50.shape)

# Define sparse observation grid
x_indices = torch.linspace(5, nx - 6, grid_size).long()
y_indices = torch.linspace(5, ny - 6, grid_size).long()
observation_grid = torch.cartesian_prod(x_indices, y_indices)

# Generate noisy observations
observed_vorticity = true_vorticity_T50[observation_grid[:, 0], observation_grid[:, 1]]
observed_vorticity_noisy = observed_vorticity + noise_std * torch.randn_like(observed_vorticity)
print("observed", observed_vorticity.shape)

# Update the Pyro model
def model(observed_data, grid, solver):
    # Parameters for multivariate normal distribution
    mean = pyro.param("mean", torch.zeros(1, device=device))
    log_sigma = pyro.param("log_sigma", torch.zeros(1, device=device))
    
    # sigma = torch.exp(log_sigma)
    
    # Sample initial vorticity from multivariate normal with identity covariance
    initial_vorticity = pyro.sample("initial_vorticity", 
                                    dist.MultivariateNormal(mean.expand(nx*nx), 
                                                            scale_tril=torch.eye(nx*nx, device=device)))
    initial_vorticity = initial_vorticity.reshape(nx, nx)
    
    # Forward simulation
    simulated_vorticity = simulate_vorticity(initial_vorticity, solver)
    
    # Observe the noisy sparse points
    pyro.sample("obs", dist.Normal(simulated_vorticity[grid[:, 0], grid[:, 1]], noise_std),
                obs=observed_data)


def run_mcmc(observed_data, grid, solver, num_samples=30, warmup_steps=0):
    nuts_kernel = NUTS(model)
    mcmc = MCMC(nuts_kernel, num_samples=num_samples, warmup_steps=warmup_steps)
    mcmc.run(observed_data, grid, solver)
    return mcmc.get_samples()

# Run MCMC for both solvers
print("Running MCMC with traditional solver...")
traditional_samples = run_mcmc(observed_vorticity_noisy, observation_grid, ns_solver)
print("Running MCMC with FNO...")
fno_samples = run_mcmc(observed_vorticity_noisy.float(), observation_grid, fno)

# Compute posterior means
traditional_posterior_mean = traditional_samples['initial_vorticity'].mean(dim=0).reshape(nx, ny)
fno_posterior_mean = fno_samples['initial_vorticity'].mean(dim=0).reshape(nx, ny)

# Simulate vorticity at T=50 using posterior means
vorticity_T50_traditional = simulate_vorticity(traditional_posterior_mean, ns_solver)
vorticity_T50_fno = simulate_vorticity(fno_posterior_mean, fno)

# Function to print learned parameters
def print_learned_params(samples):
    print("Learned MVN parameters:")
    print(f"Mean: {samples['mean'].mean().item():.4f}")
    print(f"Sigma: {samples['sigma'].mean().item():.4f}")
    print(f"Length scale: {samples['length_scale'].mean().item():.4f}")

# Print learned parameters
# print("Traditional solver results:")
# print_learned_params(traditional_samples)
# print("\nFNO results:")
# print_learned_params(fno_samples)

# Visualization
plt.figure(figsize=(15, 10))

def plot_vorticity(ax, data, title):
    im = ax.imshow(data.cpu().numpy(), cmap='coolwarm', origin='lower')
    ax.set_title(title)
    return im

# True Initial Vorticity
ax1 = plt.subplot(2, 3, 1)
im1 = plot_vorticity(ax1, true_initial_vorticity, "True Initial Vorticity")
plt.colorbar(im1, ax=ax1)

# Posterior Mean (Traditional Solver)
ax2 = plt.subplot(2, 3, 2)
im2 = plot_vorticity(ax2, traditional_posterior_mean, "Posterior Mean (Traditional Solver)")
plt.colorbar(im2, ax=ax2)

# Posterior Mean (FNO)
ax3 = plt.subplot(2, 3, 3)
im3 = plot_vorticity(ax3, fno_posterior_mean, "Posterior Mean (FNO)")
plt.colorbar(im3, ax=ax3)

# Observed Vorticity at T=50
ax4 = plt.subplot(2, 3, 4)
im4 = plot_vorticity(ax4, true_vorticity_T50, "Observed Vorticity at T=50")
ax4.scatter(observation_grid[:, 1], observation_grid[:, 0], color='black', s=20)
plt.colorbar(im4, ax=ax4)

# Vorticity at T=50 (Traditional Solver)
ax5 = plt.subplot(2, 3, 5)
im5 = plot_vorticity(ax5, true_vorticity_T50, "Vorticity at T=50 (Traditional Solver)")
plt.colorbar(im5, ax=ax5)

# Vorticity at T=50 (FNO)
ax6 = plt.subplot(2, 3, 6)
im6 = plot_vorticity(ax6, fno_vorticity_T50, "Vorticity at T=50 (FNO)")
plt.colorbar(im6, ax=ax6)

plt.tight_layout()
plt.savefig(f"NS_NUTS_grf_{loss_type}")