import torch
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS
import matplotlib.pyplot as plt
from modulus.models.fno import FNO
from generate_NS_org import *
import seaborn as sns
import math
from PINO_NS import *

def generate_dataset(num_samples, loss_type, num_init, time_step, nx=50, ny=50):
    input, output = [], []

    L1, L2 = 2 * math.pi, 2 * math.pi  # Domain size
    Re = 1000  # Reynolds number
    # Define a forcing function (or set to None)
    t = torch.linspace(0, 1, nx + 1, device="cuda")
    t = t[0:-1]
    X, Y = torch.meshgrid(t, t)
    forcing = 0.1 * (torch.sin(2 * math.pi * (X + Y)) + torch.cos(2 * math.pi * (X + Y)))

    # Initialize Navier-Stokes solver
    if loss_type != "True":
        ns_solver = fno
    else:
        ns_solver = NavierStokes2d(nx, ny, L1=L1, L2=L2, device="cuda")

    num_iter = int(num_samples / num_init)
    print("num_init: ", num_init)
    print("time step: ", num_iter)

    for s in range(num_init):
        print("gen data for init: ", s)

        # Generate initial vorticity field
        random_seed = 20 + s
        w = gaussian_random_field_2d((nx, ny), 20, random_seed)
        w_current = w.cuda()
        vorticity_data = [w_current.cpu().numpy()]

        # Solve the NS
        for i in range(num_iter):
            if loss_type != "True":
                w_current = fno(w_current.reshape(1, 1, N, N).float())
            else:
                w_current = ns_solver(w_current, f=forcing, T=time_step, Re=Re)
            vorticity_data.append(w_current.detach().cpu().numpy())

        input.append(vorticity_data[:-1])
        output.append(vorticity_data[1:])

        if s == 0:
            plot_vorticity(vorticity_data[0], s, title=f"Vorticity Field at Time Step {i + 1}")
        elif s == 1:
            plot_vorticity(vorticity_data[0], s, title=f"Vorticity Field at Time Step {i + 1}")

    return input, output

# Set up the device and random seed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
print(f"Using device: {device}")

# Define simulation parameters
N = 100  # grid size
L = 2 * math.pi  # domain size
time_step = 0.001  # time step
nu = 1e-3  # viscosity
num_init = 100
n_steps = 1  # number of simulation steps
loss_type = "JAC"
num_samples = 1000  # Number of vorticity fields for MCMC

# Initialize FNO
fno = FNO(
    in_channels=1,  # input vorticity
    out_channels=1,  # predicted vorticity at next time step
    decoder_layer_size=128,
    num_fno_layers=6,
    num_fno_modes=20,
    padding=3,
    dimension=2,
    latent_channels=64
).to(device)

if loss_type != "TRUE":
    FNO_path = f"../test_result/best_model_FNO_NS_vort_{loss_type}.pth"
    fno.load_state_dict(torch.load(FNO_path))
    fno.eval()
    input_vorticities, output_vorticities = generate_dataset(num_samples, loss_type, num_init, time_step, nx=N, ny=N)
    output_vorticities = torch.tensor(output_vorticities).cuda()
else:
    input_vorticities, output_vorticities = generate_dataset(num_samples, loss_type, num_init, time_step, nx=N, ny=N)

# Define grid size and scale
scale = 20.0
cov_matrix = gaussian_random_field_2d((N, N), scale=scale, random_seed=None)
vorticity_mean = torch.zeros(N, N, device=device)

# Define the Pyro model for NUTS
def model(output_vorticities):
    # vorticity_prior = latent variable sampled from initial belief
    vorticity_prior = pyro.sample('vorticity_prior', dist.MultivariateNormal(vorticity_mean, covariance_matrix=torch.eye(N, device='cuda')))
    vorticity_prior_reshaped = vorticity_prior.view(1, 1, N, N).cuda()
    
    # Prediction from FNO (vorticity prediction at the next time step)
    predicted_vorticity = fno(vorticity_prior_reshaped).squeeze(0).squeeze(0)
    
    # Observation model: comparing predicted vorticity with observed vorticity
    for i in range(len(output_vorticities)):
        observed_vorticity_flat = output_vorticities[i]
        pyro.sample(
            f'obs_{i}', 
            dist.MultivariateNormal(predicted_vorticity, 0.1 * torch.eye(N, device=device)), 
            obs=observed_vorticity_flat
        )

# Setup NUTS and MCMC
nuts_kernel = NUTS(model)
mcmc = MCMC(nuts_kernel, num_samples=10, warmup_steps=0, num_chains=1)
mcmc.run(output_vorticities)

# Get posterior samples
samples = mcmc.get_samples()
vorticity_posterior = samples['vorticity_prior']

print(f"Posterior samples for vorticity: {vorticity_posterior}")



'''
plot
'''

# True covariance matrix (e.g., Identity for simplicity)
true_covariance = torch.eye(N * N)

# After MCMC inference, extract posterior samples for the learned vorticities
# Assuming mcmc has completed and we can extract samples
posterior_samples = mcmc.get_samples()['vorticity_prior']  # Posterior samples from MCMC

# Reshape posterior samples if necessary
posterior_samples_flat = posterior_samples.view(posterior_samples.shape[0], -1)  # Flatten each sample

# Estimate learned covariance matrix (empirical covariance from posterior samples)
mean_estimate = torch.mean(posterior_samples_flat, dim=0)
centered_samples = posterior_samples_flat - mean_estimate
learned_covariance = torch.mm(centered_samples.T, centered_samples) / (posterior_samples_flat.shape[0] - 1)

# Convert to numpy for plotting
true_covariance_np = true_covariance.numpy()
learned_covariance_np = learned_covariance.detach().cpu().numpy()

# Plotting using seaborn for heatmap visualization
plt.figure(figsize=(14, 6))

# Plot True Covariance
plt.subplot(1, 2, 1)
sns.heatmap(true_covariance_np, cmap="viridis")
plt.title("True Covariance")

# Plot Learned Covariance
plt.subplot(1, 2, 2)
sns.heatmap(learned_covariance_np, cmap="viridis")
plt.title("Learned Covariance (Posterior)")

# Show the plots
plt.tight_layout()
plt.show()

difference_covariance = learned_covariance - true_covariance

plt.figure(figsize=(7, 6))
sns.heatmap(difference_covariance.numpy(), cmap="coolwarm")
plt.title("Difference between True and Learned Covariance")
plt.show()

# Assuming your vorticity field is of size (nx, ny)
nx, ny = N, N  # Adjust based on the size of your problem

# Extract prior vorticity sample
prior_vorticity_sample = prior_dist.sample()  # Sample from prior
prior_vorticity_field = prior_vorticity_sample.view(nx, ny)  # Reshape to 2D grid

# Extract posterior samples after MCMC inference (assuming 'vorticity_prior' is the variable of interest)
posterior_vorticity_samples = mcmc.get_samples()['vorticity_prior']  # Shape: (num_samples, nx * ny)
posterior_vorticity_field = posterior_vorticity_samples.mean(dim=0).view(nx, ny)  # Mean posterior vorticity field

# Plotting the vorticity fields
plt.figure(figsize=(12, 6))

# Plot Prior Vorticity Field
plt.subplot(1, 2, 1)
plt.imshow(prior_vorticity_field.numpy(), cmap='coolwarm', origin='lower')
plt.colorbar(label='Vorticity')
plt.title("Prior Vorticity Field")

# Plot Learned Posterior Vorticity Field (Mean)
plt.subplot(1, 2, 2)
plt.imshow(posterior_vorticity_field.numpy(), cmap='coolwarm', origin='lower')
plt.colorbar(label='Vorticity')
plt.title("Posterior Vorticity Field (Mean)")

# Show the plots
plt.tight_layout()
plt.show()