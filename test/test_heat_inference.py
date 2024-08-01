import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import matplotlib.pyplot as plt
from modulus.models.fno import FNO

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Simulation function (same as before)
def solve_heat_equation(k, q, nx=50, ny=50, num_iterations=1000):
    dx = dy = 1.0 / (nx - 1)
    T = torch.zeros((nx, ny), device=device)
    
    for _ in range(num_iterations):
        T_old = T.clone()
        T[1:-1, 1:-1] = (
            k[1:-1, 1:-1] * (T_old[2:, 1:-1] / k[2:, 1:-1] + T_old[:-2, 1:-1] / k[:-2, 1:-1] + 
                             T_old[1:-1, 2:] / k[1:-1, 2:] + T_old[1:-1, :-2] / k[1:-1, :-2])
            - dx * dy * q[1:-1, 1:-1]
        ) / (k[1:-1, 1:-1] * (1/k[2:, 1:-1] + 1/k[:-2, 1:-1] + 1/k[1:-1, 2:] + 1/k[1:-1, :-2]))
        
        # Boundary conditions (Dirichlet)
        T[0, :] = T[-1, :] = T[:, 0] = T[:, -1] = 0
    
    return T

# Generate synthetic data
nx, ny = 50, 50
true_k = torch.exp(torch.randn(nx, ny, device=device))
q = torch.ones((nx, ny), device=device) * 100
trainedFNO = FNO(
                in_channels=1,
                out_channels=1,
                num_fno_modes=21,
                padding=3,
                dimension=2,
                latent_channels=64
                ).to('cuda')
loss_type = "MSE"
FNO_path = "../test_result/best_model_FNO_Heat_"+str(loss_type)+".pth"
trainedFNO.load_state_dict(torch.load(FNO_path))
trainedFNO.eval()
# true_T = solve_heat_equation(true_k, q)
print("shape", true_k.unsqueeze(dim=0).unsqueeze(dim=1).float().cuda().shape)
true_T = trainedFNO(true_k.unsqueeze(dim=0).unsqueeze(dim=1).float().cuda()).squeeze()

# Add noise to create observed data
noise_std = 0.1
observed_T = true_T + noise_std * torch.randn_like(true_T)

# Pyro model: probabilistic model
def model(observed=None):
    # Prior for log(k)
    log_k = pyro.sample("log_k", dist.Normal(torch.zeros(nx, ny, device=device), torch.ones(nx, ny, device=device)).to_event(2))
    k = torch.exp(log_k)
    
    # Forward model
    T = solve_heat_equation(k, q)
    
    # Likelihood
    pyro.sample("obs", dist.Normal(T, noise_std * torch.ones_like(T)).to_event(2), obs=observed)
    
    return T

# call neural network model and generate data.

# Pyro guide (variational distribution)
def guide(observed=None):
    loc = pyro.param("log_k_loc", torch.zeros(nx, ny, device=device))
    scale = pyro.param("log_k_scale", torch.ones(nx, ny, device=device),
                       constraint=dist.constraints.positive)
    return pyro.sample("log_k", dist.Normal(loc, scale).to_event(2))

# Set up the variational inference
pyro.clear_param_store()
adam = pyro.optim.Adam({"lr": 0.01})
svi = SVI(model, guide, adam, loss=Trace_ELBO())

# Run inference
num_iterations = 1000
for i in range(num_iterations):
    loss = svi.step(observed_T)
    if (i+1) % 100 == 0:
        print(f"Iteration {i+1}/{num_iterations} - Loss: {loss}")

# Get the inferred k
inferred_log_k_loc = pyro.param("log_k_loc").detach()
inferred_k = torch.exp(inferred_log_k_loc)

# Plot results
fig, axes = plt.subplots(2, 2, figsize=(12, 12))

im0 = axes[0, 0].imshow(true_k.cpu().numpy(), cmap='viridis')
axes[0, 0].set_title("True Thermal Conductivity (k)")
fig.colorbar(im0, ax=axes[0, 0])

im1 = axes[0, 1].imshow(inferred_k.cpu().numpy(), cmap='viridis')
axes[0, 1].set_title("Inferred Thermal Conductivity (k)")
fig.colorbar(im1, ax=axes[0, 1])

im2 = axes[1, 0].imshow(true_T.detach().cpu().numpy(), cmap='viridis')
axes[1, 0].set_title("True Temperature (T)")
fig.colorbar(im2, ax=axes[1, 0])

im3 = axes[1, 1].imshow(observed_T.detach().cpu().numpy(), cmap='viridis')
axes[1, 1].set_title("Observed Temperature (T)")
fig.colorbar(im3, ax=axes[1, 1])

plt.tight_layout()
plt.savefig(f"../test_result/Heat_"+str(loss_type)+".png")