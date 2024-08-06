import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchdiffeq
from scipy.stats import chi2
import numpy as np

class Lorenz63(nn.Module):
    def __init__(self, sigma, rho, beta):
        super().__init__()
        self.sigma = nn.Parameter(torch.tensor(sigma, dtype=torch.float32))
        self.rho = nn.Parameter(torch.tensor(rho, dtype=torch.float32))
        self.beta = nn.Parameter(torch.tensor(beta, dtype=torch.float32))

    def forward(self, t, state):
        x, y, z = state
        dx = self.sigma * (y - x)
        dy = x * (self.rho - z) - y
        dz = x * y - self.beta * z
        return torch.stack([dx, dy, dz])


def log_likelihood(data, model_output, noise_std):
    return -0.5 * torch.sum((data - model_output)**2) / (noise_std**2) - \
           data.shape[0] * torch.log(torch.tensor(noise_std))

def compute_fim(model, initial_state, t, data, noise_std):
    params = list(model.parameters())
    fim = torch.zeros((len(params), len(params)))
    y = torchdiffeq.odeint(model, initial_state, t, method='rk4', rtol=1e-8)
    ll = log_likelihood(data, y, noise_std)

    for i in range(len(params)):
        grad_i = torch.autograd.grad(ll, params[i], create_graph=True)[0]
        for j in range(i, len(params)):
            grad_j = torch.autograd.grad(grad_i, params[j], retain_graph=True)[0]
            fim[i, j] = grad_j.item()
            fim[j, i] = fim[i, j]

    return fim

def estimate_parameters(model, initial_state, t, data, noise_std, num_iterations=300):
    # num_iterations=1000
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(num_iterations):
        optimizer.zero_grad()
        y = torchdiffeq.odeint(model, initial_state, t, method='rk4', rtol=1e-8)
        loss = -log_likelihood(data, y, noise_std)
        loss.backward()
        optimizer.step()
    return torch.tensor([model.sigma.item(), model.rho.item(), model.beta.item()])



# Set up the problem
true_params = [10.0, 28.0, 8/3]
initial_state = torch.tensor([1.0, 1.0, 1.0])
t = torch.linspace(0, 1, 100)
noise_std = 0.1

# Generate multiple datasets and estimate parameters
num_datasets = 300
estimated_params = []

for num in range(num_datasets):
    print("current sample: ", num)
    with torch.no_grad():
        true_model = Lorenz63(*true_params)
        true_solution = torchdiffeq.odeint(true_model, initial_state, t, method='rk4', rtol=1e-8)
        # true_solution = solve_ode(true_model, initial_state, t)
        data = true_solution + noise_std * torch.randn_like(true_solution)
    
    model = Lorenz63(*true_params)  # Initialize with true params for faster convergence
    estimated_params.append(estimate_parameters(model, initial_state, t, data, noise_std))

estimated_params = torch.stack(estimated_params)

# Compute FIM at true parameters
true_model = Lorenz63(*true_params)
fim = compute_fim(true_model, initial_state, t, true_solution, noise_std)
cov = torch.inverse(fim)

# Plot results
plt.rcParams.update({'font.size': 14})
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
param_pairs = [(0, 1), (0, 2), (1, 2)]
param_names = ['σ', 'ρ', 'β']

for ax, (i, j) in zip(axes, param_pairs):
    ax.scatter(estimated_params[:, i], estimated_params[:, j], alpha=0.5)
    ax.plot(true_params[i], true_params[j], 'r*', markersize=10)
    
    # Plot confidence ellipse
    eigenvalues, eigenvectors = torch.linalg.eig(cov[[i, j]][:, [i, j]])
    eigenvalues = eigenvalues.real
    eigenvectors = eigenvectors.real
    
    # Ensure correct orientation
    if eigenvectors[0, 0] < 0:
        eigenvectors[:, 0] *= -1
    if eigenvectors[1, 1] < 0:
        eigenvectors[:, 1] *= -1
    
    angle = torch.atan2(eigenvectors[1, 0], eigenvectors[0, 0]).item()
    
    ellipse = plt.matplotlib.patches.Ellipse(
        (true_params[i], true_params[j]),
        2 * torch.sqrt(eigenvalues[0]) * np.sqrt(chi2.ppf(0.95, 2)),
        2 * torch.sqrt(eigenvalues[1]) * np.sqrt(chi2.ppf(0.95, 2)),
        angle=np.degrees(angle),
        facecolor='none',
        edgecolor='r',
        linestyle='--'
    )
    ax.add_patch(ellipse)
    ax.set_xlabel(param_names[i])
    ax.set_ylabel(param_names[j])
    ax.set_title(f'{param_names[i]} vs {param_names[j]}')

plt.tight_layout()
plt.show()

print("Fisher Information Matrix:")
print(fim)