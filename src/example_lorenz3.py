import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchdiffeq
from scipy.stats import chi2
import numpy as np
from matplotlib.patches import Ellipse

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
    ll = log_likelihood(data, y, noise_std) # what about the case when we don't know the log_likelihood?

    for i in range(len(params)):
        grad_i = torch.autograd.grad(ll, params[i], create_graph=True)[0]
        for j in range(i, len(params)):
            grad_j = torch.autograd.grad(grad_i, params[j], retain_graph=True)[0]
            fim[i, j] = grad_j.item()
            fim[j, i] = fim[i, j]

    return fim

def estimate_parameters(model, initial_state, t, data, noise_std, num_iterations=300):
    '''gradient based method'''
    # num_iterations=1000
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(num_iterations):
        optimizer.zero_grad()
        y = torchdiffeq.odeint(model, initial_state, t, method='rk4', rtol=1e-8)
        loss = -log_likelihood(data, y, noise_std)
        loss.backward()
        optimizer.step()
    return torch.tensor([model.sigma.item(), model.rho.item(), model.beta.item()])


def plot_likelihood_contours(model, initial_state, t, data, noise_std, true_params, fim, estimated_params, param_indices=(0, 1)):
    # Define grid
    n_points = 50
    param_range = 0.5  # Range around true parameters to plot
    p1, p2 = param_indices
    p1_range = np.linspace(true_params[p1] - param_range, true_params[p1] + param_range, n_points)
    p2_range = np.linspace(true_params[p2] - param_range, true_params[p2] + param_range, n_points)
    P1, P2 = np.meshgrid(p1_range, p2_range)

    # Compute log-likelihood for each point
    log_liks = np.zeros_like(P1)
    for i in range(n_points):
        for j in range(n_points):
            model.sigma.data = torch.tensor(P1[i, j] if p1 == 0 else true_params[0])
            model.rho.data = torch.tensor(P2[i, j] if p2 == 1 else true_params[1])
            model.beta.data = torch.tensor(P1[i, j] if p1 == 2 else (P2[i, j] if p2 == 2 else true_params[2]))
            
            y = torchdiffeq.odeint(model, initial_state, t, method='rk4')
            log_liks[i, j] = log_likelihood(data, y, noise_std).item()

    # Plot likelihood
    plt.rcParams.update({'font.size': 14})
    plt.figure(figsize=(10, 8))
    contour = plt.contour(P1, P2, log_liks, levels=10)
    plt.colorbar(contour, label='Log-likelihood')
    
    # Plot true parameters
    plt.plot(true_params[p1], true_params[p2], 'r*', markersize=15, label='True parameters')
    
    # Plot estimated parameters
    # plt.scatter(estimated_params[:, p1], estimated_params[:, p2], alpha=0.5, s=60, label='Estimated parameters')
    
    # Plot FIM ellipse
    cov = torch.inverse(fim)
    sub_cov = cov[[p1, p2]][:, [p1, p2]]
    eigenvalues, eigenvectors = torch.linalg.eigh(sub_cov)
    # Ensure eigenvalues are positive
    eigenvalues = torch.abs(eigenvalues)

    angle = np.degrees(np.arctan2(eigenvectors[1, 0].item(), eigenvectors[0, 0].item()))
    # 95% CI
    width, height = 2 * torch.sqrt(eigenvalues) * np.sqrt(5.991)  # 95% confidence ..?
    ellipse = Ellipse(xy=(true_params[p1], true_params[p2]), width=width.item(), height=height.item(), 
                      angle=angle, facecolor='none', edgecolor='r', linestyle='--', label='95% CI (FIM)')
    plt.gca().add_patch(ellipse)
    
    # Plot eigenvectors
    for i in range(2):
        eigen_vector = eigenvectors[:, i]
        print(eigen_vector[0].item() * torch.sqrt(eigenvalues[i]).item(), 
            eigen_vector[1].item() * torch.sqrt(eigenvalues[i]).item())
        print(eigen_vector, eigenvalues)
        plt.arrow(true_params[p1], true_params[p2], 
                  eigen_vector[0].item() * eigenvalues[i].item() * 5000, 
                  eigen_vector[1].item() * eigenvalues[i].item() * 5000, 
                  color='black', alpha=0.5, width=0.0005, head_width=0.005, 
                  length_includes_head=True, label=f'Eigenvector' if i == 0 else '')

    param_names = ['σ', 'ρ', 'β']
    plt.xlabel(f'${param_names[p1]}$')
    plt.ylabel(f'${param_names[p2]}$')
    plt.title(f'Log-likelihood contours and Fisher Information Ellipse\n{param_names[p1]} vs {param_names[p2]}')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Contour_'+str(param_indices)+'.png')

# Set up the problem
true_params = [10.0, 28.0, 8/3]
initial_state = torch.tensor([1.0, 1.0, 1.0])
t = torch.linspace(0, 1, 100)
noise_std = 0.1

num_datasets = 2
estimated_params = []
for num in range(num_datasets):
    print(f"Processing dataset {num+1}/{num_datasets}")
    with torch.no_grad():
        true_model = Lorenz63(*true_params)
        true_solution = torchdiffeq.odeint(true_model, initial_state, t, method='rk4', rtol=1e-8)
        data = true_solution + noise_std * torch.randn_like(true_solution)
    
    model = Lorenz63(*true_params)  # Initialize with true params for faster convergence
    estimated_params.append(estimate_parameters(model, initial_state, t, data, noise_std))

estimated_params = torch.stack(estimated_params)

# Compute FIM at true parameters
true_model = Lorenz63(*true_params)
fim = compute_fim(true_model, initial_state, t, true_solution, noise_std)

print("Fisher Information Matrix:")
print(fim)

# Plot likelihood contours for each pair of parameters
plot_likelihood_contours(true_model, initial_state, t, data, noise_std, true_params, fim, estimated_params, param_indices=(0, 1))
plot_likelihood_contours(true_model, initial_state, t, data, noise_std, true_params, fim, estimated_params, param_indices=(0, 2))
plot_likelihood_contours(true_model, initial_state, t, data, noise_std, true_params, fim, estimated_params, param_indices=(1, 2))