import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchdiffeq
from scipy.stats import chi2
import numpy as np
from matplotlib.patches import Ellipse

class KuramotoSivashinsky(nn.Module):
    def __init__(self, L, N):
        super().__init__()
        self.L = L  # Domain size
        self.N = N  # Number of spatial points
        self.x = torch.linspace(0, L, N)
        self.k = 2 * torch.pi * torch.fft.fftfreq(N, L / N)

    def forward(self, t, u):
        u_x = torch.fft.ifft(1j * self.k * torch.fft.fft(u)).real
        u_xx = torch.fft.ifft(-self.k**2 * torch.fft.fft(u)).real
        u_xxxx = torch.fft.ifft(self.k**4 * torch.fft.fft(u)).real
        return -0.5 * u * u_x - u_xx - u_xxxx

def log_likelihood(data, model_output, noise_std):
    return -0.5 * torch.sum((data - model_output)**2) / (noise_std**2) - \
           data.numel() * torch.log(torch.tensor(noise_std))

def compute_fim(model, initial_state, t, data, noise_std):
    fim = torch.zeros((len(initial_state), len(initial_state)))
    y = torchdiffeq.odeint(model, initial_state, t, method='rk4', rtol=1e-8)
    ll = log_likelihood(data, y, noise_std)

    for i in range(len(initial_state)):
        grad_i = torch.autograd.grad(ll, initial_state[i], create_graph=True)[0]
        for j in range(i, len(initial_state)):
            grad_j = torch.autograd.grad(grad_i, initial_state[j], retain_graph=True)[0]
            fim[i, j] = grad_j.item()
            fim[j, i] = fim[i, j]

    return fim

def estimate_parameters(model, initial_state, t, data, noise_std, num_iterations=300):
    optimizer = torch.optim.Adam([initial_state], lr=0.01)
    for _ in range(num_iterations):
        optimizer.zero_grad()
        y = torchdiffeq.odeint(model, initial_state, t, method='rk4', rtol=1e-8)
        loss = -log_likelihood(data, y, noise_std)
        loss.backward()
        optimizer.step()
    return initial_state.detach()

def plot_likelihood_contours(model, initial_state, t, data, noise_std, true_initial, fim, estimated_initial, param_indices=(0, 1)):
    # Define grid
    n_points = 50
    param_range = 0.5  # Range around true initial state to plot
    p1, p2 = param_indices
    p1_range = np.linspace(true_initial[p1] - param_range, true_initial[p1] + param_range, n_points)
    p2_range = np.linspace(true_initial[p2] - param_range, true_initial[p2] + param_range, n_points)
    P1, P2 = np.meshgrid(p1_range, p2_range)

    # Compute log-likelihood for each point
    log_liks = np.zeros_like(P1)
    for i in range(n_points):
        for j in range(n_points):
            temp_initial = initial_state.clone()
            temp_initial[p1] = torch.tensor(P1[i, j])
            temp_initial[p2] = torch.tensor(P2[i, j])
            
            y = torchdiffeq.odeint(model, temp_initial, t, method='rk4')
            log_liks[i, j] = log_likelihood(data, y, noise_std).item()

    # Plot likelihood
    plt.figure(figsize=(10, 8))
    contour = plt.contour(P1, P2, log_liks, levels=10)
    plt.colorbar(contour, label='Log-likelihood')
    
    # Plot true initial state
    plt.plot(true_initial[p1], true_initial[p2], 'r*', markersize=15, label='True initial state')
    
    # Plot FIM ellipse
    cov = torch.inverse(fim)
    sub_cov = cov[[p1, p2]][:, [p1, p2]]
    eigenvalues, eigenvectors = torch.linalg.eigh(sub_cov)
    eigenvalues = torch.abs(eigenvalues)

    angle = np.degrees(np.arctan2(eigenvectors[1, 0].item(), eigenvectors[0, 0].item()))
    width, height = 2 * torch.sqrt(eigenvalues) * np.sqrt(5.991)
    ellipse = Ellipse(xy=(true_initial[p1], true_initial[p2]), width=width.item(), height=height.item(), 
                      angle=angle, facecolor='none', edgecolor='r', linestyle='--', label='95% CI (FIM)')
    plt.gca().add_patch(ellipse)
    
    plt.xlabel(f'Initial state component {p1}')
    plt.ylabel(f'Initial state component {p2}')
    plt.title(f'Log-likelihood contours and Fisher Information Ellipse\nInitial state components {p1} vs {p2}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'KS_Contour_{p1}_{p2}.png')

# Set up the problem
L = 32  # Domain size
N = 64  # Number of spatial points
initial_state = torch.sin(2 * torch.pi * torch.linspace(0, 1, N) / L) + 0.01 * torch.randn(N)
initial_state = initial_state.requires_grad_(True)
t = torch.linspace(0, 10, 100)
noise_std = 0.1

model = KuramotoSivashinsky(L, N)

# Generate data
with torch.no_grad():
    true_solution = torchdiffeq.odeint(model, initial_state, t, method='rk4', rtol=1e-8)
    data = true_solution + noise_std * torch.randn_like(true_solution)

# Estimate parameters (initial state in this case)
estimated_initial = estimate_parameters(model, initial_state, t, data, noise_std)

# Compute FIM
fim = compute_fim(model, initial_state, t, true_solution, noise_std)

print("Fisher Information Matrix:")
print(fim)

# Plot likelihood contours for a few pairs of initial state components
plot_likelihood_contours(model, initial_state, t, data, noise_std, initial_state, fim, estimated_initial, param_indices=(0, 1))
plot_likelihood_contours(model, initial_state, t, data, noise_std, initial_state, fim, estimated_initial, param_indices=(0, 2))
plot_likelihood_contours(model, initial_state, t, data, noise_std, initial_state, fim, estimated_initial, param_indices=(1, 2))