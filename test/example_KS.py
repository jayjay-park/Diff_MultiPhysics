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
        self.L = L
        self.N = N
        self.x = torch.linspace(0, L, N, endpoint=False)
        self.k = 2 * np.pi / L * torch.fft.fftfreq(N, 1/N)
        self.k2 = self.k**2
        self.k4 = self.k**4
        
        # Parameters
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, t, u):
        u_hat = torch.fft.fft(u)
        du_hat = -self.alpha * self.k2 * u_hat - self.beta * self.k4 * u_hat - 0.5j * self.k * torch.fft.fft(u**2)
        return torch.fft.ifft(du_hat).real

def log_likelihood(data, model_output, noise_std):
    return -0.5 * torch.sum((data - model_output)**2) / (noise_std**2) - \
           data.numel() * torch.log(torch.tensor(noise_std))

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

def plot_likelihood_contours(model, initial_state, t, data, noise_std, true_params, fim):
    # Define grid
    n_points = 50
    param_range = 0.2  # Range around true parameters to plot
    alpha_range = np.linspace(true_params[0] - param_range, true_params[0] + param_range, n_points)
    beta_range = np.linspace(true_params[1] - param_range, true_params[1] + param_range, n_points)
    Alpha, Beta = np.meshgrid(alpha_range, beta_range)

    # Compute log-likelihood for each point
    log_liks = np.zeros_like(Alpha)
    for i in range(n_points):
        for j in range(n_points):
            model.alpha.data = torch.tensor(Alpha[i, j])
            model.beta.data = torch.tensor(Beta[i, j])
            
            y = torchdiffeq.odeint(model, initial_state, t, method='rk4')
            log_liks[i, j] = log_likelihood(data, y, noise_std).item()

    # Plot
    plt.figure(figsize=(10, 8))
    contour = plt.contour(Alpha, Beta, log_liks, levels=20)
    plt.colorbar(contour, label='Log-likelihood')
    
    # Plot true parameters
    plt.plot(true_params[0], true_params[1], 'r*', markersize=15, label='True parameters')
    
    # Plot FIM ellipse
    cov = torch.inverse(fim)
    eigenvalues, eigenvectors = torch.linalg.eig(cov)
    eigenvalues = eigenvalues.real
    eigenvectors = eigenvectors.real
    
    angle = np.degrees(np.arctan2(eigenvectors[1, 0].item(), eigenvectors[0, 0].item()))
    width, height = 2 * np.sqrt(eigenvalues) * np.sqrt(5.991)  # 95% confidence
    
    ellipse = Ellipse(xy=(true_params[0], true_params[1]), width=width.item(), height=height.item(), 
                      angle=angle, facecolor='none', edgecolor='r', linestyle='--', label='95% CI (FIM)')
    plt.gca().add_patch(ellipse)
    
    plt.xlabel('α')
    plt.ylabel('β')
    plt.title('Log-likelihood contours and Fisher Information Ellipse\nα vs β')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Set up the problem
L = 16 * np.pi  # Domain size
N = 128  # Number of spatial points
true_params = [1.0, 1.0]  # [alpha, beta]
t = torch.linspace(0, 10, 100)
noise_std = 0.1

# Generate initial condition
x = torch.linspace(0, L, N, endpoint=False)
initial_state = torch.cos(x / 16) * (1 + torch.sin(x / 16))

# Generate data
with torch.no_grad():
    true_model = KuramotoSivashinsky(L, N)
    true_model.alpha.data = torch.tensor(true_params[0])
    true_model.beta.data = torch.tensor(true_params[1])
    true_solution = torchdiffeq.odeint(true_model, initial_state, t, method='rk4', rtol=1e-8)
    data = true_solution + noise_std * torch.randn_like(true_solution)

# Compute FIM
fim = compute_fim(true_model, initial_state, t, data, noise_std)

print("Fisher Information Matrix:")
print(fim)

# Plot likelihood contours
plot_likelihood_contours(true_model, initial_state, t, data, noise_std, true_params, fim)