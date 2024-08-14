import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchdiffeq
from scipy.stats import chi2
import numpy as np
from matplotlib.patches import Ellipse

class KuramotoSivashinsky(nn.Module):
    '''Credit: https://scicomp.stackexchange.com/questions/37336/solving-numerically-the-1d-kuramoto-sivashinsky-equation-using-spectral-methods'''

    def __init__(self, nu=1, L=100, nx=1024, dt=0.05):
        super().__init__()
        self.nu = nu
        self.L = L
        self.nx = nx
        self.dt = dt
        
        # Wave number mesh
        self.k = torch.arange(-nx/2, nx/2, 1)
        
        # Fourier Transform of the linear operator
        self.FL = (((2 * np.pi) / L) * self.k) ** 2 - nu * (((2 * np.pi) / L) * self.k) ** 4
        
        # Fourier Transform of the non-linear operator
        self.FN = - (1 / 2) * ((1j) * ((2 * np.pi) / L) * self.k)

    def forward(self, nt, u0):
        # Initialize solution meshes
        u = torch.ones((self.nx, nt))
        u_hat = torch.ones((self.nx, nt), dtype=torch.complex64)
        u_hat2 = torch.ones((self.nx, nt), dtype=torch.complex64)

        # Set initial conditions
        u[:, 0] = u0
        u_hat[:, 0] = (1 / self.nx) * torch.fft.fftshift(torch.fft.fft(u0))
        u_hat2[:, 0] = (1 / self.nx) * torch.fft.fftshift(torch.fft.fft(u0**2))

        # Time-stepping
        for j in range(nt-1):
            uhat_current = u_hat[:, j]
            uhat_current2 = u_hat2[:, j]
            
            if j == 0:
                uhat_last = u_hat[:, 0]
                uhat_last2 = u_hat2[:, 0]
            else:
                uhat_last = u_hat[:, j-1]
                uhat_last2 = u_hat2[:, j-1]
            
            # Crank-Nicholson + Adam scheme
            u_hat[:, j+1] = (1 / (1 - (self.dt / 2) * self.FL)) * (
                (1 + (self.dt / 2) * self.FL) * uhat_current + 
                (((3 / 2) * self.FN) * uhat_current2 - ((1 / 2) * self.FN) * uhat_last2) * self.dt
            )
            
            # Go back to real space
            u[:, j+1] = torch.real(self.nx * torch.fft.ifft(torch.fft.ifftshift(u_hat[:, j+1])))
            u_hat2[:, j+1] = (1 / self.nx) * torch.fft.fftshift(torch.fft.fft(u[:, j+1]**2))

        return u


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

# Setup parameters
nu = 1
L = 100
nx = 1024
t0, tN = 0, 200
dt = 0.01
t = int((tN - t0) / dt)
true_params = [1.0, 1.0]  # [alpha, beta]
noise_std = 0.1

# Generate initial condition
x = torch.linspace(0, L, nx)
initial_state = torch.cos(x / 16) * (1 + torch.sin(x / 16))

# Generate data
with torch.no_grad():
    true_model = KuramotoSivashinsky(nu, L, nx, dt)
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