import torch
import numpy as np
import matplotlib.pyplot as plt

def solve_heat_equation(k, q, nx=50, ny=50, num_iterations=1000):
    dx = dy = 1.0 / (nx - 1)
    T = torch.zeros((nx, ny), device="cuda")
    
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

def log_likelihood(data, model_output, noise_std):
    return -0.5 * torch.sum((data - model_output)**2) / (noise_std**2) - \
           data.numel() * torch.log(torch.tensor(noise_std))

def compute_fim_for_2d_heat(solve_heat_equation, k, q, T_data, noise_std, nx=50, ny=50):
    # Ensure k is a tensor with gradient tracking
    k = torch.tensor(k, requires_grad=True).cuda()
    
    fim = torch.zeros((nx*ny, nx*ny))
    
    # Solve heat equation
    T_pred = solve_heat_equation(k, q.cuda(), nx, ny)
    ll = log_likelihood(T_data, T_pred, noise_std)

    for i in range(nx*ny):
        grad_i = torch.autograd.grad(ll, k, create_graph=True)[0].flatten()[i]
        for j in range(i, nx*ny):
            grad_j = torch.autograd.grad(grad_i, k, retain_graph=True)[0].flatten()[j]
            fim[i, j] = grad_j.item()
            fim[j, i] = fim[i, j]

    return fim

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device: ", device)

# Example usage:
nx, ny = 50, 50
k = torch.exp(torch.randn(nx, ny)).cuda()  # Log-normal distribution for k
q = torch.ones((nx, ny)) * 100  # Constant heat source term
T_data = solve_heat_equation(k, q.cuda(), nx, ny)  # This is your ground truth data
noise_std = 0.01  # Adjust as needed

# Compute FIM
fim = compute_fim_for_2d_heat(solve_heat_equation, k, q, T_data, noise_std, nx, ny)

# Visualize FIM
plt.figure(figsize=(10, 10))
plt.imshow(fim.numpy(), cmap='viridis', aspect='auto')
plt.colorbar()
plt.title('Fisher Information Matrix for 2D Heat Equation (k parameter)')
plt.xlabel('k Index')
plt.ylabel('k Index')
plt.savefig('fim_2d_heat_k.png')
plt.close()

# Analyze the FIM
eigenvalues, _ = torch.linalg.eig(fim)
print("Top 10 eigenvalues of FIM:", eigenvalues.real.sort(descending=True)[0][:10])

# Visualize the diagonal of FIM
plt.figure(figsize=(10, 5))
plt.plot(fim.diag().numpy())
plt.title('Diagonal of FIM (Sensitivity of each k value)')
plt.xlabel('k Index')
plt.ylabel('FIM Diagonal Value')
plt.savefig('fim_2d_heat_k_diagonal.png')
plt.close()