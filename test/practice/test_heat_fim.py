import torch
from functorch import vmap, jacrev, hessian

def solve_heat_equation(k, q, nx=50, ny=50, num_iterations=1000):
    dx = dy = 1.0 / (nx - 1)
    T = torch.zeros((nx, ny), device=k.device)
    
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

def compute_fim_wrt_input(k, q, data, noise_std, nx=50, ny=50):
    def ll_func(k):
        T = solve_heat_equation(k, q, nx, ny)
        return log_likelihood(data, T, noise_std)
    
    # Compute the Hessian
    hessian_func = hessian(ll_func)
    hess = hessian_func(k)
    
    # The FIM is the negative expected Hessian
    fim = -hess.reshape(nx, ny, nx, ny)
    
    # Return only the diagonal elements
    return torch.diagonal(fim, dim1=0, dim2=2).transpose(0, 1)

# Example usage
nx, ny = 50, 50
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

k = torch.exp(torch.randn(nx, ny, device=device))
q = torch.ones((nx, ny), device=device) * 100  # Constant heat source term
data = solve_heat_equation(k, q, nx, ny)  # Simulated data
noise_std = 0.01

fim = compute_fim_wrt_input(k, q, data, noise_std, nx, ny)