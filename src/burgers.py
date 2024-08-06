import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define parameters
Nx, Ny = 50, 50
Lx, Ly = 1.0, 1.0
dx, dy = Lx / (Nx - 1), Ly / (Ny - 1)
dt = 0.0001
T = 0.1

# Two parameters
nu_x = torch.tensor(0.01, device=device)
nu_y = torch.tensor(0.02, device=device)

# Create grid
x = torch.linspace(0, Lx, Nx, device=device)
y = torch.linspace(0, Ly, Ny, device=device)
X, Y = torch.meshgrid(x, y, indexing='ij')

# Initial condition
u = torch.sin(2 * torch.pi * X) * torch.cos(2 * torch.pi * Y)
v = -torch.cos(2 * torch.pi * X) * torch.sin(2 * torch.pi * Y)

# Time stepping
t = 0
while t < T:
    # Compute spatial derivatives
    ux = (torch.roll(u, -1, dims=1) - torch.roll(u, 1, dims=1)) / (2 * dx)
    uy = (torch.roll(u, -1, dims=0) - torch.roll(u, 1, dims=0)) / (2 * dy)
    vx = (torch.roll(v, -1, dims=1) - torch.roll(v, 1, dims=1)) / (2 * dx)
    vy = (torch.roll(v, -1, dims=0) - torch.roll(v, 1, dims=0)) / (2 * dy)
    
    uxx = (torch.roll(u, -1, dims=1) - 2*u + torch.roll(u, 1, dims=1)) / dx**2
    uyy = (torch.roll(u, -1, dims=0) - 2*u + torch.roll(u, 1, dims=0)) / dy**2
    vxx = (torch.roll(v, -1, dims=1) - 2*v + torch.roll(v, 1, dims=1)) / dx**2
    vyy = (torch.roll(v, -1, dims=0) - 2*v + torch.roll(v, 1, dims=0)) / dy**2
    
    # Update u and v with separate diffusion coefficients
    u = u + dt * (-u * ux - v * uy + nu_x * uxx + nu_y * uyy)
    v = v + dt * (-u * vx - v * vy + nu_x * vxx + nu_y * vyy)
    
    t += dt

# Move tensors to CPU for plotting
X_cpu, Y_cpu = X.cpu(), Y.cpu()
u_cpu, v_cpu = u.cpu(), v.cpu()

# Plot results
fig = plt.figure(figsize=(12, 5))

ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X_cpu, Y_cpu, u_cpu, cmap='viridis')
ax1.set_title('u component')
ax1.set_xlabel('x')
ax1.set_ylabel('y')

ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(X_cpu, Y_cpu, v_cpu, cmap='viridis')
ax2.set_title('v component')
ax2.set_xlabel('x')
ax2.set_ylabel('y')

plt.tight_layout()
plt.show()

print(f"nu_x: {nu_x.item()}, nu_y: {nu_y.item()}")