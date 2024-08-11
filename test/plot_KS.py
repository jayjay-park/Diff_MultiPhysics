import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

def kuramoto_sivashinsky_step(u_t, nu=1, L=100, nx=1024, dt=0.05):
    '''Solve one time step of the Kuramoto-Sivashinsky equation.'''
    
    # Wave number mesh
    k = torch.arange(-nx/2, nx/2, 1, dtype=torch.float32)
    
    # Fourier Transform of the linear operator
    FL = (((2 * np.pi) / L) * k) ** 2 - nu * (((2 * np.pi) / L) * k) ** 4
    
    # Fourier Transform of the non-linear operator
    FN = - (1 / 2) * ((1j) * ((2 * np.pi) / L) * k)
    
    # Fourier Transform of current state
    u_hat = (1 / nx) * torch.fft.fftshift(torch.fft.fft(u_t))
    u_hat2 = (1 / nx) * torch.fft.fftshift(torch.fft.fft(u_t**2))
    
    # Crank-Nicholson + Adam scheme
    u_hat_next = (1 / (1 - (dt / 2) * FL)) * (
        (1 + (dt / 2) * FL) * u_hat + 
        (((3 / 2) * FN) * u_hat2 - ((1 / 2) * FN) * u_hat2) * dt
    )
    
    # Go back to real space
    u_t_next = torch.real(nx * torch.fft.ifft(torch.fft.ifftshift(u_hat_next)))

    return u_t_next

# Setup parameters
nu = 1
L = 100
nx = 1024
t0, tN = 0, 200
dt = 0.01
nt = int((tN - t0) / dt)

# Initial condition
x = torch.linspace(0, L, nx)
u0 = torch.cos((2 * np.pi * x) / L) + 0.1 * torch.cos((4 * np.pi * x) / L)

# Time-stepping
u = torch.zeros((nx, nt), dtype=torch.float32)
u[:, 0] = u0

for j in range(nt-1):
    u[:, j+1] = kuramoto_sivashinsky_step(u[:, j], nu=nu, L=L, nx=nx, dt=dt)

# Plot the result
plt.rcParams.update({'font.size': 14})
t = torch.linspace(t0, tN, nt)
fig, ax = plt.subplots(figsize=(10, 8))
xx, tt = np.meshgrid(x.numpy(), t.numpy())
levels = np.arange(-3, 3, 0.01)
cs = ax.contourf(xx, tt, u.T.numpy(), levels=levels, cmap=cm.jet)
fig.colorbar(cs)
ax.set_xlabel("x")
ax.set_ylabel("t")
plt.tight_layout()
ax.set_title(f"Kuramoto-Sivashinsky: L = {L}, nu = {nu}")
plt.savefig("../plot/KS_squential_solution.png")
