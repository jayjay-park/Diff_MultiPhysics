import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

class KuramotoSivashinsky(nn.Module):
    def __init__(self, nu=1, L=100, nx=1024, dt=0.01):
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

# Setup parameters
nu = 1
L = 100
nx = 1024
t0, tN = 0, 1000
dt = 0.05
nt = int((tN - t0) / dt)

# Create model
ks_model = KuramotoSivashinsky(nu, L, nx, dt)

# Initial condition
x = torch.linspace(0, L, nx)
u0 = torch.cos((2 * np.pi * x) / L) + 0.1 * torch.cos((4 * np.pi * x) / L)

# Solve
solution = ks_model(nt, u0)

# Plot the result
t = torch.linspace(t0, tN, nt)
fig, ax = plt.subplots(figsize=(10,8))
xx, tt = np.meshgrid(x.numpy(), t.numpy())
levels = np.arange(-3, 3, 0.01)
cs = ax.contourf(xx, tt, solution.T.numpy(), levels=levels, cmap=cm.jet)
fig.colorbar(cs)
ax.set_xlabel("x")
ax.set_ylabel("t")
ax.set_title(f"Kuramoto-Sivashinsky: L = {L}, nu = {nu}")
plt.savefig("../plot/KS_pytorch.png")