import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

# Parameters
nu = 1  # viscosity term
L = 100 
nx = 1024
t0 = 0 
tN = 1000
dt = 0.01
nt = int((tN - t0) / 0.05)

# Wave number mesh
k = torch.arange(-nx/2, nx/2, 1)
t = torch.linspace(start=t0, stop=tN, steps=nt)
x = torch.linspace(start=0, stop=L, steps=nx)

# Solution mesh in real space
u = torch.ones((nx, nt))

# Solution mesh in Fourier space
u_hat = torch.ones((nx, nt), dtype=torch.complex64)
u_hat2 = torch.ones((nx, nt), dtype=torch.complex64)

# Initial condition 
u0 = torch.cos((2 * np.pi * x) / L) + 0.1 * torch.cos((4 * np.pi * x) / L)

# Fourier transform of initial condition
u0_hat = (1 / nx) * torch.fft.fftshift(torch.fft.fft(u0))
u0_hat2 = (1 / nx) * torch.fft.fftshift(torch.fft.fft(u0**2))

# Set initial condition in real and Fourier mesh
u[:, 0] = u0
u_hat[:, 0] = u0_hat
u_hat2[:, 0] = u0_hat2

# Fourier Transform of the linear operator
FL = (((2 * np.pi) / L) * k) ** 2 - nu * (((2 * np.pi) / L) * k) ** 4

# Fourier Transform of the non-linear operator
FN = - (1 / 2) * ((1j) * ((2 * np.pi) / L) * k)

# Resolve EDP in Fourier space
for j in range(0, nt-1):
    uhat_current = u_hat[:, j]
    uhat_current2 = u_hat2[:, j]
    if j == 0:
        uhat_last = u_hat[:, 0]
        uhat_last2 = u_hat2[:, 0]
    else:
        uhat_last = u_hat[:, j-1]
        uhat_last2 = u_hat2[:, j-1]
    
    # Compute solution in Fourier space through a finite difference method
    # Crank-Nicholson + Adam 
    u_hat[:, j+1] = (1 / (1 - (dt / 2) * FL)) * ((1 + (dt / 2) * FL) * uhat_current + 
                    (((3 / 2) * FN) * (uhat_current2) - ((1 / 2) * FN) * (uhat_last2)) * dt)
    
    # Go back to real space
    u[:, j+1] = torch.real(nx * torch.fft.ifft(torch.fft.ifftshift(u_hat[:, j+1])))
    u_hat2[:, j+1] = (1 / nx) * torch.fft.fftshift(torch.fft.fft(u[:, j+1]**2))

# Plot the result
fig, ax = plt.subplots(figsize=(10, 8))
xx, tt = torch.meshgrid(x, t, indexing='ij')
levels = torch.arange(-3, 3, 0.01)
cs = ax.contourf(xx.numpy(), tt.numpy(), u.numpy(), levels=levels.numpy(), cmap=cm.jet)
fig.colorbar(cs)
ax.set_xlabel("x")
ax.set_ylabel("t")
ax.set_title(f"Kuramoto-Sivashinsky: L = {L}, nu = {nu}")
plt.savefig("../plot/KS_pytorch.png")