import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parameters
Nx = 128
Ny = 128
boxSizeX = 1.
boxSizeY = 1.
dx = boxSizeX / Nx
dy = boxSizeY / Ny
vol = dx * dy
courant_fac = 0.4
t = 0
tEnd = 2
tOut = 0.01
useSlopeLimiting = False

# Create the grid
x = torch.linspace(0.5 * dx, boxSizeX - 0.5 * dx, Nx)
y = torch.linspace(0.5 * dy, boxSizeY - 0.5 * dy, Ny)
X, Y = torch.meshgrid(x, y, indexing='ij')

# Set initial conditions for KHI
w0 = 0.1
sigma = 0.05 / torch.sqrt(torch.tensor(2.))
gamma = 5 / 3.
rho = 1. + (torch.abs(Y - 0.5) < 0.25).float()
vx = -0.5 + (torch.abs(Y - 0.5) < 0.25).float()
vy = w0 * torch.sin(4 * torch.pi * X) * (torch.exp(-(Y - 0.25) ** 2 / (2 * sigma ** 2)) + torch.exp(-(Y - 0.75) ** 2 / (2 * sigma ** 2)))
vz = torch.zeros_like(X)
P = torch.zeros_like(X) + 2.5

# Quick plotting function
def update_plot(i):
    global Mass, Momx, Momy, Energy, t, outputCount

    # Get primitive variables
    rho = Mass / vol
    vx = Momx / (rho * vol)
    vy = Momy / (rho * vol)
    P = (Energy / vol - 0.5 * rho * (vx ** 2 + vy ** 2)) * (gamma - 1)

    # Get time step (CFL)
    dt = courant_fac * torch.min(torch.min(dx / (torch.sqrt(gamma * P / rho) + torch.sqrt(vx ** 2 + vy ** 2))))
    plotThisTurn = False
    if t + dt > outputCount * tOut:
        dt = outputCount * tOut - t
        plotThisTurn = True

    # Calculate gradients
    def gradient(tensor, axis):
        tensor = tensor.clone()
        return (torch.roll(tensor, shifts=R, dims=axis) - torch.roll(tensor, shifts=L, dims=axis)) / (2. * (dx if axis == 0 else dy))

    rho_gradx = gradient(rho, 0)
    rho_grady = gradient(rho, 1)
    vx_gradx = gradient(vx, 0)
    vx_grady = gradient(vx, 1)
    vy_gradx = gradient(vy, 0)
    vy_grady = gradient(vy, 1)
    P_gradx = gradient(P, 0)
    P_grady = gradient(P, 1)
    
    # Slope limit gradients
    if useSlopeLimiting:
        def slope_limiter(grad, tensor, axis):
            grad = torch.maximum(torch.tensor(0.), torch.minimum(torch.tensor(1.), ((tensor - torch.roll(tensor, shifts=L, dims=axis)) / (dx if axis == 0 else dy)) / (grad + 1.0e-8 * (grad == 0))))
            grad = torch.maximum(torch.tensor(0.), torch.minimum(torch.tensor(1.), (-(tensor - torch.roll(tensor, shifts=R, dims=axis)) / (dx if axis == 0 else dy)) / (grad + 1.0e-8 * (grad == 0))))
            return grad

        rho_gradx = slope_limiter(rho_gradx, rho, 0)
        rho_grady = slope_limiter(rho_grady, rho, 1)
        vx_gradx = slope_limiter(vx_gradx, vx, 0)
        vx_grady = slope_limiter(vx_grady, vx, 1)
        vy_gradx = slope_limiter(vy_gradx, vy, 0)
        vy_grady = slope_limiter(vy_grady, vy, 1)
        P_gradx = slope_limiter(P_gradx, P, 0)
        P_grady = slope_limiter(P_grady, P, 1)

    # Extrapolate to cell faces (in time & space)
    def extrapolate(tensor, gradx, grady, dx, dy):
        tensor_prime = tensor - 0.5 * dt * (vx * gradx + tensor * vx_gradx + vy * grady + tensor * vy_grady)
        tensor_XL = tensor_prime - gradx * dx / 2.
        tensor_XL = torch.roll(tensor_XL, shifts=R, dims=0)
        tensor_XR = tensor_prime + gradx * dx / 2.
        tensor_YL = tensor_prime - grady * dy / 2.
        tensor_YL = torch.roll(tensor_YL, shifts=R, dims=1)
        tensor_YR = tensor_prime + grady * dy / 2.
        return tensor_XL, tensor_XR, tensor_YL, tensor_YR

    rho_XL, rho_XR, rho_YL, rho_YR = extrapolate(rho, rho_gradx, rho_grady, dx, dy)
    vx_XL, vx_XR, vx_YL, vx_YR = extrapolate(vx, vx_gradx, vx_grady, dx, dy)
    vy_XL, vy_XR, vy_YL, vy_YR = extrapolate(vy, vy_gradx, vy_grady, dx, dy)
    P_XL, P_XR, P_YL, P_YR = extrapolate(P, P_gradx, P_grady, dx, dy)

    # Compute star (averaged) states
    def average(tensor1, tensor2):
        return 0.5 * (tensor1 + tensor2)

    rho_Xstar = average(rho_XL, rho_XR)
    rho_Ystar = average(rho_YL, rho_YR)
    momx_Xstar = average(rho_XL * vx_XL, rho_XR * vx_XR)
    momx_Ystar = average(rho_YL * vx_YL, rho_YR * vx_YR)
    momy_Xstar = average(rho_XL * vy_XL, rho_XR * vy_XR)
    momy_Ystar = average(rho_YL * vy_YL, rho_YR * vy_YR)
    en_Xstar = average(P_XL / (gamma - 1) + 0.5 * rho_XL * (vx_XL ** 2 + vy_XL ** 2), P_XR / (gamma - 1) + 0.5 * rho_XR * (vx_XR ** 2 + vy_XR ** 2))
    en_Ystar = average(P_YL / (gamma - 1) + 0.5 * rho_YL * (vx_YL ** 2 + vy_YL ** 2), P_YR / (gamma - 1) + 0.5 * rho_YR * (vx_YR ** 2 + vy_YR ** 2))

    P_Xstar = (gamma - 1) * (en_Xstar - 0.5 * (momx_Xstar ** 2 + momy_Xstar ** 2) / rho_Xstar)
    P_Ystar = (gamma - 1) * (en_Ystar - 0.5 * (momx_Ystar ** 2 + momy_Ystar ** 2) / rho_Ystar)

    # Compute fluxes (local Lax-Friedrichs/Rusanov)
    def flux(rho, momx, momy, P, dx, dy):
        flux_rho = momx
        flux_momx = (momx ** 2 / rho) + P
        flux_momy = momx * momy / rho
        flux_en = (P + flux_momx) * momx / rho
        return flux_rho, flux_momx, flux_momy, flux_en

    flux_rho_X, flux_momx_X, flux_momy_X, flux_en_X = flux(rho_Xstar, momx_Xstar, momy_Xstar, P_Xstar, dx, dy)
    flux_rho_Y, flux_momx_Y, flux_momy_Y, flux_en_Y = flux(rho_Ystar, momx_Ystar, momy_Ystar, P_Ystar, dx, dy)

    # Update conserved variables
    Mass -= dt * (flux_rho_X - flux_rho_Y)
    Momx -= dt * (flux_momx_X - flux_momx_Y)
    Momx -= dt * (flux_momx_X - flux_momx_Y)
    Energy -= dt * (flux_en_X - flux_en_Y)

    t += dt

# Create the figure and axes
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
ax1, ax2, ax3, ax4 = axes.flatten()  # Flatten the 2x2 array of Axes

def update_plot(frame):
    # Your existing update_plot function
    # Plotting
    ax1.clear()
    c1 = ax1.pcolormesh(X, Y, rho.numpy(), shading='auto', cmap='viridis')
    fig.colorbar(c1, ax=ax1)
    ax1.set_title('Density')
    
    ax2.clear()
    c2 = ax2.pcolormesh(X, Y, vx.numpy(), shading='auto', cmap='viridis')
    fig.colorbar(c2, ax=ax2)
    ax2.set_title('Velocity X')
    
    ax3.clear()
    c3 = ax3.pcolormesh(X, Y, vy.numpy(), shading='auto', cmap='viridis')
    fig.colorbar(c3, ax=ax3)
    ax3.set_title('Velocity Y')
    
    ax4.clear()
    c4 = ax4.pcolormesh(X, Y, P.numpy(), shading='auto', cmap='viridis')
    fig.colorbar(c4, ax=ax4)
    ax4.set_title('Pressure')

    plt.draw()
    plt.pause(0.001)

# Run the simulation and update plot
anim = animation.FuncAnimation(fig, update_plot, frames=range(int(tEnd / tOut)), repeat=False)
plt.show()