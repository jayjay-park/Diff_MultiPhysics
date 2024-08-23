import jax.numpy as jnp
from jax import grad, jit
import matplotlib.pyplot as plt

# Parameters
Nx, Ny = 128, 128
boxSizeX, boxSizeY = 1.0, 1.0
dx, dy = boxSizeX / Nx, boxSizeY / Ny
vol = dx * dy
Y, X = jnp.meshgrid(jnp.linspace(0.5 * dy, boxSizeY - 0.5 * dy, Ny), jnp.linspace(0.5 * dx, boxSizeX - 0.5 * dx, Nx))
courant_fac = 0.4
tEnd = 2
tOut = 0.01
useSlopeLimiting = False
gamma = 5 / 3.0

# Set initial conditions for KHI
w0 = 0.1
sigma = 0.05 / jnp.sqrt(2.0)
rho = 1.0 + (jnp.abs(Y - 0.5) < 0.25)
vx = -0.5 + (jnp.abs(Y - 0.5) < 0.25)
vy = w0 * jnp.sin(4 * jnp.pi * X) * (jnp.exp(-(Y - 0.25) ** 2 / (2 * sigma ** 2)) + jnp.exp(-(Y - 0.75) ** 2 / (2 * sigma ** 2)))
P = jnp.ones_like(X) * 2.5

# Directions for rolling arrays
R = -1  # right
L = 1   # left

# quick plotting function 
def myPlot():
  plt.clf()
  plt.imshow(rho.T)
  plt.clim(0.8, 2.2)
  ax = plt.gca()
  ax.invert_yaxis()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
  plt.draw()
 
myPlot()
outputCount = 1

# Function to compute the total energy
def compute_total_energy(Mass, Momx, Momy, Energy):
    rho = Mass / vol
    vx = Momx / (rho * vol)
    vy = Momy / (rho * vol)
    P = (Energy / vol - 0.5 * rho * (vx ** 2 + vy ** 2)) * (gamma - 1)
    total_energy = jnp.sum(Energy)
    return total_energy

@jit
def time_step(Mass, Momx, Momy, Energy, t):
    rho = Mass / vol
    vx = Momx / (rho * vol)
    vy = Momy / (rho * vol)
    P = (Energy / vol - 0.5 * rho * (vx ** 2 + vy ** 2)) * (gamma - 1)
    
    # Time step calculation (CFL)
    speed_of_sound = jnp.sqrt(gamma * P / rho)
    velocity_magnitude = jnp.sqrt(vx ** 2 + vy ** 2)
    dt = courant_fac * jnp.min(jnp.min(jnp.array([dx, dy])) / (speed_of_sound + velocity_magnitude))
    plotThisTurn = False
    if t + dt > outputCount*tOut:
        dt = outputCount*tOut - t
        plotThisTurn = True

    # Gradient calculations
    rho_gradx = (jnp.roll(rho, R, axis=0) - jnp.roll(rho, L, axis=0)) / (2.0 * dx)
    rho_grady = (jnp.roll(rho, R, axis=1) - jnp.roll(rho, L, axis=1)) / (2.0 * dy)
    vx_gradx = (jnp.roll(vx, R, axis=0) - jnp.roll(vx, L, axis=0)) / (2.0 * dx)
    vx_grady = (jnp.roll(vx, R, axis=1) - jnp.roll(vx, L, axis=1)) / (2.0 * dy)
    vy_gradx = (jnp.roll(vy, R, axis=0) - jnp.roll(vy, L, axis=0)) / (2.0 * dx)
    vy_grady = (jnp.roll(vy, R, axis=1) - jnp.roll(vy, L, axis=1)) / (2.0 * dy)
    P_gradx = (jnp.roll(P, R, axis=0) - jnp.roll(P, L, axis=0)) / (2.0 * dx)
    P_grady = (jnp.roll(P, R, axis=1) - jnp.roll(P, L, axis=1)) / (2.0 * dy)

    if useSlopeLimiting:
        # Apply slope limiting if enabled (same as before)
        rho_gradx = jnp.maximum(0.0, jnp.minimum(1.0, ((rho - jnp.roll(rho, L, axis=0)) / dx) / (rho_gradx + 1.0e-8 * (rho_gradx == 0)))) * rho_gradx
        rho_gradx = jnp.maximum(0.0, jnp.minimum(1.0, (-(rho - jnp.roll(rho, R, axis=0)) / dx) / (rho_gradx + 1.0e-8 * (rho_gradx == 0)))) * rho_gradx
        rho_grady = jnp.maximum(0.0, jnp.minimum(1.0, ((rho - jnp.roll(rho, L, axis=1)) / dy) / (rho_grady + 1.0e-8 * (rho_grady == 0)))) * rho_grady
        rho_grady = jnp.maximum(0.0, jnp.minimum(1.0, (-(rho - jnp.roll(rho, R, axis=1)) / dy) / (rho_grady + 1.0e-8 * (rho_grady == 0)))) * rho_grady
        vx_gradx = jnp.maximum(0.0, jnp.minimum(1.0, ((vx - jnp.roll(vx, L, axis=0)) / dx) / (vx_gradx + 1.0e-8 * (vx_gradx == 0)))) * vx_gradx
        vx_gradx = jnp.maximum(0.0, jnp.minimum(1.0, (-(vx - jnp.roll(vx, R, axis=0)) / dx) / (vx_gradx + 1.0e-8 * (vx_gradx == 0)))) * vx_gradx
        vx_grady = jnp.maximum(0.0, jnp.minimum(1.0, ((vx - jnp.roll(vx, L, axis=1)) / dy) / (vx_grady + 1.0e-8 * (vx_grady == 0)))) * vx_grady
        vx_grady = jnp.maximum(0.0, jnp.minimum(1.0, (-(vx - jnp.roll(vx, R, axis=1)) / dy) / (vx_grady + 1.0e-8 * (vx_grady == 0)))) * vx_grady
        vy_gradx = jnp.maximum(0.0, jnp.minimum(1.0, ((vy - jnp.roll(vy, L, axis=0)) / dx) / (vy_gradx + 1.0e-8 * (vy_gradx == 0)))) * vy_gradx
        vy_gradx = jnp.maximum(0.0, jnp.minimum(1.0, (-(vy - jnp.roll(vy, R, axis=0)) / dx) / (vy_gradx + 1.0e-8 * (vy_gradx == 0)))) * vy_gradx
        vy_grady = jnp.maximum(0.0, jnp.minimum(1.0, ((vy - jnp.roll(vy, L, axis=1)) / dy) / (vy_grady + 1.0e-8 * (vy_grady == 0)))) * vy_grady
        vy_grady = jnp.maximum(0.0, jnp.minimum(1.0, (-(vy - jnp.roll(vy, R, axis=1)) / dy) / (vy_grady + 1.0e-8 * (vy_grady == 0)))) * vy_grady
        P_gradx = jnp.maximum(0.0, jnp.minimum(1.0, ((P - jnp.roll(P, L, axis=0)) / dx) / (P_gradx + 1.0e-8 * (P_gradx == 0)))) * P_gradx
        P_gradx = jnp.maximum(0.0, jnp.minimum(1.0, (-(P - jnp.roll(P, R, axis=0)) / dx) / (P_gradx + 1.0e-8 * (P_gradx == 0)))) * P_gradx
        P_grady = jnp.maximum(0.0, jnp.minimum(1.0, ((P - jnp.roll(P, L, axis=1)) / dy) / (P_grady + 1.0e-8 * (P_grady == 0)))) * P_grady
        P_grady = jnp.maximum(0.0, jnp.minimum(1.0, (-(P - jnp.roll(P, R, axis=1)) / dy) / (P_grady + 1.0e-8 * (P_grady == 0)))) * P_grady

    # Compute updated conserved variables
    Mass_new = Mass - dt * (vx_gradx + vy_grady) * vol
    Momx_new = Momx - dt * (vx * vx_gradx + vy * vx_grady) * vol
    Momy_new = Momy - dt * (vx * vy_gradx + vy * vy_grady) * vol
    Energy_new = Energy - dt * (vx * (P_gradx + P_grady)) * vol

    return Mass_new, Momx_new, Momy_new, Energy_new


# Initial conserved variables
Mass = rho * vol
Momx = rho * vx * vol
Momy = rho * vy * vol
Energy = (P / (gamma - 1) + 0.5 * rho * (vx ** 2 + vy ** 2)) * vol

# Time-stepping
t = 0
while t < tEnd:
    Mass, Momx, Momy, Energy = time_step(Mass, Momx, Momy, Energy, t)
    t += tOut

# Compute the gradient of the total energy with respect to initial conditions
grad_energy_wrt_Mass = grad(compute_total_energy)(Mass, Momx, Momy, Energy)

# Visualization (optional)
plt.imshow(Energy, cmap='viridis')
plt.colorbar()
plt.title('Energy distribution')
plt.show()

# Print results
print("Total Energy:", compute_total_energy(Mass, Momx, Momy, Energy))
print("Gradient of Total Energy with respect to Mass:", grad_energy_wrt_Mass)
