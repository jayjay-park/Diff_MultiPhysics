import numpy as np
import matplotlib.pyplot as plt

def cheb(N):
    """Compute Chebyshev differentiation matrix."""
    if N == 0:
        return np.array([[0]])
    x = np.cos(np.pi * np.arange(N+1) / N)
    c = np.concatenate(([2], np.ones(N-1), [2])) * (-1)**np.arange(N+1)
    X = np.outer(x, np.ones(N+1))
    dX = X - X.T
    D = np.outer(c, 1/c) / (dX + np.eye(N+1))
    D = D - np.diag(np.sum(D, axis=1))
    return D, x

def solve_darcy_flow(N):
    """Solve 2D Darcy flow using spectral method."""
    D, x = cheb(N)
    y = x.copy()
    xx, yy = np.meshgrid(x, y)

    # Laplacian operator
    I = np.eye(N+1)
    L = np.kron(D@D, I) + np.kron(I, D@D)

    # Boundary conditions (zero pressure on boundaries)
    b = np.zeros((N+1)**2)
    mask = np.ones((N+1, N+1), dtype=bool)
    mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = False
    interior = mask.flatten()

    # Source term (e.g., injection well at the center)
    f = np.zeros((N+1, N+1))
    f[N//2, N//2] = 1

    # Solve the system
    L_int = L[interior][:, interior]
    f_int = f.flatten()[interior] - L[interior][:, ~interior] @ b[~interior]
    p_int = np.linalg.solve(L_int, f_int)

    # Reconstruct full solution
    p = b.copy()
    p[interior] = p_int

    return xx, yy, p.reshape((N+1, N+1))

# Solve
N = 32
xx, yy, p = solve_darcy_flow(N)

# Plot and save pressure distribution
plt.figure(figsize=(10, 8))
cont = plt.contourf(xx, yy, p, levels=20, cmap='viridis')
plt.colorbar(cont, label='Pressure')
plt.title('2D Darcy Flow - Pressure Distribution')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('darcy_flow_pressure.png', dpi=300, bbox_inches='tight')
plt.close()

# Calculate velocity field
D, _ = cheb(N)
ux = -D @ p
uy = -p @ D.T

# Plot and save velocity field (quiver plot)
plt.figure(figsize=(10, 8))
plt.quiver(xx, yy, ux, uy)
plt.title('2D Darcy Flow - Velocity Field')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('darcy_flow_velocity_quiver.png', dpi=300, bbox_inches='tight')
plt.close()

# Calculate velocity magnitude
velocity_magnitude = np.sqrt(ux**2 + uy**2)

# Plot and save velocity magnitude (contour plot)
plt.figure(figsize=(10, 8))
cont = plt.contourf(xx, yy, velocity_magnitude, levels=20, cmap='viridis')
plt.colorbar(cont, label='Velocity Magnitude')
plt.title('2D Darcy Flow - Velocity Magnitude')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('darcy_flow_velocity_magnitude.png', dpi=300, bbox_inches='tight')
plt.close()

print("Plots have been saved as 'darcy_flow_pressure.png', 'darcy_flow_velocity_quiver.png', and 'darcy_flow_velocity_magnitude.png'.")