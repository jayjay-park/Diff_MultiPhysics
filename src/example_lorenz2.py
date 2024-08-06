import torch
import matplotlib.pyplot as plt

class Lorenz63(torch.nn.Module):
    def __init__(self, sigma, rho, beta):
        super().__init__()
        self.sigma = torch.nn.Parameter(torch.tensor(sigma))
        self.rho = torch.nn.Parameter(torch.tensor(rho))
        self.beta = torch.nn.Parameter(torch.tensor(beta))

    def forward(self, t, state):
        x, y, z = state
        dx = self.sigma * (y - x)
        dy = x * (self.rho - z) - y
        dz = x * y - self.beta * z
        return torch.stack([dx, dy, dz])

def solve_ode(func, y0, t):
    solution = [y0]
    for i in range(1, len(t)):
        dt = t[i] - t[i-1]
        dy = func(t[i-1], solution[-1]) * dt
        solution.append(solution[-1] + dy)
    return torch.stack(solution)

# Set up initial conditions and time steps
y0 = torch.tensor([1.0, 1.0, 1.0])
t = torch.linspace(0, 1, 100)

# Create the Lorenz63 model
model = Lorenz63(sigma=10.0, rho=28.0, beta=8.0/3.0)

# Solve the ODE
trajectory = solve_ode(model, y0, t)

def compute_fim(model, trajectory):
    fim = torch.zeros(3, 3)  # 3x3 for sigma, rho, beta
    params = [model.sigma, model.rho, model.beta]

    for state in trajectory:
        log_likelihood = torch.sum(state**2)  # Simple likelihood function
        log_likelihood.backward(retain_graph=True)

        for i, param_i in enumerate(params):
            for j, param_j in enumerate(params):
                fim[i, j] += param_i.grad * param_j.grad

        model.zero_grad()

    fim /= len(trajectory)
    return fim

fim = compute_fim(model, trajectory)
print("Fisher Information Matrix:")
print(fim)

# Compute eigenvectors and eigenvalues
eigenvalues, eigenvectors = torch.linalg.eigh(fim)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the Lorenz attractor
x, y, z = trajectory.detach().numpy().T
ax.plot(x, y, z, lw=0.5)

# Choose a point to visualize the eigenvectors
point_index = len(t) // 2  # Middle point of the trajectory

# Plot the eigenvectors at this point
colors = ['r', 'g', 'b']
for i in range(3):
    eigenvector = eigenvectors[:, i].numpy()
    ax.quiver(x[point_index], y[point_index], z[point_index], 
              eigenvector[0], eigenvector[1], eigenvector[2], 
              color=colors[i], lw=2, length=1, normalize=True)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Lorenz Attractor with FIM Eigenvectors')

plt.tight_layout()
plt.show()