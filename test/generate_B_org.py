import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

class BurgersSimulator(torch.nn.Module):
    def __init__(self, L=2, M=2, Nx=40, Ny=40, Nt=2500, nu_mean=0.5, nu_std=0.1):
        super(BurgersSimulator, self).__init__()
        self.L = L
        self.M = M
        self.Nx = Nx
        self.Ny = Ny
        self.Nt = Nt
        self.nu_mean = nu_mean
        self.nu_std = nu_std
        self.nu = nu_mean

        # Calculate derived parameters
        self.dx = self.L / self.Nx
        self.dy = self.M / self.Ny
        self.dt = 1.0 / self.Nt
        
    def curl(self, u, v):
        du_dy = (u[2:, 1:-1] - u[:-2, 1:-1]) / (2 * self.dy)
        dv_dx = (v[1:-1, 2:] - v[1:-1, :-2]) / (2 * self.dx)
        curl = dv_dx - du_dy
        return curl

    def initialize_conditions(self):
        u = torch.ones(self.Ny, self.Nx)
        v = torch.ones(self.Ny, self.Nx)
        
        # # Draw nu from a Gaussian distribution
        # self.nu = torch.normal(mean=torch.tensor([self.nu_mean]), std=torch.tensor([self.nu_std])).item()
        
        # Introduce stochasticity in the start coordinates of the square using normal distribution
        mean_x = int(0.5 / self.dx)  # Center the mean at the middle of the grid
        mean_y = int(0.5 / self.dy)
        std_x = int(0.1 / self.dx)  # Standard deviation, adjust for spread
        std_y = int(0.1 / self.dy)
        
        x_start = int(torch.normal(mean=torch.tensor([mean_x], dtype=torch.float32), std=torch.tensor([std_x])).item())
        y_start = int(torch.normal(mean=torch.tensor([mean_y], dtype=torch.float32), std=torch.tensor([std_y])).item())
        # x_start = mean_x
        # y_start = mean_y

        # Ensure x_start and y_start are within bounds
        x_start = max(0, min(self.Nx - 1, x_start))
        y_start = max(0, min(self.Ny - 1, y_start))
        
        square_size = int(0.5 / self.dx)  # You can adjust the size as needed
        
        # Ensure the square stays within bounds
        x_end = min(self.Nx, x_start + square_size)
        y_end = min(self.Ny, y_start + square_size)

        u[x_start:x_end, y_start:y_end] = 2
        v[x_start:x_end, y_start:y_end] = 2

        return u, v

    def forward(self, u, v):
        un = u.clone()
        vn = v.clone()

        u_next = (
            un[1:-1, 1:-1]
            - un[1:-1, 1:-1] * self.dt / self.dx * (un[1:-1, 1:-1] - un[1:-1, :-2])
            - vn[1:-1, 1:-1] * self.dt / self.dy * (un[1:-1, 1:-1] - un[:-2, 1:-1])
            + self.nu * self.dt / self.dx**2 * (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, :-2])
            + self.nu * self.dt / self.dy**2 * (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[:-2, 1:-1])
        )

        v_next = (
            vn[1:-1, 1:-1]
            - un[1:-1, 1:-1] * self.dt / self.dx * (vn[1:-1, 1:-1] - vn[:-2, 1:-1])
            - vn[1:-1, 1:-1] * self.dt / self.dy * (vn[1:-1, 1:-1] - vn[1:-1, :-2])
            + self.nu * self.dt / self.dx**2 * (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[:-2, 1:-1])
            + self.nu * self.dt / self.dy**2 * (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, :-2])
        )

        u_new = u.clone()
        v_new = v.clone()

        # Update u and v in the cloned tensors
        u_new[1:-1, 1:-1] = u_next
        v_new[1:-1, 1:-1] = v_next

        return u_new, v_new

def simulate(simulator, u, v, n_steps):
    # Compute the initial curl (analogous to vorticity)
    curl_input = simulator.curl(u, v).cpu().numpy()
    seq = []

    # Run simulation for n_steps
    for s in range(n_steps):
        seq.append(torch.stack([u, v]))
        u, v = simulator(u, v)
        # Compute the curl (analogous to vorticity) after n_steps
        curl_output = simulator.curl(u, v).cpu().numpy()
        # print(f"Simlating step {s}")

        if s % 500 == 0:
            # Save the plots of u, v, input, and output curl fields
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))

            # Plot u field
            im1 = axes[0, 0].imshow(u.cpu().detach().numpy(), cmap='viridis')
            axes[0, 0].set_title(f'u Field at time={s}')
            axes[0, 0].invert_yaxis()
            axes[0, 0].axis('off')
            fig.colorbar(im1, ax=axes[0, 0], orientation='vertical')

            # Plot v field
            im2 = axes[0, 1].imshow(v.cpu().detach().numpy(), cmap='viridis')
            axes[0, 1].set_title(f'v Field at time={s}')
            axes[0, 1].invert_yaxis()
            axes[0, 1].axis('off')
            fig.colorbar(im2, ax=axes[0, 1], orientation='vertical')

            # Plot initial curl field
            im3 = axes[1, 0].imshow(curl_input, cmap='RdBu')
            axes[1, 0].set_title('Initial Curl')
            axes[1, 0].invert_yaxis()
            axes[1, 0].axis('off')
            fig.colorbar(im3, ax=axes[1, 0], orientation='vertical')

            # Plot output curl field
            im4 = axes[1, 1].imshow(curl_output, cmap='RdBu')
            axes[1, 1].set_title(f'Curl at time={s}')
            axes[1, 1].invert_yaxis()
            axes[1, 1].axis('off')
            fig.colorbar(im4, ax=axes[1, 1], orientation='vertical')

            plt.tight_layout()
            plt.savefig(f'burgers_fields_{s}.png', dpi=300)
            plt.close()

    return u, v, curl_output, torch.stack(seq)

def simulate_and_plot(simulator, u, v, n_steps):
    for i in range(n_steps):
        u, v = simulator(u, v)
        print(f"Simulating Step {i}")

        if i % 500 == 0:
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(u.cpu().detach().numpy(), cmap='viridis', origin='lower', extent=[0, 1, 0, 1])
            plt.colorbar()
            plt.title(f'u at step {i}')

            plt.subplot(1, 2, 2)
            plt.imshow(v.cpu().detach().numpy(), cmap='viridis', origin='lower', extent=[0, 1, 0, 1])
            plt.colorbar()
            plt.title(f'v at step {i}')

            plt.savefig(f"fig_{i}.png")
            plt.show()


def compute_jacobian(simulator, u, v):
    def simulation_step(input_tensor):
        u, v = torch.chunk(input_tensor, 2, dim=0)
        u, v = simulator(u, v)
        return torch.cat([u, v], dim=0)

    input_tensor = torch.cat([u, v], dim=0)
    return torch.autograd.functional.jacobian(simulation_step, input_tensor)


if __name__ == "__main__":
    # Instantiate the simulator with default parameters
    simulator = BurgersSimulator()

    # Initialize conditions
    u, v = simulator.initialize_conditions()

    # Run the simulation and plot results
    # simulate_and_plot(simulator, u, v, n_steps=2500)
    u, v, _, _ = simulate(simulator, u, v, n_steps=2500)

    # Compute the Jacobian
    jacobian = compute_jacobian(simulator, u, v)
    print(f"Jacobian shape: {jacobian.shape}")
