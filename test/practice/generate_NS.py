import torch
import torch.autograd.functional as F
import matplotlib.pyplot as plt
import numpy as np

class NavierStokesSimulator(torch.nn.Module):
    def __init__(self, N, L, dt, nu, forcing_func=None):
        super().__init__()
        self.N = N
        self.L = L
        self.dt = dt
        self.nu = nu
        self.forcing_func = forcing_func  # Forcing function
        self.setup_fourier_space()

    def setup_fourier_space(self):
        klin = 2.0 * torch.pi / self.L * torch.arange(-self.N//2, self.N//2, device='cuda')
        self.kx, self.ky = torch.meshgrid(klin, klin, indexing='ij')
        self.kx = torch.fft.ifftshift(self.kx)
        self.ky = torch.fft.ifftshift(self.ky)
        self.kSq = self.kx**2 + self.ky**2
        self.kSq_inv = 1.0 / self.kSq
        self.kSq_inv[self.kSq == 0] = 1

        xlin = torch.linspace(0, self.L, self.N, device='cuda')
        self.xx, self.yy = torch.meshgrid(xlin, xlin, indexing='ij')

        kmax = torch.max(klin)
        self.dealias = ((torch.abs(self.kx) < (2./3.)*kmax) & (torch.abs(self.ky) < (2./3.)*kmax)).float()

    def forward(self, vx, vy):
        dvx_x, dvx_y = self.grad(vx)
        dvy_x, dvy_y = self.grad(vy)
        
        # RHS including advection and forcing term
        rhs_x = -(vx * dvx_x + vy * dvx_y)
        rhs_y = -(vx * dvy_x + vy * dvy_y)
        
        if self.forcing_func:
            fx, fy = self.forcing_func(self.xx, self.yy)
            rhs_x += fx
            rhs_y += fy
        
        rhs_x = self.apply_dealias(rhs_x)
        rhs_y = self.apply_dealias(rhs_y)

        vx = vx + self.dt * rhs_x
        vy = vy + self.dt * rhs_y
        
        div_rhs = self.div(rhs_x, rhs_y)
        P = self.poisson_solve(div_rhs)
        dPx, dPy = self.grad(P)
        
        vx = vx - self.dt * dPx
        vy = vy - self.dt * dPy
        
        vx = self.diffusion_solve(vx)
        vy = self.diffusion_solve(vy)
        
        return vx, vy

    def grad(self, v):
        v_hat = torch.fft.fftn(v)
        return (
            torch.real(torch.fft.ifftn(1j * self.kx * v_hat)),
            torch.real(torch.fft.ifftn(1j * self.ky * v_hat))
        )

    def div(self, vx, vy):
        return (
            torch.real(torch.fft.ifftn(1j * self.kx * torch.fft.fftn(vx))) +
            torch.real(torch.fft.ifftn(1j * self.ky * torch.fft.fftn(vy)))
        )

    def curl(self, vx, vy):
        return (
            torch.real(torch.fft.ifftn(1j * self.ky * torch.fft.fftn(vx))) -
            torch.real(torch.fft.ifftn(1j * self.kx * torch.fft.fftn(vy)))
        )

    def poisson_solve(self, rho):
        rho_hat = torch.fft.fftn(rho)
        phi_hat = -rho_hat * self.kSq_inv
        return torch.real(torch.fft.ifftn(phi_hat))

    def diffusion_solve(self, v):
        v_hat = torch.fft.fftn(v)
        v_hat = v_hat / (1.0 + self.dt * self.nu * self.kSq)
        return torch.real(torch.fft.ifftn(v_hat))

    def apply_dealias(self, f):
        f_hat = self.dealias * torch.fft.fftn(f)
        return torch.real(torch.fft.ifftn(f_hat))


def simulate(simulator, vx, vy, n_steps):
    # Compute the initial vorticity
    wz_input = simulator.curl(vx, vy).cpu().numpy()
    seq = []

    # Run simulation for n_steps
    for s in range(n_steps):
        seq.append(torch.stack([vx, vy]))
        vx, vy = simulator(vx, vy)
        # Compute the final vorticity after n_steps
        wz_output = simulator.curl(vx, vy).cpu().numpy()


        if s % 50 == 0:
            # Save the plots of vx, vy, input, and output vorticity fields
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))

            # Plot vx field
            im1 = axes[0, 0].imshow(vx.cpu().numpy(), cmap='RdBu')
            axes[0, 0].set_title(f'vx Velocity Field at time={s}')
            axes[0, 0].invert_yaxis()
            axes[0, 0].axis('off')
            fig.colorbar(im1, ax=axes[0, 0], orientation='vertical')

            # Plot vy field
            im2 = axes[0, 1].imshow(vy.cpu().numpy(), cmap='RdBu')
            axes[0, 1].set_title(f'vy Velocity Field at time={s}')
            axes[0, 1].invert_yaxis()
            axes[0, 1].axis('off')
            fig.colorbar(im2, ax=axes[0, 1], orientation='vertical')

            # Plot input vorticity field
            im3 = axes[1, 0].imshow(wz_input, cmap='RdBu')
            axes[1, 0].set_title('Input Vorticity')
            axes[1, 0].invert_yaxis()
            axes[1, 0].axis('off')
            fig.colorbar(im3, ax=axes[1, 0], orientation='vertical')

            # Plot output vorticity field
            im4 = axes[1, 1].imshow(wz_output, cmap='RdBu')
            axes[1, 1].set_title('Output Vorticity')
            axes[1, 1].invert_yaxis()
            axes[1, 1].axis('off')
            fig.colorbar(im4, ax=axes[1, 1], orientation='vertical')

            plt.tight_layout()
            plt.savefig(f'velocity_fields_{s}.png', dpi=300)
            plt.close()

    return vx, vy, wz_output, torch.stack(seq)


def compute_jacobian(simulator, vx, vy):
    def simulation_step(input_tensor):
        vx, vy = torch.chunk(input_tensor, 2, dim=0)
        vx, vy = simulator(vx, vy)
        return torch.cat([vx, vy], dim=0)

    input_tensor = torch.cat([vx, vy], dim=0)
    return F.jacobian(simulation_step, input_tensor)

def generate_dataset(simulator, vx, vy, n_steps_early, n_steps_late, save_interval=1):
    # Initial vorticity at time 0
    wz_input = simulator.curl(vx, vy).cpu().numpy()
    
    # Storage for vorticity data
    dataset = []
    
    # Run simulation from time 0 to 10 (early time steps)
    for s in range(n_steps_early):
        vx, vy = simulator(vx, vy)
        if s % save_interval == 0:
            wz_early = simulator.curl(vx, vy).cpu().numpy()
            dataset.append((wz_input, wz_early))
            
    # Store vorticity at time 10
    wz_time10 = simulator.curl(vx, vy).cpu().numpy()
    
    # Run simulation from time 10 to T (late time steps)
    for s in range(n_steps_late):
        vx, vy = simulator(vx, vy)
        if s % save_interval == 0:
            wz_late = simulator.curl(vx, vy).cpu().numpy()
            dataset.append((wz_time10, wz_late))

    return dataset

def plot_simulation_data(simulation_data, n_steps, save_path='../plot/NS_datagen'):
    """
    Plots velocity and vorticity fields from simulation data at different time steps.

    Args:
        simulation_data: torch.Tensor of shape (n_steps, 2, N, N) 
                         containing velocity fields (vx, vy).
        n_steps: int, total number of simulation steps.
        save_path: str, directory to save the plots.
    """
    for s in range(n_steps):
        vx = simulation_data[s, 0].cpu().numpy()
        vy = simulation_data[s, 1].cpu().numpy()
        
        # Compute vorticity
        wz = compute_vorticity(vx, vy)
        
        # Plot vx, vy, and vorticity fields
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot vx
        im1 = axes[0].imshow(vx, cmap='RdBu', origin='lower')
        axes[0].set_title(f'vx Velocity at step {s}')
        fig.colorbar(im1, ax=axes[0])
        
        # Plot vy
        im2 = axes[1].imshow(vy, cmap='RdBu', origin='lower')
        axes[1].set_title(f'vy Velocity at step {s}')
        fig.colorbar(im2, ax=axes[1])
        
        # Plot vorticity
        im3 = axes[2].imshow(wz, cmap='RdBu', origin='lower')
        axes[2].set_title(f'Vorticity at step {s}')
        fig.colorbar(im3, ax=axes[2])
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/fields_step_{s}.png')
        plt.close()

def compute_vorticity(vx, vy):
    """
    Computes the vorticity from velocity fields vx and vy.

    Args:
        vx: np.array, x-component of the velocity.
        vy: np.array, y-component of the velocity.

    Returns:
        np.array, vorticity field.
    """
    dvx_dy, dvx_dx = np.gradient(vx)
    dvy_dy, dvy_dx = np.gradient(vy)
    vorticity = dvy_dx - dvx_dy  # ∇ × u (curl)
    return vorticity




if __name__ == "__main__":

    # Example of how to use the modified code
    N = 128  # Grid resolution
    L = 1.0  # Domain size
    dt = 1e-3  # Time step
    nu = 1e-3  # Viscosity
    n_steps_early = 100  # Number of steps from 0 to 10
    n_steps_late = 200  # Number of steps from 10 to T

    # Define a forcing function, e.g., random noise or a sinusoidal pattern
    def forcing_func(xx, yy):
        fx = 0
        fy = 0
        return fx, fy

    simulator = NavierStokesSimulator(N=N, L=L, dt=dt, nu=nu, forcing_func=forcing_func)

    dx = L / N  # Grid spacing

    # Create a grid with requires_grad=True
    x = torch.linspace(0, L, N, device='cuda', requires_grad=True)
    y = torch.linspace(0, L, N, device='cuda', requires_grad=True)
    X, Y = torch.meshgrid(x, y, indexing='ij')

    # Initialize stream function ψ(x,y) for incompressible flow (to satisfy ∇⋅u = 0)
    psi = torch.sin(2 * np.pi * X) * torch.sin(2 * np.pi * Y)

    # Compute velocity components from stream function
    vx = torch.autograd.grad(psi, Y, grad_outputs=torch.ones_like(psi), create_graph=True)[0]  # ∂ψ/∂y
    vy = -torch.autograd.grad(psi, X, grad_outputs=torch.ones_like(psi), create_graph=True)[0]  # -∂ψ/∂x

    # Compute vorticity: w(x,t) = ∇×u = ∂vy/∂x - ∂vx/∂y
    w = torch.autograd.grad(vy, X, grad_outputs=torch.ones_like(vy), create_graph=True)[0] - \
        torch.autograd.grad(vx, Y, grad_outputs=torch.ones_like(vx), create_graph=True)[0]

    # Forcing function f(x) can be added as needed, for now, initialize to zeros
    f = torch.zeros(N, N, device='cuda')

    # Vorticity at t=0 is w0(x)
    w0 = w.clone()

    # Generate dataset
    dataset = generate_dataset(simulator, vx.detach(), vy.detach(), n_steps_early, n_steps_late)

    # Assuming `simulation_data` contains the velocity fields from simulate() function
    n_steps = 100  # number of steps to plot
    print("shape", torch.tensor(dataset).shape) # [6, 2, 128, 128]
    plot_simulation_data(torch.tensor(dataset), n_steps)
