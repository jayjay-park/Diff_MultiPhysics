import torch
import torch.autograd.functional as F
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft2, ifft2, fftshift

class NavierStokesSimulator(torch.nn.Module):
    def __init__(self, N, L, dt, nu):
        super().__init__()
        self.N = N
        self.L = L
        self.dt = dt
        self.nu = nu
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
        
        rhs_x = -(vx * dvx_x + vy * dvx_y)
        rhs_y = -(vx * dvy_x + vy * dvy_y)
        
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


import torch.fft

# def generate_gaussian_random_field(nx, ny, mean, std):
#     """Generate a Gaussian random field using Fourier-based method."""
#     noise = torch.normal(mean=0.0, std=std, size=(nx, ny), device='cuda')
#     noise_fft = torch.fft.fftn(noise)
    
#     kx = torch.fft.fftfreq(nx).reshape(-1, 1).to('cuda')
#     ky = torch.fft.fftfreq(ny).reshape(1, -1).to('cuda')
#     k_square = kx**2 + ky**2
    
#     gaussian_filter = torch.exp(-k_square / (2 * (std**2)))
#     smooth_noise_fft = noise_fft * gaussian_filter
    
#     smooth_noise = torch.fft.ifftn(smooth_noise_fft).real
#     smooth_noise = mean + smooth_noise * std
    
#     return smooth_noise

def gaussian_random_field_2d(size, scale=10.0, random_seed=None):
    """
    Generate a 2D Gaussian random field using spectral synthesis.

    Args:
    - size (tuple): Grid size (nx, ny)
    - scale (float): Controls the correlation length (larger values = smoother field)
    - random_seed (int): Random seed for reproducibility

    Returns:
    - field (2D numpy array): Generated Gaussian random field
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    nx, ny = size
    # Generate white noise in the Fourier domain (complex numbers)
    noise = np.random.randn(nx, ny) + 1j * np.random.randn(nx, ny)

    # Generate grid of frequency indices
    kx = np.fft.fftfreq(nx)[:, None]  # Frequency indices for x-axis
    ky = np.fft.fftfreq(ny)[None, :]  # Frequency indices for y-axis

    # Compute the radial frequency (distance from the origin in the Fourier domain)
    k = np.sqrt(kx**2 + ky**2)

    # Power spectrum: scale controls the smoothness (higher scale = smoother field)
    power_spectrum = np.exp(-k**2 * scale**2)

    # Apply the power spectrum to the noise
    field_fourier = noise * np.sqrt(power_spectrum)

    # Inverse Fourier transform to get the field in spatial domain
    field = np.real(ifft2(field_fourier))

    # Normalize the field (optional)
    field -= np.mean(field)
    field /= np.std(field)

    return field

def vorticity_to_velocity(wz, nx, ny):
    """Convert vorticity field to velocity field (vx, vy) using Fourier transform."""
    kx = torch.fft.fftfreq(nx).reshape(-1, 1).to('cuda')
    ky = torch.fft.fftfreq(ny).reshape(1, -1).to('cuda')
    k_square = kx**2 + ky**2
    k_square[0, 0] = 1.0  # avoid division by zero

    # Fourier transform of vorticity
    wz_fft = torch.fft.fftn(wz)

    # Compute velocity components in Fourier space using Biot-Savart relation
    vx_fft = 1j * ky * wz_fft / k_square
    vy_fft = -1j * kx * wz_fft / k_square

    # Inverse Fourier transform to get velocity in real space
    vx = torch.fft.ifftn(vx_fft).real
    vy = torch.fft.ifftn(vy_fft).real

    return vx, vy

# def generate_dataset(simulator, num_samples=500, n_steps=200, nx=64, ny=64):
#     dataset = []
#     for s in range(int(num_samples)):
#         print("initial condition :", s)

#         # Initialize vorticity as a Gaussian random field
#         # wz = generate_gaussian_random_field(nx, ny, mean=0.0, std=1.0)
#         wz = 

#         # Convert vorticity to velocity fields using Biot-Savart law (or equivalent)
#         vx, vy = vorticity_to_velocity(wz, nx, ny)

#         vx_init = vx.detach()
#         vy_init = vy.detach()

#         # Simulate for n_steps
#         with torch.no_grad():
#             vx_final, vy_final, wz_final, seq_val = simulate(simulator, vx, vy, n_steps)

#         # Store the data
#         dataset.append({
#             'vx': seq_val[:, 0].cpu().numpy(),
#             'vy': seq_val[:, 1].cpu().numpy(),
#             'wz': wz.cpu().numpy(),  # Storing initial vorticity as well
#         })
#         torch.cuda.empty_cache()

#     return dataset

def simulate_vorticity_driven(simulator, wz_init, n_steps):
    # Convert initial vorticity to velocity using Biot-Savart law
    wz_init = torch.tensor(wz_init).cuda()
    vx, vy = vorticity_to_velocity(wz_init, simulator.N, simulator.N)
    
    # Store initial vorticity for reference
    wz_input = wz_init.cpu().numpy()
    seq = []

    # Run the simulation for n_steps
    for s in range(n_steps):
        seq.append(torch.stack([vx, vy]))

        # Evolve velocity fields
        vx, vy = simulator(vx, vy)

        # Compute vorticity at each step
        wz_output = simulator.curl(vx, vy).cpu().numpy()

        if s % 50 == 0:
            # Save plots of vx, vy, and input/output vorticity fields
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
            plt.savefig(f'vorticity_fields_{s}.png', dpi=300)
            plt.close()

    return vx, vy, wz_output, torch.stack(seq)



if __name__ == "__main__":
    # Usage
    N = 64  # Grid size
    L = 1.0  # Domain length
    dt = 0.001  # Time step
    nu = 0.001  # Viscosity
    n_steps = 300  # Number of time steps to simulate

    simulator = NavierStokesSimulator(N, L, dt, nu).cuda()

    # Initialize with random initial conditions
    xlin = torch.linspace(0, L, N, device='cuda')
    xx, yy = torch.meshgrid(xlin, xlin, indexing='ij')

    # # Initial Condition (vortex)
    # Generate random parameters for the sinusoidal function
    # freq_x = torch.normal(mean=3.7697, std=0.0443, size=(1,), device='cuda').item()
    # freq_y = torch.normal(mean=2.9267, std=0.2585, size=(1,), device='cuda').item()
    # phase_x = torch.normal(mean=0.5348, std=0.1121, size=(1,), device='cuda').item()
    # Set parameters
    # size = (64, 64)  # Grid size
    # scale = 20.0       # Controls smoothness
    # random_seed = 42   # Set seed for reproducibility

    # # Generate the field
    # wz_init = gaussian_random_field_2d(size, scale, random_seed)
    from gstools import SRF, Gaussian

    x = y = range(64)
    model = Gaussian(dim=2, var=1, len_scale=10)
    srf = SRF(model, seed=20170519)  
    field = srf.structured([x, y])
    srf.plot()
    vx = srf.structured([x, y])
    vy = srf.structured([x, y])
    vx = torch.tensor(vx).cuda()
    vy = torch.tensor(vy).cuda()
    wz_init = simulator.curl(vx, vy)

    # Simulate and plot the vorticity fields
    # simulate(simulator, vx, vy, n_steps)
    simulate_vorticity_driven(simulator, wz_init, n_steps)