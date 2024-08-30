import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class NavierStokesSimulator(nn.Module):
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

        kmax = torch.max(klin)
        self.dealias = ((torch.abs(self.kx) < (2./3.)*kmax) & (torch.abs(self.ky) < (2./3.)*kmax)).float()

    def forward(self, wz):
        wz_hat = torch.fft.fftn(wz)
        psi_hat = -wz_hat * self.kSq_inv
        u_hat = 1j * self.ky * psi_hat
        v_hat = -1j * self.kx * psi_hat
        
        u = torch.real(torch.fft.ifftn(u_hat))
        v = torch.real(torch.fft.ifftn(v_hat))
        
        dwz_dx, dwz_dy = self.grad(wz)
        adv = -(u * dwz_dx + v * dwz_dy)
        
        adv = self.apply_dealias(adv)
        
        wz_new = wz + self.dt * (adv + self.nu * self.laplacian(wz))
        
        return wz_new

    def grad(self, f):
        f_hat = torch.fft.fftn(f)
        return (
            torch.real(torch.fft.ifftn(1j * self.kx * f_hat)),
            torch.real(torch.fft.ifftn(1j * self.ky * f_hat))
        )

    def laplacian(self, f):
        f_hat = torch.fft.fftn(f)
        return torch.real(torch.fft.ifftn(-self.kSq * f_hat))

    def apply_dealias(self, f):
        f_hat = self.dealias * torch.fft.fftn(f)
        return torch.real(torch.fft.ifftn(f_hat))

    def vorticity_to_velocity(self, wz):
        wz_hat = torch.fft.fftn(wz)
        psi_hat = -wz_hat * self.kSq_inv
        u_hat = 1j * self.ky * psi_hat
        v_hat = -1j * self.kx * psi_hat
        return torch.real(torch.fft.ifftn(u_hat)), torch.real(torch.fft.ifftn(v_hat))

def compute_jacobian(simulator, wz):
    def simulation_step(input_tensor):
        return simulator(input_tensor)

    return torch.autograd.functional.jacobian(simulation_step, wz)

def velocity_to_vorticity(vx, vy):
    dvx_dy = torch.gradient(vx, dim=1)[0]
    dvy_dx = torch.gradient(vy, dim=0)[0]
    return dvx_dy - dvy_dx

def simulate_and_plot(simulator, wz, n_steps):
    wz_input = wz.cpu().numpy()

    for _ in range(n_steps):
        wz = simulator(wz)

    wz_output = wz.cpu().numpy()
    vx, vy = simulator.vorticity_to_velocity(wz)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    im1 = axes[0, 0].imshow(vx.cpu().numpy(), cmap='RdBu')
    axes[0, 0].set_title('vx Velocity Field')
    axes[0, 0].invert_yaxis()
    axes[0, 0].axis('off')
    fig.colorbar(im1, ax=axes[0, 0], orientation='vertical')

    im2 = axes[0, 1].imshow(vy.cpu().numpy(), cmap='RdBu')
    axes[0, 1].set_title('vy Velocity Field')
    axes[0, 1].invert_yaxis()
    axes[0, 1].axis('off')
    fig.colorbar(im2, ax=axes[0, 1], orientation='vertical')

    im3 = axes[1, 0].imshow(wz_input, cmap='RdBu')
    axes[1, 0].set_title('Input Vorticity')
    axes[1, 0].invert_yaxis()
    axes[1, 0].axis('off')
    fig.colorbar(im3, ax=axes[1, 0], orientation='vertical')

    im4 = axes[1, 1].imshow(wz_output, cmap='RdBu')
    axes[1, 1].set_title('Output Vorticity')
    axes[1, 1].invert_yaxis()
    axes[1, 1].axis('off')
    fig.colorbar(im4, ax=axes[1, 1], orientation='vertical')

    plt.tight_layout()
    plt.savefig('vorticity_and_velocity_fields2.png', dpi=300)

# Usage
N = 64  # Grid size
L = 1.0  # Domain length
dt = 0.001  # Time step
nu = 0.001  # Viscosity
n_steps = int(torch.ceil(torch.tensor(0.6 / dt)).item())  # Number of time steps to simulate
print("steps", n_steps)
simulator = NavierStokesSimulator(N, L, dt, nu).cuda()

# Initialize with random initial conditions
xlin = torch.linspace(0, L, N, device='cuda')
xx, yy = torch.meshgrid(xlin, xlin, indexing='ij')



freq_x = torch.normal(mean=4.0, std=0.5, size=(1,), device='cuda').item()
freq_y = torch.normal(mean=2.0, std=0.5, size=(1,), device='cuda').item()
phase_x = torch.normal(mean=0.0, std=1., size=(1,), device='cuda').item()
phase_y = torch.normal(mean=0.0, std=1., size=(1,), device='cuda').item()
print(freq_x, freq_y, phase_x, phase_y)

# Initial Condition (vortex) using sinusoidal function with random parameters
vx = -torch.sin(freq_y * torch.pi * yy + phase_y)
vy = torch.sin(freq_x * torch.pi * xx + phase_x)

# Convert initial velocity to vorticity
wz = velocity_to_vorticity(vx, vy)

# Simulate and plot the vorticity fields
simulate_and_plot(simulator, wz, n_steps)

# Compute Jacobian
jacobian = compute_jacobian(simulator, wz)

print("Jacobian shape:", jacobian.shape)
print("Jacobian mean:", jacobian.mean().item())
print("Jacobian std:", jacobian.std().item())
