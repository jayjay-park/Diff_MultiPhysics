import torch
import matplotlib.pyplot as plt

def one_step_vorticity(wz_input, kx, ky, dealias, dt, nu, kSq, kSq_inv):
    # Compute initial velocity components from vorticity
    vx, vy = compute_velocity_from_vorticity(wz_input, kx, ky, kSq_inv)
    
    # Perform one simulation step
    vx, vy = step(vx, vy, kx, ky, dealias, dt, nu, kSq, kSq_inv)
    
    # Compute the output vorticity
    wz_output = curl(vx, vy, kx, ky)
    
    return wz_output

def compute_velocity_from_vorticity(wz, kx, ky, kSq_inv):
    # Fourier space vorticity
    wz_hat = torch.fft.fftn(wz)
    
    # Compute velocity components in Fourier space
    vx_hat = 1j * ky * wz_hat * kSq_inv
    vy_hat = -1j * kx * wz_hat * kSq_inv
    
    # Transform back to real space
    vx = torch.real(torch.fft.ifftn(vx_hat))
    vy = torch.real(torch.fft.ifftn(vy_hat))
    
    return vx, vy

def simulate_NS(N=400, tEnd=1.0, dt=0.001, nu=0.001, plotRealTime=True):
    # Simulation parameters
    N = 64
    L = 1
    xlin = torch.linspace(0, L, N, device='cuda')
    xx, yy = torch.meshgrid(xlin, xlin, indexing='ij')

    # Initial Condition (vortex) using sinusoidal function with random parameters
    freq_x = torch.normal(mean=4.0, std=0.2, size=(1,), device='cuda').item()
    freq_y = torch.normal(mean=2.0, std=0.5, size=(1,), device='cuda').item()
    phase_x = torch.normal(mean=0., std=1, size=(1,), device='cuda').item()
    phase_y = torch.normal(mean=0., std=1, size=(1,), device='cuda').item()

    vx = -torch.sin(freq_y * torch.pi * yy + phase_y)
    vy = torch.sin(freq_x * torch.pi * xx + phase_x)

    # Fourier Space Variables
    klin = 2.0 * torch.pi / L * torch.arange(-N//2, N//2, device='cuda')
    kx, ky = torch.meshgrid(klin, klin, indexing='ij')
    kx = torch.fft.ifftshift(kx)
    ky = torch.fft.ifftshift(ky)
    kSq = kx**2 + ky**2
    kSq_inv = 1.0 / kSq
    kSq_inv[kSq == 0] = 1

    dealias = ((torch.abs(kx) < (2./3.)*torch.max(klin)) & (torch.abs(ky) < (2./3.)*torch.max(klin))).float()

    # Save initial vorticity field as input
    wz_input = curl(vx, vy, kx, ky)
    wz_input_init = wz_input.detach().cpu()
    
    # Perform one-step simulation to get output vorticity
    # wz_output = one_step_vorticity(wz_input, kx, ky, dealias, dt, nu, kSq, kSq_inv)
    for i in range(700):
        wz_output = one_step_vorticity(wz_input, kx, ky, dealias, dt, nu, kSq, kSq_inv)
        wz_input = wz_output

    # Plot and save the results
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    im1 = axes[0].imshow(wz_input_init, cmap='RdBu')
    axes[0].set_title('Input Vorticity')
    axes[0].invert_yaxis()
    axes[0].axis('off')
    fig.colorbar(im1, ax=axes[0], orientation='vertical')

    im2 = axes[1].imshow(wz_output.detach().cpu(), cmap='RdBu')
    axes[1].set_title('Output Vorticity (after 1 step)')
    axes[1].invert_yaxis()
    axes[1].axis('off')
    fig.colorbar(im2, ax=axes[1], orientation='vertical')

    plt.tight_layout()
    plt.savefig('one_step_vorticity.png', dpi=300)
    
    return wz_input, wz_output

# Helper Functions
def grad(v, kx, ky):
    v_hat = torch.fft.fftn(v)
    dvx = torch.real(torch.fft.ifftn(1j * kx * v_hat))
    dvy = torch.real(torch.fft.ifftn(1j * ky * v_hat))
    return dvx, dvy

def div(vx, vy, kx, ky):
    return torch.real(torch.fft.ifftn(1j * kx * torch.fft.fftn(vx))) + \
           torch.real(torch.fft.ifftn(1j * ky * torch.fft.fftn(vy)))

def curl(vx, vy, kx, ky):
    return torch.real(torch.fft.ifftn(1j * ky * torch.fft.fftn(vx))) - \
           torch.real(torch.fft.ifftn(1j * kx * torch.fft.fftn(vy)))

def poisson_solve(rho, kSq_inv):
    rho_hat = torch.fft.fftn(rho)
    phi_hat = -rho_hat * kSq_inv
    return torch.real(torch.fft.ifftn(phi_hat))

def diffusion_solve(v, dt, nu, kSq):
    v_hat = torch.fft.fftn(v)
    v_hat = v_hat / (1.0 + dt * nu * kSq)
    return torch.real(torch.fft.ifftn(v_hat))

def apply_dealias(f, dealias):
    f_hat = dealias * torch.fft.fftn(f)
    return torch.real(torch.fft.ifftn(f_hat))

def step(vx, vy, kx, ky, dealias, dt, nu, kSq, kSq_inv):
    dvx_x, dvx_y = grad(vx, kx, ky)
    dvy_x, dvy_y = grad(vy, kx, ky)

    rhs_x = -(vx * dvx_x + vy * dvx_y)
    rhs_y = -(vx * dvy_x + vy * dvy_y)

    rhs_x = apply_dealias(rhs_x, dealias)
    rhs_y = apply_dealias(rhs_y, dealias)

    vx += dt * rhs_x
    vy += dt * rhs_y

    div_rhs = div(rhs_x, rhs_y, kx, ky)
    P = poisson_solve(div_rhs, kSq_inv)
    dPx, dPy = grad(P, kx, ky)

    vx += - dt * dPx
    vy += - dt * dPy

    vx = diffusion_solve(vx, dt, nu, kSq)
    vy = diffusion_solve(vy, dt, nu, kSq)

    return vx, vy

# Run simulation
simulate_NS()
