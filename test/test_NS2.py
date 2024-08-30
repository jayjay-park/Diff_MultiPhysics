import torch
import torch.fft
import matplotlib.pyplot as plt

def simulate_NS(N=400, tEnd=1.0, dt=0.001, nu=0.001, plotRealTime=True):
    # Simulation parameters
    N = 64
    t = 0
    tEnd = 0.7
    dt = 0.001
    tOut = 0.01
    nu = 0.001
    plotRealTime = True

    # Domain [0,1] x [0,1]
    L = 1
    xlin = torch.linspace(0, L, N, device='cuda')
    xx, yy = torch.meshgrid(xlin, xlin, indexing='ij')

    # # Initial Condition (vortex)
    # Generate random parameters for the sinusoidal function
    freq_x = torch.normal(mean=4.0, std=1, size=(1,), device='cuda').item()
    freq_y = torch.normal(mean=2.0, std=1, size=(1,), device='cuda').item()
    phase_x = torch.normal(mean=1.0, std=1, size=(1,), device='cuda').item()
    phase_y = torch.normal(mean=-1.0, std=1, size=(1,), device='cuda').item()
    print(freq_x, freq_y, phase_x, phase_y)

    # Initial Condition (vortex) using sinusoidal function with random parameters
    vx = -torch.sin(freq_y * torch.pi * yy + phase_y)
    vy = torch.sin(freq_x * torch.pi * xx + phase_x)

    # Fourier Space Variables
    klin = 2.0 * torch.pi / L * torch.arange(-N//2, N//2, device='cuda')
    kmax = torch.max(klin)
    kx, ky = torch.meshgrid(klin, klin, indexing='ij')
    kx = torch.fft.ifftshift(kx)
    ky = torch.fft.ifftshift(ky)
    kSq = kx**2 + ky**2
    kSq_inv = 1.0 / kSq
    kSq_inv[kSq == 0] = 1

    # dealias with the 2/3 rule
    dealias = ((torch.abs(kx) < (2./3.)*kmax) & (torch.abs(ky) < (2./3.)*kmax)).float()

    # number of timesteps
    Nt = int(torch.ceil(torch.tensor(tEnd/dt)))

    # Save initial vorticity field as input
    wz_input = curl(vx, vy, kx, ky).cpu().numpy()
    
    # Main Loop
    for i in range(Nt):
        # Advection: rhs = -(v.grad)v
        dvx_x, dvx_y = grad(vx, kx, ky)
        dvy_x, dvy_y = grad(vy, kx, ky)
        
        rhs_x = -(vx * dvx_x + vy * dvx_y)
        rhs_y = -(vx * dvy_x + vy * dvy_y)
        
        rhs_x = apply_dealias(rhs_x, dealias)
        rhs_y = apply_dealias(rhs_y, dealias)

        vx += dt * rhs_x
        vy += dt * rhs_y
        
        # Poisson solve for pressure
        div_rhs = div(rhs_x, rhs_y, kx, ky)
        P = poisson_solve(div_rhs, kSq_inv)
        dPx, dPy = grad(P, kx, ky)
        
        # Correction (to eliminate divergence component of velocity)
        vx += - dt * dPx
        vy += - dt * dPy
        
        # Diffusion solve (implicit)
        vx = diffusion_solve(vx, dt, nu, kSq)
        vy = diffusion_solve(vy, dt, nu, kSq)
        
        # vorticity (for plotting)
        wz_output = curl(vx, vy, kx, ky).cpu().numpy()
        
        # update time
        t += dt
        
    # Save the plots of vx, vy, input, and output vorticity fields
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot vx field
    im1 = axes[0, 0].imshow(vx.cpu().numpy(), cmap='RdBu')
    axes[0, 0].set_title('vx Velocity Field')
    axes[0, 0].invert_yaxis()
    axes[0, 0].axis('off')
    fig.colorbar(im1, ax=axes[0, 0], orientation='vertical')

    # Plot vy field
    im2 = axes[0, 1].imshow(vy.cpu().numpy(), cmap='RdBu')
    axes[0, 1].set_title('vy Velocity Field')
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
    plt.savefig('velocity_and_vorticity_fields.png', dpi=300)
    
    return vx, vy, wz_input, wz_output

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

simulate_NS()