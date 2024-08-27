import torch
import matplotlib.pyplot as plt

def poisson_solve(rho, kSq_inv):
    V_hat = -(torch.fft.fftn(rho)) * kSq_inv
    V = torch.real(torch.fft.ifftn(V_hat))
    return V

def diffusion_solve(v, dt, nu, kSq):
    v_hat = (torch.fft.fftn(v)) / (1.0 + dt * nu * kSq)
    v = torch.real(torch.fft.ifftn(v_hat))
    return v

def grad(v, kx, ky):
    v_hat = torch.fft.fftn(v)
    dvx = torch.real(torch.fft.ifftn(1j * kx * v_hat))
    dvy = torch.real(torch.fft.ifftn(1j * ky * v_hat))
    return dvx, dvy

def div(vx, vy, kx, ky):
    dvx_x = torch.real(torch.fft.ifftn(1j * kx * torch.fft.fftn(vx)))
    dvy_y = torch.real(torch.fft.ifftn(1j * ky * torch.fft.fftn(vy)))
    return dvx_x + dvy_y

def curl(vx, vy, kx, ky):
    dvx_y = torch.real(torch.fft.ifftn(1j * ky * torch.fft.fftn(vx)))
    dvy_x = torch.real(torch.fft.ifftn(1j * kx * torch.fft.fftn(vy)))
    return dvy_x - dvx_y

def apply_dealias(f, dealias):
    f_hat = dealias * torch.fft.fftn(f)
    return torch.real(torch.fft.ifftn(f_hat))

def main():
    # Simulation parameters
    N = 400
    t = 0
    tEnd = 1
    dt = 0.001
    tOut = 0.01
    nu = 0.001
    plotRealTime = True

    # Domain [0,1] x [0,1]
    L = 1
    xlin = torch.linspace(0, L, N, device='cuda')
    xx, yy = torch.meshgrid(xlin, xlin, indexing='ij')

    # Initial Condition (vortex)
    vx = -torch.sin(2 * torch.pi * yy)
    vy = torch.sin(2 * torch.pi * xx * 2)

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

    # prep figure
    fig = plt.figure(figsize=(4,4), dpi=80)
    outputCount = 1

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
        wz = curl(vx, vy, kx, ky)
        
        # update time
        t += dt
        print(t)
        
        # plot in real time
        plotThisTurn = False
        if t + dt > outputCount * tOut:
            plotThisTurn = True
        if (plotRealTime and plotThisTurn) or (i == Nt-1):
            plt.cla()
            plt.imshow(wz.cpu().numpy(), cmap='RdBu')
            plt.clim(-20, 20)
            ax = plt.gca()
            ax.invert_yaxis()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)    
            ax.set_aspect('equal')    
            plt.pause(0.001)
            outputCount += 1
            
    # Save figure
    plt.savefig('navier-stokes-spectral-pytorch.png', dpi=240)
    plt.show()

if __name__ == "__main__":
    main()