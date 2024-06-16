import torch
import torch.nn.functional as F
import numpy as np

class Setup_Param:
    def __init__(self, N, L, N_KL, obs_ΔN, N_θ, d=2.0, τ=3.0, seed=123):
        self.N = N
        self.L = L
        self.Δx = L / (N - 1)
        self.xx = torch.linspace(0, L, N)
        
        self.logκ_2d, self.φ, self.λ, self.θ_ref = self.generate_θ_KL(N_KL, d, τ, seed)
        self.f_2d = self.compute_f_2d(self.xx)
        
        self.x_locs = torch.arange(obs_ΔN, N - obs_ΔN, obs_ΔN)
        self.y_locs = torch.arange(obs_ΔN, N - obs_ΔN, obs_ΔN)
        self.N_y = len(self.x_locs) * len(self.y_locs)
        
        self.θ_names = ["logκ"]
        self.N_θ = N_θ
        self.d = d
        self.τ = τ
        self.N_KL = N_KL

    def compute_f_2d(self, yy):
        N = len(yy)
        f_2d = torch.zeros(N, N)
        for i in range(N):
            if yy[i] <= 4/6:
                f_2d[:, i] = 1000.0
            elif 4/6 < yy[i] <= 5/6:
                f_2d[:, i] = 2000.0
            elif yy[i] > 5/6:
                f_2d[:, i] = 3000.0
        return f_2d

    def compute_seq_pairs(self, N_KL):
        pairs = []
        trunc_Nx = int(np.sqrt(2 * N_KL)) + 1
        for i in range(trunc_Nx + 1):
            for j in range(trunc_Nx + 1):
                if i == 0 and j == 0:
                    continue
                pairs.append((i, j))
        pairs = sorted(pairs, key=lambda x: x[0]**2 + x[1]**2)
        return torch.tensor(pairs[:N_KL])

    def generate_θ_KL(self, N_KL, d=2.0, τ=3.0, seed=123):
        torch.manual_seed(seed)
        N = len(self.xx)
        X, Y = torch.meshgrid(self.xx, self.xx)
        
        seq_pairs = self.compute_seq_pairs(N_KL)
        
        φ = torch.zeros(N_KL, N, N)
        λ = torch.zeros(N_KL)
        
        for i in range(N_KL):
            if seq_pairs[i, 0] == 0 and seq_pairs[i, 1] == 0:
                φ[i, :, :] = 1.0
            elif seq_pairs[i, 0] == 0:
                φ[i, :, :] = np.sqrt(2) * torch.cos(np.pi * seq_pairs[i, 1] * Y)
            elif seq_pairs[i, 1] == 0:
                φ[i, :, :] = np.sqrt(2) * torch.cos(np.pi * seq_pairs[i, 0] * X)
            else:
                φ[i, :, :] = 2 * torch.cos(np.pi * seq_pairs[i, 0] * X) * torch.cos(np.pi * seq_pairs[i, 1] * Y)
            
            λ[i] = (np.pi**2 * (seq_pairs[i, 0]**2 + seq_pairs[i, 1]**2) + τ**2)**(-d)
        
        θ_ref = torch.randn(N_KL)
        logκ_2d = torch.zeros(N, N)
        for i in range(N_KL):
            logκ_2d += θ_ref[i] * torch.sqrt(λ[i]) * φ[i, :, :]
        
        return logκ_2d, φ, λ, θ_ref

    def compute_logκ_2d(self, θ):
        N, N_KL = self.N, self.N_KL
        λ, φ = self.λ, self.φ
        N_θ = len(θ)
        
        logκ_2d = torch.zeros(N, N)
        for i in range(N_θ):
            logκ_2d += θ[i] * torch.sqrt(λ[i]) * φ[i, :, :]
        
        return logκ_2d

    def compute_dκ_dθ(self, θ):
        N, N_KL = self.N, self.N_KL
        λ, φ = self.λ, self.φ
        N_θ = len(θ)
        
        logκ_2d = torch.zeros(N * N)
        dκ_dθ = torch.zeros(N * N, N_θ)

        for i in range(N_θ):
            logκ_2d += (θ[i] * torch.sqrt(λ[i]) * φ[i, :, :]).flatten()
        
        for i in range(N_θ):
            dκ_dθ[:, i] = (torch.sqrt(λ[i]) * φ[i, :, :]).flatten() * torch.exp(logκ_2d)
        
        return dκ_dθ

    def ind(self, ix, iy):
        return (ix - 1) + (iy - 2) * (self.N - 2)

    def ind_all(self, ix, iy):
        return ix + (iy - 1) * self.N

    def solve_Darcy_2D(self, κ_2d):
        N, L = self.N, self.L

        # Compute grid spacing
        Δx = Δy = L / (N - 1)

        # Define the number of interior points
        N_i = (N - 2) ** 2

        # Initialize the dense matrix for the Laplace operator
        A = torch.zeros((N_i, N_i))

        # Generate the dense matrix for the Laplace operator
        for i in range(1, N - 1):
            for j in range(1, N - 1):
                row = (i - 1) * (N - 2) + (j - 1)

                # Center
                A[row, row] = -4 / (Δx * Δy) * κ_2d[i, j]

                # Right
                if j < N - 2:
                    A[row, row + 1] = κ_2d[i, j + 1] / (Δx * Δy)

                # Left
                if j > 1:
                    A[row, row - 1] = κ_2d[i, j - 1] / (Δx * Δy)

                # Up
                if i > 1:
                    A[row, row - (N - 2)] = κ_2d[i - 1, j] / (Δx * Δy)

                # Down
                if i < N - 2:
                    A[row, row + (N - 2)] = κ_2d[i + 1, j] / (Δx * Δy)

        # Right-hand side vector
        b = torch.zeros(N_i)

        # Solve the dense linear system using torch.linalg.solve
        x = torch.linalg.solve(A, b)

        # Reshape the solution to a 2D field
        h_2d = torch.zeros(N, N)
        h_2d[1:N-1, 1:N-1] = x.view(N-2, N-2)

        return h_2d


    def θ_posterior(self, θ):
        f_2d = self.f_2d
        N = self.N
        
        logκ_2d = self.compute_logκ_2d(θ)
        κ_2d = torch.exp(logκ_2d)
        u_2d = self.solve_Darcy_2D(κ_2d)
        
        ux_obs = torch.zeros(self.N_y)
        
        for i in range(self.N_y):
            ux_obs[i] = u_2d[self.x_locs[i], self.y_locs[i]]
        
        return ux_obs


# Initialize the parameters
N = 10
L = 1.0
N_KL = 5
obs_ΔN = 2
N_θ = 3
d = 2.0
τ = 3.0
seed = 123

# Create an instance of the Setup_Param class
params = Setup_Param(N, L, N_KL, obs_ΔN, N_θ, d, τ, seed)

# Use the compute_logκ_2d method with a sample θ
θ = torch.randn(N_KL)
logκ_2d = params.compute_logκ_2d(θ)
print("logκ_2d:", logκ_2d)

# Use the solve_Darcy_2D method
κ_2d = torch.exp(logκ_2d)
h_2d = params.solve_Darcy_2D(κ_2d)
print("h_2d:", h_2d)

# Compute the posterior
ux_obs = params.θ_posterior(θ)
print("Observed u_x:", ux_obs)

