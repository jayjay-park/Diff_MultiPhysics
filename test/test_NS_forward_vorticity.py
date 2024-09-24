import torch
import matplotlib.pyplot as plt
from modulus.models.fno import FNO
from generate_NS_org import NavierStokesSimulator

# Step 1: Set up the environment
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
print(f"Using device: {device}")

# Step 2: Define simulation parameters
N = 64  # grid size
L = 1.0  # domain size
dt = 0.001  # time step
nu = 1e-3  # viscosity
n_steps = 100  # number of simulation steps

# Step 3: Initialize NavierStokesSimulator and FNO model
simulator = NavierStokesSimulator(N, L, dt, nu).to(device)
xlin = torch.linspace(0, L, N, device=device)
xx, yy = torch.meshgrid(xlin, xlin, indexing='ij')

fno = FNO(
    in_channels=2,  # vx and vy
    out_channels=2,  # predicted vx and vy
    decoder_layer_size=128,
    num_fno_layers=6,
    num_fno_modes=20,
    padding=3,
    dimension=2,
    latent_channels=64
).to(device)

# Step 4: Load trained FNO model
loss_type = "Dissipative"
# FNO_path = f"../test_result/best_model_FNO_NS_{loss_type}_nx_64.pth"
FNO_path = f"../test_result/best_model_FNO_NS_{loss_type}.pth"
fno.load_state_dict(torch.load(FNO_path))
fno.eval()


# Step 6: Generate synthetic data for multiple time steps
# def generate_synthetic_data(n_steps):
#     freq_x = torch.normal(mean=4.0, std=0.3, size=(1,), device=device).item()
#     freq_y = torch.normal(mean=2.0, std=0.5, size=(1,), device=device).item()
#     phase_x = torch.normal(mean=0.0, std=1., size=(1,), device=device).item()
#     phase_y = torch.normal(mean=0.0, std=1., size=(1,), device=device).item()

#     vx = -torch.sin(freq_y * torch.pi * yy + phase_x)
#     vy = torch.sin(freq_x * torch.pi * xx + phase_y)
#     wz_true = simulator.curl(vx, vy)
    
#     vx_steps = [vx]
#     vy_steps = [vy]
#     wz_true_steps = [wz_true]
#     wz_pred_steps = [simulator.curl(vx, vy)]
    
#     for _ in range(n_steps - 1):
#         input_field = torch.stack([vx, vy], dim=0).unsqueeze(0)
#         output_field = fno(input_field).squeeze(0)
#         vx, vy = output_field[0], output_field[1]
#         wz_pred_steps.append(simulator.curl(vx, vy))
        
#         vx_true, vy_true = simulator(vx, vy)
#         wz_true = simulator.curl(vx_true, vy_true)
        
#         vx_steps.append(vx)
#         vy_steps.append(vy)
#         wz_true_steps.append(wz_true)
    
#     return torch.stack(wz_true_steps), torch.stack(wz_pred_steps)

# Step 6: Generate synthetic data using only the FNO model for multiple time steps
def generate_synthetic_data(n_steps, model):
    freq_x = torch.normal(mean=4.0, std=0.3, size=(1,), device=device).item()
    freq_y = torch.normal(mean=2.0, std=0.5, size=(1,), device=device).item()
    phase_x = torch.normal(mean=0.0, std=1., size=(1,), device=device).item()
    phase_y = torch.normal(mean=0.0, std=1., size=(1,), device=device).item()

    vx = -torch.sin(freq_y * torch.pi * yy + phase_x)
    vy = torch.sin(freq_x * torch.pi * xx + phase_y)
    
    vx_steps = [vx]
    vy_steps = [vy]
    wz_pred_steps = [simulator.curl(vx, vy)]
    
    for _ in range(n_steps - 1):
        if model == "FNO":
            input_field = torch.stack([vx, vy], dim=0).unsqueeze(0)
            output_field = fno(input_field).squeeze(0)
            vx, vy = output_field[0], output_field[1]
        else:
            vx, vy = simulator(vx, vy)
        wz_pred_steps.append(simulator.curl(vx, vy))
        
        vx_steps.append(vx)
        vy_steps.append(vy)
    
    return torch.stack(vx_steps), torch.stack(vy_steps), torch.stack(wz_pred_steps)



# Generate data
_, _, predicted_wz = generate_synthetic_data(n_steps, "FNO")
_, _, true_wz = generate_synthetic_data(n_steps, "TRUE")

# Step 7: Visualize the results for selected time steps
time_steps_to_plot = [0, 4, 9, 14, 19, 24, 29, 34, 39, 44, 49, 54, 59, 64, 69, 74, 79, 84, 89, 94, 99] # Plot initial, middle, and final time steps
fig, axs = plt.subplots(len(time_steps_to_plot), 2, figsize=(12, 7*len(time_steps_to_plot)))

for i, t in enumerate(time_steps_to_plot):
    # Plot true vorticity
    im0 = axs[i, 0].imshow(true_wz[t].detach().cpu().numpy(), cmap='RdBu_r')
    axs[i, 0].set_title(f'True Vorticity (t={t})')
    plt.colorbar(im0, ax=axs[i, 0], fraction=0.046, pad=0.04)

    # Plot predicted vorticity
    im1 = axs[i, 1].imshow(predicted_wz[t].detach().cpu().numpy(), cmap='RdBu_r')
    axs[i, 1].set_title(f'Predicted Vorticity (t={t})')
    plt.colorbar(im1, ax=axs[i, 1], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig(f"../test_result/Forward_Simulation_Vorticity_MultiStep_{loss_type}.png")
plt.show()
