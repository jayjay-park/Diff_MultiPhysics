import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from modulus.models.fno import FNO

import sys
sys.path.append('../test')
from generate_NS_org import *
from PINO_NS import *

# Set up the device and random seed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
print(f"Using device: {device}")

# Define simulation parameters
N = 64  # grid size
L = 2 * math.pi  # domain size
nu = 1e-3  # viscosity
num_init = 100
n_steps = 1  # number of simulation steps
loss_type = "JAC"


# Function to save data into a CSV file
# row is number of dataset, column is 2 x N x N {input, output}
# Function to save data into a NumPy file
def save_vorticity_to_npy(input_vorticities, output_vorticities, filename):
    for i, arr in enumerate(input_vorticities):
        print(f"Shape of input_vorticities[{i}]: {len(arr), len(arr[0]), len(arr[1])}")
    for i, arr in enumerate(output_vorticities):
        print(f"Shape of output_vorticities[{i}]: {len(arr), len(arr[0]), len(arr[1])}")

    # print(input_vorticities)
    # print(torch.tensor(input_vorticities[1]).shape)
    
    np.savez(filename, input_vorticities=np.array(input_vorticities), output_vorticities=np.array(output_vorticities))
    print(f"Saved dataset to {filename}")

# Function to save a vorticity plot for visual inspection
def save_vorticity_plot(vorticity_field, title, filename):
    plt.figure(figsize=(6, 6))
    plt.imshow(vorticity_field, cmap='jet', origin='lower')
    plt.colorbar(label='Vorticity')
    plt.title(title)
    plt.savefig(filename)
    plt.close()
    print(f"Saved figure to {filename}")

# Dataset generation function with plotting
def generate_dataset(num_samples, loss_type, num_init, time_step, nx=50, ny=50):
    input_vorticities, output_vorticities = [], []

    L1, L2 = 2 * math.pi, 2 * math.pi  # Domain size
    Re = 1000  # Reynolds number
    t = torch.linspace(0, 1, nx + 1, device="cuda")
    t = t[0:-1]
    X, Y = torch.meshgrid(t, t)
    forcing = 0.1 * (torch.sin(2 * math.pi * (X + Y)) + torch.cos(2 * math.pi * (X + Y)))


    if loss_type != "True":
        # Initialize FNO
        fno = FNO(
            in_channels=1,  # input vorticity
            out_channels=1,  # predicted vorticity at next time step
            decoder_layer_size=128,
            num_fno_layers=6,
            num_fno_modes=20,
            padding=3,
            dimension=2,
            latent_channels=64
        ).to(device)

        FNO_path = f"../test_result/best_model_FNO_NS_vort_{loss_type}.pth"
        fno.load_state_dict(torch.load(FNO_path))
        fno.eval()

    ns_solver = fno if loss_type != "True" else NavierStokes2d(nx, ny, L1=L1, L2=L2, device="cuda")

    num_iter = int(num_samples / num_init)
    print("Generating dataset...")

    for s in range(num_init):
        print(f"Generating data for initial condition {s + 1}/{num_init}")

        random_seed = 20 + s
        w = gaussian_random_field_2d((nx, ny), 20, random_seed)
        w_current = w.cuda()
        vorticity_data = [w_current.cpu().numpy()]
        # print("initial", len(vorticity_data))

        # Save the initial condition plot
        if s % 10 == 0:
            save_vorticity_plot(vorticity_data[0], f"Initial Vorticity Field (Init {s})", f"vorticity_init_{loss_type}_{s}.png")

        # Solve the NS for each time step
        for i in range(num_iter):
            if loss_type != "True":
                w_current = fno(w_current.reshape(1, 1, N, N).float()).squeeze()
                vorticity_data.append(w_current.detach().cpu().numpy())
                # print("w_current", w_current.shape)
            else:
                w_current = ns_solver(w_current, f=forcing, T=time_step, Re=Re)
                vorticity_data.append(w_current.detach().cpu().numpy())
            # print(i, len(vorticity_data), len(vorticity_data[0]), len(vorticity_data[1]))

        input_vorticities.append(vorticity_data[:-1])
        output_vorticities.append(vorticity_data[1:])
        # for i, arr in enumerate(input_vorticities):
        #     print(f"MSE Shape of input_vorticities[{i}]: {len(arr), len(arr[0]), len(arr[1])}")
        # for i, arr in enumerate(output_vorticities):
        #     print(f"MSE Shape of output_vorticities[{i}]: {len(arr), len(arr[0]), len(arr[1])}")

        # Save the final vorticity plot
        if s % 10 == 0:
            save_vorticity_plot(vorticity_data[-1].squeeze(), f"Final Vorticity Field (Init {s})", f"vorticity_final_{loss_type}_{s}.png")

    return input_vorticities, output_vorticities

# Save the dataset for each model type (MSE, JAC, True)
def save_all_datasets(N):
    # Parameters
    num_samples = 20000  # Total samples (20 initial conditions, 60 steps each)
    num_init = 400
    time_step = 0.05

    # Generate dataset for True solver
    # true_input, true_output = generate_dataset(num_samples, "True", num_init, time_step, nx=N, ny=N)
    # save_vorticity_to_npy(true_input, true_output, "true_solver_vorticity_64.npz")

    # Generate dataset for MSE model
    mse_input, mse_output = generate_dataset(num_samples, "MSE", num_init, time_step, nx=N, ny=N)
    save_vorticity_to_npy(mse_input, mse_output, "mse_model_vorticity_64.npz")

    # Generate dataset for JAC model
    # jac_input, jac_output = generate_dataset(num_samples, "JAC", num_init, time_step, nx=N, ny=N)
    # save_vorticity_to_npy(jac_input, jac_output, "jac_model_vorticity_64.npz")


# Execute the dataset generation and saving
save_all_datasets(N)
