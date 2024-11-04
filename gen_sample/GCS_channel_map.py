import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from modulus.models.fno import FNO
import h5py
from torchmetrics.image import StructuralSimilarityIndexMeasure
import sys
sys.path.append('../test')


def plot_single(true1, path, cmap='Blues'):
    plt.figure(figsize=(10, 10))
    plt.rcParams.update({'font.size': 16})

    # Create a centered normalization around 0
    norm = colors.CenteredNorm()

    # Apply the norm both to the image and the colorbar
    ax = plt.imshow(true1, cmap=cmap, norm=norm)
    plt.colorbar(ax, fraction=0.045, pad=0.06, norm=norm)

    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_single_abs(true1, path, cmap='Blues'):
    plt.figure(figsize=(10, 10))
    plt.rcParams.update({'font.size': 16})

    # Apply the norm both to the image and the colorbar
    ax = plt.imshow(true1, cmap=cmap)
    plt.colorbar(ax, fraction=0.045, pad=0.06)

    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_diff_with_shared_colorbar(figures, path, cmap='Blues'):
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))  # 1 row, 5 columns
    plt.rcParams.update({'font.size': 13})

    # Plot the first three subplots (without shared color bar)
    for i, ax in enumerate(axes[:3]):
        im = ax.imshow(figures[i], cmap='Blues')
        if i == 0:
            ax.set_title(f'True')
        elif i == 1:
            ax.set_title(f'MSE')
        elif i == 2:
            ax.set_title(f'PBI')
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.06)

    # Plot the last two subplots with shared color bar
    norm = plt.Normalize(vmin=0, vmax=np.max(figures[3:]))  # Normalize color bar

    for i, ax in enumerate(axes[3:], 3):
        im = ax.imshow(figures[i], cmap=cmap, norm=norm)
        if i == 3:
            ax.set_title(f'Abs Diff MSE')
        elif i == 4:
            ax.set_title(f'Abs Diff PBI')
            # Create a shared color bar for the last two plots
            fig.colorbar(im, ax=axes[3:], fraction=0.02, pad=0.06)

    # Save and close
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()



def plot_multiple(figures, path, cmap='Blues'):
    fig, axes = plt.subplots(1, 8, figsize=(40, 5))  # Create 1 row and 8 columns
    plt.rcParams.update({'font.size': 16})

    for i, (true1, ax) in enumerate(zip(figures, axes)):
        im = ax.imshow(true1, cmap=cmap)
        if i < 8:
            ax.set_title(f'Time step {i+1}')
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.06)

    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_multiple_abs(figures, path, cmap='Blues'):
    fig, axes = plt.subplots(1, 8, figsize=(40, 5))  # Create 1 row and 8 columns
    plt.rcParams.update({'font.size': 16})

    for i, (true1, ax) in enumerate(zip(figures, axes)):
        im = ax.imshow(true1, cmap=cmap)
        if i < 8:
            ax.set_title(f'Time step {i+1}')
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.06)

    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_share_bar(data, path, cmap='magma'):

    # Set global font size
    plt.rcParams.update({'font.size': 12})  # Set desired font size globally

    # Create subplots
    fig, axs = plt.subplots(2, 8, figsize=(16, 4), sharex=True, sharey=True)
    vmax = max(torch.max(d) for d in data)

    # Plot each subplot
    for i, ax in enumerate(axs.flatten()):
        c = ax.imshow(data[i], vmin=0, vmax=vmax, cmap=cmap)
        if i < 8:
            ax.set_title(f'Time Step {i+1}')

    # Create a single colorbar
    cbar = fig.colorbar(c, ax=axs, orientation='vertical', fraction=0.02, pad=0.04)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()


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
loss_type = "MSE"
set_x, set_y = [], []
batch_size = 100
MSE_output = []
JAC_output = []
ssim_value = 0.
ssim = StructuralSimilarityIndexMeasure()

# Load MSE FNO
MSE_model = FNO(
    in_channels=8,
    out_channels=8,
    decoder_layer_size=128,
    num_fno_layers=6,
    num_fno_modes=[33, 33],
    padding=3,
    dimension=2,
    latent_channels=64
).to(device)

# load JAC FNO
JAC_model = FNO(
    in_channels=8,
    out_channels=8,
    decoder_layer_size=128,
    num_fno_layers=6,
    num_fno_modes=[33, 33],
    padding=3,
    dimension=2,
    latent_channels=64
).to(device)
JAC_path = f"../test_result/best_model_FNO_GCS_JAC_1000_20.pth"
JAC_model.load_state_dict(torch.load(JAC_path))
JAC_model.eval()

MSE_path = f"../test_result/best_model_FNO_GCS_MSE.pth"
MSE_model.load_state_dict(torch.load(MSE_path))
MSE_model.eval()

# Load input data K
with h5py.File('../FNO-NF.jl/data/training-data/cons=1e-5_delta=25_num_sample=10000_theta0=5.jld2', 'r') as f:
    print("Keys: %s" % f.keys())  # List all the datasets in the file
    K = f['perm'][:]
    set_x.append(K)

set_x = set_x[0]
set_x = torch.tensor(set_x)
set_x = set_x[:2000]  # Reduce the dataset size
set_x = set_x[1800:] # only test data
set_x = set_x.unsqueeze(1).repeat(1, 8, 1, 1)  # Reshape [2000, 8, 64, 64]
set_x = set_x.reshape(-1, batch_size, 8, 64, 64)  # Reshape to batches
if loss_type == "JAC":
    FNO_type = JAC_model
else:
    FNO_type = MSE_model

# Load output data S (observed data)
for s_idx in range(1, 2001):
    with h5py.File(f'../data/GCS_channel/FIM_Vjp_conc/conc_sample_{s_idx}_nobs_10.jld2', 'r') as f1:
        S = f1['single_stored_object'][:]
        set_y.append(S)

set_y = torch.tensor(set_y)
set_y = set_y[1800:]
set_y = set_y.reshape(-1, batch_size, 8, 64, 64)
num_batch = set_x.shape[0]

# Function for least squares posterior estimation, where the input_data is updated
def least_squares_posterior_estimation(model, input_data, true_data, num_iterations=500, learning_rate=1e-3):
    model.eval()  # Ensure the model is in evaluation mode (weights won't be updated)
    mse_loss = torch.nn.MSELoss()  # MSE loss function

    # Set the input data as a tensor that requires gradients (so it can be optimized)
    input_data = input_data.clone().detach().requires_grad_(True).to(device)

    # Define an optimizer to update the input data (instead of the model parameters)
    optimizer = torch.optim.Adam([input_data], lr=learning_rate)

    for iteration in range(num_iterations):
        optimizer.zero_grad()  # Reset gradients
        output = model(input_data)
        loss = mse_loss(output, true_data)
        loss.backward()
        optimizer.step()

        if iteration % 50 == 0:
            print(f"Iteration {iteration}, Loss: {loss.item()}")

    # Return the optimized input data (permeability K)
    return input_data.detach()  # Detach from the computational graph

# Perform posterior estimation using least squares on the MSE model
for i in range(num_batch):
    print(f"Batch {i}: Performing least squares posterior estimation")
    
    X = set_x[i].to(device).float()  # Input permeability
    print(X.shape)
    Y_true = set_y[i].to(device).float()  # True observation S
    print(Y_true.shape)

    # Perform least squares posterior estimation by updating input permeability K
    posterior_estimate_mse = least_squares_posterior_estimation(MSE_model, X, Y_true, num_iterations=5000)
    posterior_estimate_jac = least_squares_posterior_estimation(JAC_model, X, Y_true, num_iterations=5000)

    # Save or visualize posterior results as needed
    print(f"Posterior estimate for batch {i} completed.")

    # Plot or save the first and last estimates for comparison
    for b in range(batch_size):
        # plot_single_abs(X[b][-1].detach().cpu(), f'GCS_sample/{loss_type}/posterior_true_{i}_{b}', cmap='viridis')
        # plot_single_abs(posterior_estimate[b][-1].detach().cpu(), f'GCS_sample/{loss_type}/posterior_{i}_{b}', cmap='viridis')
        # plot_single_abs(abs(posterior_estimate[b][-1].detach().cpu() - X[b][-1].detach().cpu()), f'GCS_sample/{loss_type}/posterior_{i}_{b}_diff', cmap='magma')
        abs_diff_mse = abs(posterior_estimate_mse[b][-1].detach().cpu() - X[b][-1].detach().cpu())
        abs_diff_jac = abs(posterior_estimate_jac[b][-1].detach().cpu() - X[b][-1].detach().cpu())
        path = f'GCS_sample/both_20/posterior_{i}_{b}'
        plot_diff_with_shared_colorbar([X[b][-1].detach().cpu(), posterior_estimate_mse[b][-1].detach().cpu(), posterior_estimate_jac[b][-1].detach().cpu(), abs_diff_mse, abs_diff_jac], path, cmap='magma')
