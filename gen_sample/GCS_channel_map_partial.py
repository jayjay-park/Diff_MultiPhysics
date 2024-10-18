import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from modulus.models.fno import FNO
import h5py
from torchmetrics.image import StructuralSimilarityIndexMeasure
import torch.nn.functional as F
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

def plot_diff_with_shared_colorbar_all(figures, t, ssim_mse, ssim_pbi, path, cmap='Blues'):
    fig, axes = plt.subplots(1, 6, figsize=(30, 5))  # 1 row, 5 columns
    plt.rcParams.update({'font.size': 13})

    # Plot the first three subplots (without shared color bar)
    for i, ax in enumerate(axes[:4]):
        im = ax.imshow(figures[i], cmap='Blues')
        if i == 0:
            ax.set_title(f'True K0')
        elif i == 1:
            ax.set_title(f'H(K0)')
        elif i == 2:
            ax.set_title(f'MSE | iter={t+1} | SSIM={ssim_mse:.4f}')
        elif i == 3:
            ax.set_title(f'PBI | iter={t+1} | SSIM={ssim_pbi:.4f}')
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.06)

    # Plot the last two subplots with shared color bar
    norm = plt.Normalize(vmin=0, vmax=np.max(figures[3:]))  # Normalize color bar

    for i, ax in enumerate(axes[4:], 4):
        im = ax.imshow(figures[i], cmap=cmap, norm=norm)
        if i == 4:
            ax.set_title(f'Abs Diff MSE')
        elif i == 5:
            ax.set_title(f'Abs Diff PBI')
            # Create a shared color bar for the last two plots
            fig.colorbar(im, ax=axes[4:], fraction=0.02, pad=0.06)

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


num_col = 0
kernel_size = 11  # Example kernel size (should be odd)
sigma = 20.0  # Standard deviation of the Gaussian
learning_rate = 50.0 # [0.5, 1.0, 5.0, 20.0, 50.0, 100.0]
num_epoch = 2

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
JAC_path = f"../test_result/best_model_FNO_GCS_JAC.pth"
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
set_x = set_x[1900:] # only test data
# H(K0)
zero_X = torch.mean(torch.tensor(set_x).to(device).float(), dim=0)
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
set_y = set_y[1900:]
org_set_y = set_y
plot_multiple_abs(org_set_y[0], f'GCS_partial/S_org', cmap='Blues')
# three column.
first_col_idx = 0
middle_col_idx = set_y.size(2) // 2
last_col_idx = set_y.size(2) - 1

# Create a mask that keeps only the first, middle, and last columns
mask = torch.zeros_like(set_y)
if num_col == 3:
    mask[:, :, :, first_col_idx] = 1
    mask[:, :, :, middle_col_idx] = 1
    mask[:, :, :, last_col_idx] = 1
elif num_col == 2:
    mask[:, :, :, first_col_idx] = 1
    mask[:, :, :, last_col_idx] = 1
elif num_col == 1:
    mask[:, :, :, first_col_idx] = 1


# Apply the mask to set all other columns to 0


# set_y = set_y * mask
plot_multiple_abs(set_y[0], f'GCS_partial/S_masked3', cmap='Blues')
set_y = set_y.reshape(-1, batch_size, 8, 64, 64)
num_batch = set_x.shape[0]

# Function for least squares posterior estimation, where the input_data is updated
def least_squares_posterior_estimation(model, input_data, true_data, model_type, learning_rate, num_iterations=500):
    model.eval()  # Ensure the model is in evaluation mode (weights won't be updated)
    mse_loss = torch.nn.MSELoss()  # MSE loss function

    # Set the input data as a tensor that requires gradients (so it can be optimized)
    input_data = input_data.clone().detach().requires_grad_(True).to(device)
    posterior_set = []
    # input_data = torch.nn.Parameter(input_data)

    # Define an optimizer to update the input data (instead of the model parameters)
    optimizer = torch.optim.Adam([input_data], lr=learning_rate)
    losses = []

    for iteration in range(num_iterations):
        optimizer.zero_grad()  # Reset gradients
        output = model(input_data)
        plot_single_abs(input_data.detach().cpu()[0, 0], f'GCS_partial/iter_{model_type}_{iteration}', cmap='Blues')
        # mask is well operator here
        output = output #* mask[:100].cuda().float()
        # loss = mse_loss(output[:, :, 15:-15], true_data[:, :, 15:-15])
        loss = mse_loss(output, true_data)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

        # Plot loss
        plt.figure(figsize=(10, 6))
        if model_type == "MSE":
            plt.plot(losses, label=f'MSE Model', color='red', marker='^')
        else:
            plt.plot(losses, label='PBI Model', color='blue', marker='o')
        plt.xlabel('Iterations', fontsize=15)
        plt.ylabel('Loss', fontsize=15)
        plt.legend(fontsize=15)
        plt.grid(True)
        plt.savefig(f'GCS_partial/{model_type}_{num_col}/loss_plot_{learning_rate}_{num_epoch}.png')

        # if iteration % 50 == 0:
        print(f"Iteration {iteration}, Loss: {loss.item()}")
        posterior_set.append(input_data.detach())

    # Return the optimized input data (permeability K)
    return posterior_set, losses  # Detach from the computational graph

def gaussian_kernel(size: int, sigma: float):
    """Creates a 2D Gaussian kernel."""
    scale_factor = 1.

    # Create a 1D Gaussian kernel
    x = torch.arange(-size // 2 + 1., size // 2 + 1.)
    gauss = torch.exp(-x.pow(2) / (2 * sigma ** 2))
    gauss = gauss / gauss.sum()
    
    # Create a 2D Gaussian kernel by outer product
    kernel = gauss[:, None] @ gauss[None, :]
    kernel *= scale_factor

    return kernel

def apply_gaussian_smoothing(batch_matrix: torch.Tensor, kernel_size: int, sigma: float):
    """Applies Gaussian smoothing to a batch of input matrices using a Gaussian kernel."""
    # Create the Gaussian kernel
    kernel = gaussian_kernel(kernel_size, sigma)
    
    # Reshape the kernel to be compatible with 2D convolution
    kernel = kernel.unsqueeze(0).unsqueeze(0)  # Shape: 1 x 1 x kernel_size x kernel_size
    
    # Expand the kernel for each channel
    kernel = kernel.expand(8, 1, kernel_size, kernel_size).cuda()  # Shape: 8 x 1 x kernel_size x kernel_size
    
    # Apply 2D convolution to smooth each matrix in the batch
    smoothed_batch = F.conv2d(batch_matrix, kernel, padding=kernel_size // 2, groups=8)
    
    return smoothed_batch  # Output shape: batch_size x 8 x 64 x 64


# MLE
for i in range(num_batch):
    print(f"Batch {i}: Performing least squares posterior estimation")
    
    X = set_x[i].to(device).float() # /10 # Input permeability
    print(X.shape)
    Y_true = set_y[i].to(device).float() # / 10 # True observation S
    print(Y_true.shape)

    # Perform least squares posterior estimation by updating input permeability K
    # zero_X = apply_gaussian_smoothing(X, kernel_size, sigma)
    # zero_X = torch.mean(torch.tensor(set_x).to(device).float(), dim=0)
    posterior_estimate_mse, mse_losses = least_squares_posterior_estimation(MSE_model, zero_X, Y_true, "MSE", learning_rate, num_iterations=num_epoch)
    posterior_estimate_jac, jac_losses = least_squares_posterior_estimation(JAC_model, zero_X, Y_true, "JAC", learning_rate, num_iterations=num_epoch)

    # Plot loss
    plt.figure(figsize=(10, 6))
    plt.plot(mse_losses, label='MSE Model', color='red', marker='^')
    plt.plot(jac_losses, label='PBI Model', color='blue', marker='o')
    plt.xlabel('Iterations', fontsize=15)
    plt.ylabel('Loss', fontsize=15)
    plt.legend(fontsize=15)
    plt.grid(True)
    plt.savefig(f'GCS_partial/both_{num_col}/loss_plot_{learning_rate}_{num_epoch}.png')

    # Save or visualize posterior results as needed
    print(f"Posterior estimate for batch {i} completed.")
    ssim_all_mse = 0.
    ssim_all_jac = 0.
    # Plot every 50th epoch
    for t in range(0, num_epoch, 50):
        print(t)
        for b in range(batch_size): # for every test sample
            abs_diff_mse = abs(posterior_estimate_mse[t][b][-1].detach().cpu() - X[b][-1].detach().cpu())
            abs_diff_jac = abs(posterior_estimate_jac[t][b][-1].detach().cpu() - X[b][-1].detach().cpu())
            ssim_mse = ssim(posterior_estimate_mse[t][b][-1].detach().cpu().unsqueeze(dim=0).unsqueeze(0), X[b][-1].detach().cpu().unsqueeze(dim=0).unsqueeze(dim=0))
            ssim_jac = ssim(posterior_estimate_jac[t][b][-1].detach().cpu().unsqueeze(dim=0).unsqueeze(0), X[b][-1].detach().cpu().unsqueeze(dim=0).unsqueeze(dim=0))
            path = f'GCS_partial/both_{num_col}_200/posterior_{t}_{i}_{b}'
            print("shape", X[b][-1].shape, zero_X[b][-1].shape, posterior_estimate_mse[t][b][-1].shape)
            
            plot_diff_with_shared_colorbar_all([X[b][-1].detach().cpu(), zero_X[b][-1].detach().cpu(), posterior_estimate_mse[t][b][-1].detach().cpu(), posterior_estimate_jac[t][b][-1].detach().cpu(), abs_diff_mse, abs_diff_jac], t, ssim_mse, ssim_jac, path, cmap='magma')

    t = num_epoch - 1
    print(t)
    for b in range(batch_size): # for every test sample
        abs_diff_mse = abs(posterior_estimate_mse[t][b][-1].detach().cpu() - X[b][-1].detach().cpu())
        abs_diff_jac = abs(posterior_estimate_jac[t][b][-1].detach().cpu() - X[b][-1].detach().cpu())
        ssim_mse = ssim(posterior_estimate_mse[t][b][-1].detach().cpu().unsqueeze(dim=0).unsqueeze(0), X[b][-1].detach().cpu().unsqueeze(dim=0).unsqueeze(dim=0))
        ssim_jac = ssim(posterior_estimate_jac[t][b][-1].detach().cpu().unsqueeze(dim=0).unsqueeze(0), X[b][-1].detach().cpu().unsqueeze(dim=0).unsqueeze(dim=0))
        path = f'GCS_partial/both_{num_col}_200/posterior_{t}_{i}_{b}'
        ssim_all_mse += ssim_mse
        ssim_all_jac += ssim_jac
        print("shape", X[b][-1].shape, zero_X[b][-1].shape, posterior_estimate_mse[t][b][-1].shape)
        
        plot_diff_with_shared_colorbar_all([X[b][-1].detach().cpu(), zero_X[b][-1].detach().cpu(), posterior_estimate_mse[t][b][-1].detach().cpu(), posterior_estimate_jac[t][b][-1].detach().cpu(), abs_diff_mse, abs_diff_jac], t, ssim_mse, ssim_jac, path, cmap='magma')
    print("PBI SSIM Full:", ssim_all_jac)
    print("MSE SSIM Full:", ssim_all_mse)