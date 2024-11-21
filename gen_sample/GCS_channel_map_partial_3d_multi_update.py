import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from modulus.models.fno import FNO
import h5py
# from torchmetrics.image import StructuralSimilarityIndexMeasure
from skimage.metrics import structural_similarity as ssim
# from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from matplotlib import colors

import torch.nn.functional as F
import sys
sys.path.append('../test')

'''
Can compute average of but then inversion result will look a bit mo probabilistic.
'''

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

# def plot_single(true1, path, cmap='Blues'):
#     plt.figure(figsize=(10, 10))
#     plt.rcParams.update({'font.size': 16})

#     # Apply the norm both to the image and the colorbar
#     ax = plt.imshow(true1, cmap=cmap)
#     plt.colorbar(ax, fraction=0.045, pad=0.06)
#     ax.set_xticks([])
#     ax.set_yticks([])

#     plt.savefig(path, dpi=150, bbox_inches='tight')
#     plt.close()

def plot_single(true1, path, cmap='Blues', vmin=None, vmax=None):
    plt.figure(figsize=(10, 10))
    plt.rcParams.update({'font.size': 16})
    
    # norm = colors.CenteredNorm()
    # Use a fixed range for vmin and vmax, if provided
    print("vmin", vmin, vmax)
    if vmin != 0:
        norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax) if vmin is not None and vmax is not None else colors.CenteredNorm()
    else:
        norm = colors.Normalize(vmin=vmin, vmax=vmax) if vmin is not None and vmax is not None else colors.CenteredNorm()
    
    # Initialize ax properly and plot the image
    fig, ax = plt.subplots()
    cax = ax.imshow(true1, cmap=cmap, norm=norm)
    
    # Add colorbar
    plt.colorbar(cax, ax=ax, fraction=0.045, pad=0.06)
    
    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Save the plot to the specified path
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_single_abs(true1, path, cmap='Blues', vmin=None, vmax=None):
    plt.figure(figsize=(10, 10))
    plt.rcParams.update({'font.size': 16})

    # Use Normalize to set the colorbar range to vmin and vmax
    norm = colors.Normalize(vmin=vmin, vmax=vmax) if vmin is not None and vmax is not None else colors.Normalize()

    # Plot the image and get the AxesImage object
    im = plt.imshow(true1, cmap=cmap, norm=norm)
    
    # Get the current axis and add the colorbar
    cbar = plt.colorbar(im, fraction=0.045, pad=0.06)
    
    # Set ticks on the axis
    ax = plt.gca()  # Get the current Axes object
    ax.set_xticks([])  # Remove x-axis ticks
    ax.set_yticks([])  # Remove y-axis ticks

    # Save the plot to the specified path
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_single(true1, path, cmap='Blues', vmin=None, vmax=None):
    plt.figure(figsize=(10, 10))
    plt.rcParams.update({'font.size': 16})
    
    # norm = colors.CenteredNorm()
    # Use a fixed range for vmin and vmax, if provided
    print("vmin", vmin, vmax)
    if vmin != 0:
        norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax) if vmin is not None and vmax is not None else colors.CenteredNorm()
    else:
        norm = colors.Normalize(vmin=vmin, vmax=vmax) if vmin is not None and vmax is not None else colors.CenteredNorm()
    
    # Initialize ax properly and plot the image
    fig, ax = plt.subplots()
    cax = ax.imshow(true1, cmap=cmap, norm=norm)
    
    # Add colorbar
    plt.colorbar(cax, ax=ax, fraction=0.045, pad=0.06)
    
    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Save the plot to the specified path
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_diff_with_shared_colorbar_all(figures, t, ssim_mse, ssim_pbi, path, cmap='Blues'):
    fig, axes = plt.subplots(1, 6, figsize=(35, 5))  # 1 row, 6 columns
    plt.rcParams.update({'font.size': 13})

    # Plot the first two subplots (with individual color bars)
    for i, ax in enumerate(axes[:2]):
        im = ax.imshow(figures[i], cmap='Blues')
        if i == 0:
            ax.set_title(f'True K0')
        elif i == 1:
            ax.set_title(f'H(K0)')
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.06)
        ax.set_xticks([])
        ax.set_yticks([])

    # Normalize color scale for the third and fourth subplots
    norm = plt.Normalize(vmin=np.min(figures[2:4]), vmax=np.max(figures[2:4]))  # Normalize color bar for axes[2] and axes[3]
    norm_2 = plt.Normalize(vmin=0, vmax=np.max(figures[4:]))  # Normalize color bar

    # Plot the third and fourth subplots with shared color bar
    for i, ax in enumerate(axes[2:4], 2):
        im = ax.imshow(figures[i], cmap='Blues', norm=norm)
        if i == 2:
            ax.set_title(f'FNO | SSIM={ssim_mse:.4f}')
        elif i == 3:
            ax.set_title(f'DeFINO | SSIM={ssim_pbi:.4f}')
            # Create a shared color bar for the third and fourth subplots
            fig.colorbar(im, ax=axes[2:4], fraction=0.02, pad=0.06)
        ax.set_xticks([])
        ax.set_yticks([])

    # Plot the last two subplots (with individual color bars)
    for i, ax in enumerate(axes[4:], 4):
        im = ax.imshow(figures[i], cmap=cmap, norm=norm_2)
        if i == 4:
            ax.set_title(f'Abs Diff FNO')
        elif i == 5:
            ax.set_title(f'Abs Diff DeFINO')
            fig.colorbar(im, ax=axes[4:], fraction=0.02, pad=0.06)
        ax.set_xticks([])
        ax.set_yticks([])

    # Save and close
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_multiple(figures, path, cmap='Blues'):
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))  # Create 2 rows and 4 columns
    plt.rcParams.update({'font.size': 16})

    for i, (true1, ax) in enumerate(zip(figures, axes.flat)):  # Flatten axes to loop through them
        norm = colors.CenteredNorm()
        im = ax.imshow(true1, cmap=cmap, norm=norm)
        ax.set_title(f'Time step {i+1}')
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04, norm=norm)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()

# def plot_multiple_abs(figures, path, cmap='Blues', vmin=None, vmax=None):
#     fig, axes = plt.subplots(2, 4, figsize=(20, 10))  # Create 2 rows and 4 columns
#     plt.rcParams.update({'font.size': 16})
#     if vmin != 0:
#         norm = colors.TwoSlopeNorm(vmin=vmin, vmax=vmax) if vmin is not None and vmax is not None else colors.CenteredNorm()
#     else:
#         norm = colors.Normalize(vmin=vmin, vmax=vmax) if vmin is not None and vmax is not None else colors.CenteredNorm()
    

#     for i, (true1, ax) in enumerate(zip(figures, axes.flat)):  # Flatten axes to loop through them
#         im = ax.imshow(true1, cmap=cmap, norm=norm)
#         ax.set_title(f'Time step {i+1}')
#         fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
#         ax.set_xticks([])
#         ax.set_yticks([])

#     plt.tight_layout()  # Adjust layout to prevent overlap
#     plt.savefig(path, dpi=150, bbox_inches='tight')
#     plt.close()

# def plot_multiple_abs(figures, path, cmap='Blues', vmin=None, vmax=None, mse=None):
#     fig, axes = plt.subplots(2, 4, figsize=(20, 10))  # Create 2 rows and 4 columns
#     plt.rcParams.update({'font.size': 16})
    
#     # Use Normalize to set the colorbar range to vmin and vmax
#     norm = colors.Normalize(vmin=vmin, vmax=vmax) if vmin is not None and vmax is not None else colors.Normalize()

#     for i, (true1, ax) in enumerate(zip(figures, axes.flat)):  # Flatten axes to loop through them
#         if mse == True:
#             true1 = abs(true1)
#         im = ax.imshow(true1, cmap=cmap, norm=norm)
#         ax.set_title(f'Time step {i+1}')
#         fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
#         ax.set_xticks([])  # Remove x-axis ticks
#         ax.set_yticks([])  # Remove y-axis ticks

#     plt.tight_layout()  # Adjust layout to prevent overlap
#     plt.savefig(path, dpi=150, bbox_inches='tight')
#     plt.close()

def plot_multiple_abs(figures, path, cmap='Blues', vmin=None, vmax=None, mse=None):
    # Select the 4th and the last figure from the figures list
    selected_figures = [figures[3], figures[-1]]

    fig, axes = plt.subplots(2, 1, figsize=(10, 10))  # Create 2 rows and 1 column
    plt.rcParams.update({'font.size': 16})
    
    # Use Normalize to set the colorbar range to vmin and vmax
    norm = colors.Normalize(vmin=vmin, vmax=vmax) if vmin is not None and vmax is not None else colors.Normalize()

    for i, (true1, ax) in enumerate(zip(selected_figures, axes)):  # Loop through the selected figures
        if mse == True:
            true1 = abs(true1)
        im = ax.imshow(true1, cmap=cmap, norm=norm)
        ax.set_title(f'Time Step {(i+1)*4}')
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
        ax.set_xticks([])  # Remove x-axis ticks
        ax.set_yticks([])  # Remove y-axis ticks

    plt.tight_layout()  # Adjust layout to prevent overlap
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
            ax.set_xticks([])
            ax.set_yticks([])

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
batch_size = 1
MSE_output, JAC_output = [], []
ssim_value = 0.
num_vec = 20
start_idx = 400
end_idx = 500
# ssim = StructuralSimilarityIndexMeasure()


num_col = 0
kernel_size = 11  # Example kernel size (should be odd)
sigma = 20.0  # Standard deviation of the Gaussian
learning_rate = 100.0 # [0.5, 1.0, 5.0, 20.0, 50.0, 100.0]
num_epoch = 101

# Load MSE FNO
MSE_model = FNO(
        in_channels=1,
        out_channels=1,
        decoder_layer_size=128,
        num_fno_layers=5,
        num_fno_modes=[8, 15, 15],
        padding=3,
        dimension=3,
        latent_channels=64
    ).to(device)

# load JAC FNO
JAC_model = FNO(
        in_channels=1,
        out_channels=1,
        decoder_layer_size=128,
        num_fno_layers=5,
        num_fno_modes=[8, 15, 15],
        padding=3,
        dimension=3,
        latent_channels=64
    ).to(device)

# JAC_path = f"../test_result/best_model_FNO_GCS_JAC.pth"
# JAC_path = f"../test_result/best_model_FNO_GCS_vec_{num_vec}_JAC_train=50.pth"
# C:\Users\D2A2\Diff_MultiPhysics\test_result\best_model_FNO_GCS_vec_10_JAC_train=5_eig=10.pth
# JAC_path = "../test_result/best_model_FNO_GCS_vec_10_JAC_train=5_eig=10.pth"
'''
num sample=3
Epoch 100 - PBI SSIM Full: 2.2954256577249, MSE SSIM Full: 2.2876617832768718
Epoch 100 - PBI MSE Full: 1931.91845703125, MSE MSE Full: 1980.6263427734375
Epoch 100 - PBI Forward Loss: 0.00045383395627141, MSE Forward Loss: 0.0005023191915825009
'''

JAC_path = "../test_result/best_model_FNO_GCS_vec_20_JAC_train=5.pth"
'''
num sample=3
Epoch 100 - PBI SSIM Full: 2.337537771488906, MSE SSIM Full: 2.2876617832768718
Epoch 100 - PBI MSE Full: 1658.1009521484375, MSE MSE Full: 1980.6263427734375
Epoch 100 - PBI Forward Loss: 0.0006113179260864854, MSE Forward Loss: 0.0005023191915825009
'''
JAC_model.load_state_dict(torch.load(JAC_path))
JAC_model.eval()

# MSE_path = f"../test_result/best_model_FNO_GCS_MSE.pth"
# MSE_path = f"../test_result/best_model_FNO_GCS_vec_1_MSE_train=50.pth"
MSE_path = f"../test_result/best_model_FNO_GCS_vec_0_MSE_train=5.pth"
MSE_model.load_state_dict(torch.load(MSE_path))
MSE_model.eval()

# Load input data K
with h5py.File('../FNO-NF.jl/data/training-data/cons=1e-5_delta=25_num_sample=10000_theta0=5.jld2', 'r') as f:
    print("Keys: %s" % f.keys())  # List all the datasets in the file
    K = f['perm'][:]
    set_x.append(K)

set_x = set_x[0]
set_x = torch.tensor(set_x)
org_x = set_x.detach()
set_x = set_x[start_idx-1:end_idx-1]  # indexing btw python and julia is different
set_x = set_x.unsqueeze(1).repeat(1, 8, 1, 1)  # Reshape [2000, 8, 64, 64]
set_x = set_x.reshape(-1, batch_size, 8, 64, 64)  # Reshape to batches


# Read the each file s_idx: sample index
for s_idx in range(start_idx, end_idx):

    with h5py.File(f'../gen_sample/num_obs_20/states_sample_{s_idx}_nobs_20.jld2', 'r') as f1:

        states_refs = f1['single_stored_object'][:]  # Load the array of object references
        states_tensors = []
        # Loop over the references, dereference them, and convert to tensors
        for ref in states_refs:
            # Dereference the object reference
            state_data = f1[ref][:]
            
            # Convert the dereferenced data to a PyTorch tensor
            state_tensor = torch.tensor(state_data)
            states_tensors.append(state_tensor)

        # set_y.append(S) 
        set_y.append(torch.stack(states_tensors).reshape(8, 64, 64))

# set_y = torch.stack(set_y[900:])
set_y = torch.stack(set_y)
org_set_y = set_y
plot_multiple_abs(org_set_y[0], f'GCS_partial/vec={num_vec}/S_org', cmap='Blues')
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

plot_multiple_abs(set_y[0], f'GCS_partial/vec={num_vec}/S_masked3', cmap='Blues')
set_y = set_y.reshape(-1, batch_size, 8, 64, 64)
num_batch = set_x.shape[0]
print("num_batch", num_batch, "batch size:", batch_size)

# Function for least squares posterior estimation, where the input_data is updated
def least_squares_posterior_estimation(model, input_data, true_data, model_type, learning_rate, batch_num, num_iterations=500):
    model.eval()  # Ensure the model is in evaluation mode (weights won't be updated)
    mse_loss = torch.nn.MSELoss()  # MSE loss function

    # Set the input data as a tensor that requires gradients (so it can be optimized)
    input_data = input_data.clone().detach().requires_grad_(True).to(device)
    posterior_set = []

    # Define an optimizer to update the input data (instead of the model parameters)
    optimizer = torch.optim.Adam([input_data], lr=learning_rate)
    losses = []
    min_mse, max_mse = [], []
    min_jac, max_jac = [], []

    for iteration in range(num_iterations):
        optimizer.zero_grad()  # Reset gradients
        output = model(input_data.unsqueeze(dim=1))
        output = output.squeeze()
        print("shape", output.shape, true_data.shape) #torch.Size([100, 1, 8, 64, 64]) torch.Size([100, 8, 64, 64])
        # output = torch.clamp(output, min=0, max=0.9)
        # print("input", input_data.shape) [batchsize, 8, 64, 64]
        if batch_num < 12:
            plot_multiple_abs(input_data.clone().detach().cpu()[0], f'GCS_partial/vec={num_vec}/lr={learning_rate}/batch_{batch_num}/iter_{model_type}_{iteration}', cmap='Blues')
        # mask is well operator here
        output = output #* mask[:100].cuda().float()
        # loss = mse_loss(output[:, :, 15:-15], true_data[:, :, 15:-15])
        # print("loss type:", model_type, torch.min(output), torch.max(output))
        for batch in [0]:
            if model_type == "MSE":
                plot_multiple_abs(output.detach().cpu().numpy(), f"GCS_partial/vec={num_vec}/lr={learning_rate}/S/{model_type}_{batch}_{iteration}", vmin=0., vmax=0.9, mse=True)
            else:
                plot_multiple_abs(output.detach().cpu().numpy(), f"GCS_partial/vec={num_vec}/lr={learning_rate}/S/{model_type}_{batch}_{iteration}", vmin=0., vmax=0.9)
            plot_multiple_abs(true_data.squeeze().detach().cpu().numpy(), f"GCS_partial/vec={num_vec}/lr={learning_rate}/S/True_{batch}", vmin=0., vmax=0.9)
        if model_type == "MSE":
            min_mse.append(torch.min(output).detach().cpu().numpy())
            max_mse.append(torch.max(output).detach().cpu().numpy())
        else:
            min_jac.append(torch.min(output).detach().cpu().numpy())
            max_jac.append(torch.max(output).detach().cpu().numpy())
        loss = mse_loss(output, true_data)
        losses.append(loss.item())
        # loss.backward(retain_graph=True)
        loss.backward()
        optimizer.step()
        # input_data.data = torch.clamp(input_data.data, min=0, max=135)
        input_data.data = torch.clamp(input_data.data, min=10, max=130)

        # Plot loss
        plt.figure(figsize=(10, 6))
        if model_type == "MSE":
            plt.plot(losses, label=f'FNO', color='red', marker='^')
        else:
            plt.plot(losses, label='DeFINO', color='blue', marker='o')
        plt.xlabel('Iterations', fontsize=15)
        plt.ylabel('Loss', fontsize=15)
        plt.legend(fontsize=15)
        plt.grid(True)
        plt.savefig(f'GCS_partial/vec={num_vec}/lr={learning_rate}/loss_plot_{learning_rate}_{num_epoch}_{model_type}_{num_col}.png')
        plt.close()

        # if iteration % 50 == 0:
        print(f"Iteration {iteration}, {model_type} Loss: {loss.item()}")
        posterior_set.append(input_data.clone().detach().cpu().numpy())
        plot_single_abs(input_data.clone().detach().cpu()[0].mean(dim=0), f'GCS_partial/vec={num_vec}/lr={learning_rate}/batch_{batch_num}/iter_{model_type}_{iteration}_afterclamp', cmap='Blues', vmin=0, vmax=130)

    # Return the optimized input data (permeability K)
    return posterior_set, losses, min_mse, min_jac, max_mse, max_jac  # Detach from the computational graph


# Perform least squares posterior estimation by updating input permeability K
# zero_X = apply_gaussian_smoothing(X, kernel_size, sigma)
# zero_X = torch.mean(torch.tensor(set_x).to(device).float(), dim=0)
# H(K0)
print(org_x.shape)
zero_X = torch.mean(org_x, dim=0).unsqueeze(dim=0) #[64, 64]
print("first", zero_X.shape)
zero_X = zero_X.unsqueeze(1).repeat(1, 8, 1, 1)  # Reshape [100, 8, 64, 64]
# print(zero_X.shape)
zero_X = zero_X.repeat(batch_size, 1, 1, 1).to(device)
# print(zero_X.shape)

# MLE: Iterate over epochs first
for epoch in range(0, num_epoch, 10):
    print(f"Epoch {epoch}: Performing least squares posterior estimation")

    ssim_all_batch_mse, ssim_all_batch_jac = 0., 0.
    mse_jac, mse_mse = 0.0, 0.0

    # run it for one time!!!
    if epoch == 0:
        posterior_estimate_jac_all, posterior_estimate_mse_all = [], []
        mse_loss_all, jac_loss_all = [], []
        for i in range(num_batch):
            print("batch", i)
            X = set_x[i].to(device).float()  # Input permeability [batch_size, 8, 64, 64]
            if i < 12:
                plot_multiple_abs(X[0].clone().detach().cpu(), f'GCS_partial/vec={num_vec}/lr={learning_rate}/batch_{i}/True_K', cmap='Blues')
            Y_true = set_y[i].to(device).float()  # True observation S [batch_size, 8, 64, 64]

            # Call MLE for each sample in the batch within the current epoch
            posterior_estimate_mse, mse_losses, min_mse, min_jac, max_mse, max_jac = least_squares_posterior_estimation(
                MSE_model, zero_X, Y_true, "MSE", learning_rate, i, num_iterations=num_epoch
            )
            posterior_estimate_jac, jac_losses, min_mse, min_jac, max_mse, max_jac = least_squares_posterior_estimation(
                JAC_model, zero_X, Y_true, "JAC", learning_rate, i, num_iterations=num_epoch
            )
            posterior_estimate_jac_all.append(posterior_estimate_jac)
            posterior_estimate_mse_all.append(posterior_estimate_mse)
            mse_loss_all.append(mse_losses)
            jac_loss_all.append(jac_losses)

        print("shape,", len(posterior_estimate_jac_all), len(posterior_estimate_mse_all), len(mse_loss_all), len(jac_loss_all))
        mse_loss_all = torch.tensor(mse_loss_all)
        jac_loss_all = torch.tensor(jac_loss_all)
        posterior_estimate_jac_all = torch.tensor(posterior_estimate_jac_all)
        posterior_estimate_mse_all = torch.tensor(posterior_estimate_mse_all)
        print(mse_loss_all.shape, jac_loss_all.shape) #torch.Size([num_batch, num_epoch]) torch.Size([2, 3])
        print(posterior_estimate_mse_all.shape, posterior_estimate_jac_all.shape) #torch.Size([2, 3, 5, 8, 64, 64]) [num_batch, num_epoch, batch_size, 8, 64, 64]
        mse_loss_all = torch.sum(mse_loss_all, dim=0)
        jac_loss_all = torch.sum(jac_loss_all, dim=0)
        # posterior_estimate_mse_all = posterior_estimate_mse_all.permute(1, 0, 2, 3, 4, 5)
        # posterior_estimate_jac_all = posterior_estimate_jac_all.permute(1, 0, 2, 3, 4, 5)

        # Plot loss for each batch per epoch
        plt.figure(figsize=(10, 6))
        plt.plot(mse_loss_all, label='MSE Model', color='red', marker='^')
        plt.plot(jac_loss_all, label='PBI Model', color='blue', marker='o')
        plt.xlabel('Iterations', fontsize=15)
        plt.ylabel('Loss', fontsize=15)
        plt.legend(fontsize=15)
        plt.grid(True)
        plt.savefig(f'GCS_partial/vec={num_vec}/lr={learning_rate}/loss_plot_{learning_rate}_all.png')

    # Calculations for SSIM and MSE, for every 50th epoch
    # torch.stack(posterior_estimate_jac).shape: [1, 5, 8, 64, 64]
    ssim_all_mse, ssim_all_jac = 0.0, 0.0
    print("post", torch.tensor(posterior_estimate_jac_all).shape)
    for i in range(num_batch):
        X = set_x[i].to(device).float()
        for b in range(batch_size):
            print("b", b)
            # mask
            mask_S = (Y_true[b][-1] != 0).int().detach().cpu()
            masked_X = X[b][-1].detach().cpu()
            masked_X[mask_S == 0] = 20
            masked_posterior_estimate_mse_all = posterior_estimate_mse_all[i][epoch][b].clone().mean(dim=0).detach().cpu().numpy()
            masked_posterior_estimate_mse_all[mask_S == 0] = 20
            # masked_posterior_estimate_mse_all[:15,:] = 20
            # masked_posterior_estimate_mse_all[45:,:] = 20
            masked_mse = masked_posterior_estimate_mse_all

            masked_posterior_estimate_jac_all = posterior_estimate_jac_all[i][epoch][b].clone().mean(dim=0).detach().cpu().numpy()
            masked_posterior_estimate_jac_all[mask_S == 0] = 20
            # masked_posterior_estimate_jac_all[:15,:] = 20
            # masked_posterior_estimate_jac_all[45:,:] = 20
            masked_jac = masked_posterior_estimate_jac_all

            # [num_batch, num_epoch, batch_size, 8, 64, 64]
            pe_mse = posterior_estimate_mse_all[i][epoch][b].clone().mean(dim=0).detach().cpu().numpy()
            pe_jac = posterior_estimate_jac_all[i][epoch][b].clone().mean(dim=0).detach().cpu().numpy()
            true_x = X[b][-1].detach().cpu()
            abs_diff_mse = abs(pe_mse - true_x.numpy())
            abs_diff_jac = abs(pe_jac - true_x.numpy())
            plot_single(pe_jac, f"debug_{b}.png", "magma")
            
            # metrics
            mse_mse += F.mse_loss(torch.tensor(pe_mse), true_x) # Compute MSE
            mse_jac += F.mse_loss(torch.tensor(pe_jac), true_x)
            
            # ssim_mse = ssim(pe_mse, masked_X.numpy(), data_range=pe_mse.max()-pe_mse.min())
            # ssim_jac = ssim(pe_jac, masked_X.numpy(), data_range=pe_jac.max()-pe_jac.min())
            ssim_mse = ssim(masked_mse, masked_X.numpy(), data_range=masked_mse.max()-masked_mse.min())
            ssim_jac = ssim(masked_jac, masked_X.numpy(), data_range=masked_jac.max()-masked_jac.min())
            
            path = f'GCS_partial/vec={num_vec}/lr={learning_rate}/posterior_{num_col}_{epoch}_{i}_{b}'
            plot_diff_with_shared_colorbar_all(
                [true_x, zero_X[b][-1].detach().cpu(), pe_mse, pe_jac, abs_diff_mse, abs_diff_jac], 
                epoch, ssim_mse, ssim_jac, path, cmap='magma'
            )
            # ssim_all_mse += ssim_mse
            # ssim_all_jac += ssim_jac
            # Save or visualize posterior results as needed
            print(f"Posterior estimate for epoch {epoch}, batch {i} completed.")
        ssim_all_batch_mse += ssim_mse
        ssim_all_batch_jac += ssim_jac
        print("MSE", ssim_mse, "JAC", ssim_jac, "All MSE", ssim_all_batch_mse, "All JAC", ssim_all_batch_jac)

    # Print SSIM metrics after each epoch
    print(f"number of samples:", end_idx-start_idx)
    print(f"Epoch {epoch} - PBI SSIM per batch: {ssim_all_jac}, MSE SSIM per batch: {ssim_all_mse}")
    print(f"Epoch {epoch} - PBI SSIM Full: {ssim_all_batch_jac}, MSE SSIM Full: {ssim_all_batch_mse}")
    print(f"Epoch {epoch} - PBI MSE Full: {mse_jac}, MSE MSE Full: {mse_mse}")
    print(f"Epoch {epoch} - PBI Forward Loss: {jac_losses[-1]}, MSE Forward Loss: {mse_losses[-1]}")
    
    # Plot min/max
    plt.figure(figsize=(10, 6))
    plt.plot(min_mse, label='MSE Model', color='red', marker='^')
    plt.plot(min_jac, label='PBI Model', color='blue', marker='o')
    plt.xlabel('Iterations', fontsize=15)
    plt.ylabel('Min', fontsize=15)
    plt.legend(fontsize=15)
    plt.grid(True)
    plt.savefig(f'GCS_partial/vec={num_vec}/lr={learning_rate}/min_{epoch}.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(max_mse, label='MSE Model', color='red', marker='^')
    plt.plot(max_jac, label='PBI Model', color='blue', marker='o')
    plt.xlabel('Iterations', fontsize=15)
    plt.ylabel('Max', fontsize=15)
    plt.legend(fontsize=15)
    plt.grid(True)
    plt.savefig(f'GCS_partial/vec={num_vec}/lr={learning_rate}/max_{epoch}.png')
    plt.close()


# masked

# num vec = 5
# PBI SSIM Full: 83.98503979173813 PBI forward losses 4.736108530778438e-05 posterior MSE: tensor(33788.2891)
# MSE SSIM Full: 84.1456410530959 MSE forward losses: 4.9569829570828006e-05 posterior MSE: tensor(33595.8516)

# num vec = 3
# PBI SSIM Full: 83.65200806990418 PBI forward losses 4.771849125972949e-05 posterior MSE: tensor(34878.3750)
# MSE SSIM Full: 84.1456410530959 MSE forward losses: 4.9569829570828006e-05 posterior MSE: tensor(33595.8516)

# unmasked vec = 3
# PBI SSIM Full: 66.28495136885891 PBI forward losses 4.771849125972949e-05 posterior MSE: tensor(33764.3516)
# MSE SSIM Full: 68.05121160517054 MSE forward losses: 4.9569829570828006e-05 posterior MSE: tensor(32645.8848)

# batch = 100, lr=100
# PBI SSIM Full: 59.48639890028781 PBI forward losses 0.0015426325844600797 posterior MSE: tensor(46993.8086)
# MSE SSIM Full: 58.171512016226515 MSE forward losses: 0.0015358485979959369 posterior MSE: tensor(48314.2461)


# 0
# PBI SSIM Full: 2.126361960578845 PBI forward losses 0.0011622852180153131 posterior MSE: tensor(4013.8411)
# MSE SSIM Full: 0.7244899201381758 MSE forward losses: 0.0012628999538719654 posterior MSE: tensor(13945.9629)

# number of samples: 10
# Epoch 50 - PBI SSIM per batch: 2.0783713925077816, MSE SSIM per batch: 2.0665565498970184
# Epoch 50 - PBI SSIM Full: 2.0783713925077816, MSE SSIM Full: 2.0665565498970184
# Epoch 50 - PBI MSE Full: 10433.447265625, MSE MSE Full: 11035.55859375
# Epoch 50 - PBI Forward Loss: 7.545437256339937e-05, MSE Forward Loss: 0.00038309264346025884