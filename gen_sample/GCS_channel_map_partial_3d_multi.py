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

def plot_single(true1, path, cmap='Blues'):
    plt.figure(figsize=(10, 10))
    plt.rcParams.update({'font.size': 16})

    # Apply the norm both to the image and the colorbar
    ax = plt.imshow(true1, cmap=cmap)
    plt.colorbar(ax, fraction=0.045, pad=0.06)

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


def plot_diff_with_shared_colorbar_all(figures, t, ssim_mse, ssim_pbi, path, cmap='Blues'):
    fig, axes = plt.subplots(1, 6, figsize=(30, 5))  # 1 row, 6 columns
    plt.rcParams.update({'font.size': 13})

    # Plot the first two subplots (with individual color bars)
    for i, ax in enumerate(axes[:2]):
        im = ax.imshow(figures[i], cmap='Blues')
        if i == 0:
            ax.set_title(f'True K0')
        elif i == 1:
            ax.set_title(f'H(K0)')
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.06)

    # Normalize color scale for the third and fourth subplots
    norm = plt.Normalize(vmin=np.min(figures[2:4]), vmax=np.max(figures[2:4]))  # Normalize color bar for axes[2] and axes[3]
    norm_2 = plt.Normalize(vmin=0, vmax=np.max(figures[4:]))  # Normalize color bar

    # Plot the third and fourth subplots with shared color bar
    for i, ax in enumerate(axes[2:4], 2):
        im = ax.imshow(figures[i], cmap='Blues', norm=norm)
        if i == 2:
            ax.set_title(f'MSE | iter={t+1} | SSIM={ssim_mse:.4f}')
        elif i == 3:
            ax.set_title(f'PBI | iter={t+1} | SSIM={ssim_pbi:.4f}')
            # Create a shared color bar for the third and fourth subplots
            fig.colorbar(im, ax=axes[2:4], fraction=0.02, pad=0.06)

    # Plot the last two subplots (with individual color bars)
    for i, ax in enumerate(axes[4:], 4):
        im = ax.imshow(figures[i], cmap=cmap, norm=norm_2)
        if i == 4:
            ax.set_title(f'Abs Diff MSE')
        elif i == 5:
            ax.set_title(f'Abs Diff PBI')
            fig.colorbar(im, ax=axes[4:], fraction=0.02, pad=0.06)

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

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_multiple_abs(figures, path, cmap='Blues'):
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))  # Create 2 rows and 4 columns
    plt.rcParams.update({'font.size': 16})

    for i, (true1, ax) in enumerate(zip(figures, axes.flat)):  # Flatten axes to loop through them
        im = ax.imshow(true1, cmap=cmap)
        ax.set_title(f'Time step {i+1}')
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)

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
batch_size = 5
MSE_output, JAC_output = [], []
ssim_value = 0.
num_vec = 10
start_idx = 100
end_idx = 200
# ssim = StructuralSimilarityIndexMeasure()


num_col = 0
kernel_size = 11  # Example kernel size (should be odd)
sigma = 20.0  # Standard deviation of the Gaussian
learning_rate = 100.0 # [0.5, 1.0, 5.0, 20.0, 50.0, 100.0]
num_epoch = 100

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
JAC_path = f"../test_result/best_model_FNO_GCS_vec_{num_vec}_JAC.pth"
# "test_result/best_model_FNO_GCS_full epoch_JAC_vec_5.pth"
JAC_model.load_state_dict(torch.load(JAC_path))
JAC_model.eval()

# MSE_path = f"../test_result/best_model_FNO_GCS_MSE.pth"
MSE_path = f"../test_result/best_model_FNO_GCS_vec_1_MSE.pth"
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

    with h5py.File(f'../FNO-NF.jl/scripts/num_obs_20/states_sample_{s_idx}_nobs_20.jld2', 'r') as f1:

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

plot_multiple_abs(set_y[0], f'GCS_partial/vec={num_vec}/S_masked3', cmap='Blues')
set_y = set_y.reshape(-1, batch_size, 8, 64, 64)
num_batch = set_x.shape[0]
print("num_batch", num_batch, "batch size:", batch_size)

# Function for least squares posterior estimation, where the input_data is updated
def least_squares_posterior_estimation(model, input_data, true_data, model_type, learning_rate, num_iterations=500):
    model.eval()  # Ensure the model is in evaluation mode (weights won't be updated)
    mse_loss = torch.nn.MSELoss()  # MSE loss function

    # Set the input data as a tensor that requires gradients (so it can be optimized)
    input_data = input_data.clone().detach().requires_grad_(True).to(device)
    plot_single_abs(input_data.clone().detach().cpu()[0, 0], f'GCS_partial/vec={num_vec}/True_K', cmap='Blues')
    posterior_set = []

    # Define an optimizer to update the input data (instead of the model parameters)
    optimizer = torch.optim.Adam([input_data], lr=learning_rate)
    losses = []
    min_mse, max_mse = [], []
    min_jac, max_jac = [], []

    for iteration in range(num_iterations):
        optimizer.zero_grad()  # Reset gradients
        output = model(input_data.unsqueeze(dim=1))
        # print(output.shape, true_data.shape) torch.Size([100, 1, 8, 64, 64]) torch.Size([100, 8, 64, 64])
        # output = torch.clamp(output, min=0, max=0.9)
        
        plot_single_abs(input_data.clone().detach().cpu()[0, 0], f'GCS_partial/vec={num_vec}/iter_{model_type}_{iteration}', cmap='Blues')
        # mask is well operator here
        output = output #* mask[:100].cuda().float()
        # loss = mse_loss(output[:, :, 15:-15], true_data[:, :, 15:-15])
        print("loss type:", model_type, torch.min(output), torch.max(output))
        for batch in [0, 4]:
            plot_multiple_abs(output[batch][0].detach().cpu().numpy(), f"GCS_partial/vec={num_vec}/lr={learning_rate}/S/{model_type}_{batch}_{iteration}")
            plot_multiple_abs(true_data[batch].detach().cpu().numpy(), f"GCS_partial/vec={num_vec}/lr={learning_rate}/S/True_{batch}")
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
        # input_data.data = torch.clamp(input_data.data, min=10, max=130)

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
        plt.savefig(f'GCS_partial/vec={num_vec}/loss_plot_{learning_rate}_{num_epoch}_{model_type}_{num_col}.png')

        # if iteration % 50 == 0:
        print(f"Iteration {iteration}, Loss: {loss.item()}")
        posterior_set.append(input_data.clone().detach().cpu())

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

# MLE
for i in range(num_batch):
    print(f"Batch {i}: Performing least squares posterior estimation")
    
    X = set_x[i].to( device).float() # /10 # Input permeability [5, 8, 64, 64]
    Y_true = set_y[i].to(device).float() # / 10 # True observation S [5, 8, 64, 64]

    ####### Call MLE for batch_size number of test samples #######
    posterior_estimate_mse, mse_losses, min_mse, min_jac, max_mse, max_jac = least_squares_posterior_estimation(MSE_model, zero_X, Y_true, "MSE", learning_rate, num_iterations=num_epoch)
    posterior_estimate_jac, jac_losses, min_mse, min_jac, max_mse, max_jac = least_squares_posterior_estimation(JAC_model, zero_X, Y_true, "JAC", learning_rate, num_iterations=num_epoch)
    print("posterior estimate", len(posterior_estimate_jac), len(posterior_estimate_jac[0]))
    # print(posterior_estimate_jac[0][0][0]) # epoch x batch_size x 8 x 64 x 64
    # print(posterior_estimate_jac[1][0][0])

    # Plot loss
    plt.figure(figsize=(10, 6))
    plt.plot(mse_losses, label='MSE Model', color='red', marker='^')
    plt.plot(jac_losses, label='PBI Model', color='blue', marker='o')
    plt.xlabel('Iterations', fontsize=15)
    plt.ylabel('Loss', fontsize=15)
    plt.legend(fontsize=15)
    plt.grid(True)
    plt.savefig(f'GCS_partial/vec={num_vec}/loss_plot_{learning_rate}_{num_epoch}_{num_col}.png')

    # Save or visualize posterior results as needed
    print(f"Posterior estimate for batch {i} completed.")
    ssim_all_mse = 0.
    ssim_all_jac = 0.
    # Plot every 50th epoch
    for t in range(0, num_epoch, 50):
        print("----- Num epoch ------", t)
        for b in range(batch_size): # for every test sample
            print("sample index", b)
            print(len(posterior_estimate_mse[t][b]), X.shape)
            abs_diff_mse = abs(posterior_estimate_mse[t][b][-1].detach().cpu() - X[b][-1].detach().cpu())
            abs_diff_jac = abs(posterior_estimate_jac[t][b][-1].detach().cpu() - X[b][-1].detach().cpu())

            # Create a mask: disregard null space
            mask_S = (Y_true[b][-1] != 0).int()
            masked_X = X[b][-1].detach().cpu()
            # masked_X[mask_S == 0] = 20
            # if b in [0, 5, 10, 20, 30, 40] and t == 0:
            #     plot_single_abs(X[b][-1].detach().cpu(), f"masked_before_{b}")
            #     plot_single_abs(masked_X.detach().cpu(), f"masked_after_{b}")
            # masked_posterior_estimate_mse = posterior_estimate_mse[t][b][-1]
            # masked_posterior_estimate_mse[mask_S == 0] = 20
            # masked_posterior_estimate_jac = posterior_estimate_jac[t][b][-1]
            # masked_posterior_estimate_jac[mask_S == 0] = 20

            # ssim_mse = ssim(preds=masked_posterior_estimate_mse.detach().cpu().unsqueeze(dim=0).unsqueeze(0), target=masked_X.detach().cpu().unsqueeze(dim=0).unsqueeze(dim=0))
            # ssim_jac = ssim(preds=masked_posterior_estimate_jac.detach().cpu().unsqueeze(dim=0).unsqueeze(0), target=masked_X.detach().cpu().unsqueeze(dim=0).unsqueeze(dim=0))
            pe_mse = posterior_estimate_mse[t][b][-1].detach().cpu().numpy()
            pe_jac = posterior_estimate_jac[t][b][-1].detach().cpu().numpy()
            ssim_mse = ssim(pe_mse, masked_X.numpy(), data_range = pe_mse.max()-pe_mse.min())
            ssim_jac = ssim(pe_jac, masked_X.numpy(), data_range = pe_jac.max()-pe_jac.min())
            path = f'GCS_partial/vec={num_vec}/lr={learning_rate}/posterior_{num_col}_{t}_{i}_{b}'
            print("shape", X[b][-1].shape, zero_X[b][-1].shape, posterior_estimate_mse[t][b][-1].shape)
            
            plot_diff_with_shared_colorbar_all([X[b][-1].detach().cpu(), zero_X[b][-1].detach().cpu(), posterior_estimate_mse[t][b][-1].detach().cpu(), posterior_estimate_jac[t][b][-1].detach().cpu(), abs_diff_mse, abs_diff_jac], t, ssim_mse, ssim_jac, path, cmap='magma')

    t = num_epoch - 1
    print(t)
    mse_mse = 0
    mse_jac = 0
    mse_loss_func = torch.nn.MSELoss()
    for b in range(batch_size): # for every test sample
        print("sample index", b)
        # print(len(posterior_estimate_mse[t][b]), X.shape)
        abs_diff_mse = abs(posterior_estimate_mse[t][b][-1].detach().cpu() - X[b][-1].detach().cpu())
        abs_diff_jac = abs(posterior_estimate_jac[t][b][-1].detach().cpu() - X[b][-1].detach().cpu())

        # Create a mask: disregard null space
        mask_S = (Y_true[b][-1] != 0).int()
        masked_X = X[b][-1].detach().cpu()
        # masked_X[mask_S == 0] = 20
        # if b in [0, 5, 10, 20, 30, 40] and t == 0:
        #     plot_single_abs(X[b][-1].detach().cpu(), "masked_before_{b}")
        #     plot_single_abs(masked_X.detach().cpu(), "masked_after_{b}")
        # masked_posterior_estimate_mse = posterior_estimate_mse[t][b][-1]
        # masked_posterior_estimate_mse[mask_S == 0] = 20
        # masked_posterior_estimate_jac = posterior_estimate_jac[t][b][-1]
        # masked_posterior_estimate_jac[mask_S == 0] = 20

        # ssim_mse = ssim(preds=masked_posterior_estimate_mse.detach().cpu().unsqueeze(dim=0).unsqueeze(0), target=masked_X.detach().cpu().unsqueeze(dim=0).unsqueeze(dim=0))
        # ssim_jac = ssim(preds=masked_posterior_estimate_jac.detach().cpu().unsqueeze(dim=0).unsqueeze(0), target=masked_X.detach().cpu().unsqueeze(dim=0).unsqueeze(dim=0))
        pe_mse = posterior_estimate_mse[t][b][-1].detach().cpu()
        pe_jac = posterior_estimate_jac[t][b][-1].detach().cpu()

        true_x = X[b][-1].detach().cpu()
        # print("shape", pe_mse.shape, true_x.shape)
        mse_mse += F.mse_loss(pe_mse, true_x) # Compute MSE
        mse_jac += F.mse_loss(pe_jac, true_x)
        ssim_mse = ssim(pe_mse.numpy(), masked_X.numpy(), data_range = pe_mse.numpy().max()-pe_mse.numpy().min())
        ssim_jac = ssim(pe_jac.numpy(), masked_X.numpy(), data_range = pe_jac.numpy().max()-pe_jac.numpy().min())
        ssim_all_mse += ssim_mse
        ssim_all_jac += ssim_jac
        path = f'GCS_partial/vec={num_vec}/lr={learning_rate}/posterior_{num_col}_{t}_{i}_{b}'
        # print("shape", X[b][-1].shape, zero_X[b][-1].shape, posterior_estimate_mse[t][b][-1].shape)
        
        plot_diff_with_shared_colorbar_all([X[b][-1].detach().cpu(), zero_X[b][-1].detach().cpu(), posterior_estimate_mse[t][b][-1].detach().cpu(), posterior_estimate_jac[t][b][-1].detach().cpu(), abs_diff_mse, abs_diff_jac], t, ssim_mse, ssim_jac, path, cmap='magma')

    print("PBI SSIM Full:", ssim_all_jac, "PBI forward losses", jac_losses[-1], "posterior MSE:", mse_jac)
    print("MSE SSIM Full:", ssim_all_mse, "MSE forward losses:", mse_losses[-1], "posterior MSE:", mse_mse)

    # plot min max
    plt.figure(figsize=(10, 6))
    plt.plot(min_mse, label=f'MSE Model', color='red', marker='^')
    plt.plot(min_jac, label='PBI Model', color='blue', marker='o')
    plt.xlabel('Iterations', fontsize=15)
    plt.ylabel('Min', fontsize=15)
    plt.legend(fontsize=15)
    plt.grid(True)
    plt.savefig(f'GCS_partial/recent/lr={learning_rate}/min_{num_epoch}.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(max_mse, label=f'MSE Model', color='red', marker='^')
    plt.plot(max_jac, label='PBI Model', color='blue', marker='o')
    plt.xlabel('Iterations', fontsize=15)
    plt.ylabel('Max', fontsize=15)
    plt.legend(fontsize=15)
    plt.grid(True)
    plt.savefig(f'GCS_partial/recent/lr={learning_rate}/max_{num_epoch}.png')
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