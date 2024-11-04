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
# ssim = StructuralSimilarityIndexMeasure()


num_col = 0
kernel_size = 11  # Example kernel size (should be odd)
sigma = 20.0  # Standard deviation of the Gaussian
learning_rate = 100.0 # [0.5, 1.0, 5.0, 20.0, 50.0, 100.0]
num_epoch = 200

# Load MSE FNO
MSE_model = FNO(
    in_channels=2,
    out_channels=1,
    decoder_layer_size=128,
    num_fno_layers=6,
    num_fno_modes=[33, 33],
    padding=3,
    dimension=2,
    latent_channels=64
).to(device)

# load JAC FNO
JAC_model = FNO(
    in_channels=2,
    out_channels=1,
    decoder_layer_size=128,
    num_fno_layers=6,
    num_fno_modes=[33, 33],
    padding=3,
    dimension=2,
    latent_channels=64
).to(device)
JAC_path = f"../test_result/best_model_FNO_GCS_onestep_JAC.pth"
JAC_model.load_state_dict(torch.load(JAC_path))
JAC_model.eval()

MSE_path = f"../test_result/best_model_FNO_GCS_onestep_MSE.pth"
MSE_model.load_state_dict(torch.load(MSE_path))
MSE_model.eval()

# # Load input data K
# with h5py.File('../FNO-NF.jl/data/training-data/cons=1e-5_delta=25_num_sample=10000_theta0=5.jld2', 'r') as f:
#     print("Keys: %s" % f.keys())  # List all the datasets in the file
#     K = f['perm'][:]
#     set_x.append(K)

# set_x = set_x[0]
# set_x = torch.tensor(set_x)
# set_x = set_x[:1000]  # Reduce the dataset size
# set_x = set_x[900:] # only test data
# set_x = set_x.unsqueeze(1).repeat(1, 8, 1, 1)  # Reshape [2000, 8, 64, 64]
# set_x = set_x.reshape(-1, batch_size, 8, 64, 64)  # Reshape to batches
# if loss_type == "JAC":
#     FNO_type = JAC_model
# else:
#     FNO_type = MSE_model


# # Read the each file s_idx: sample index
# for s_idx in range(1, 1001):

#     with h5py.File(f'../FNO-NF.jl/scripts/num_obs_2/states_sample_{s_idx}_nobs_2.jld2', 'r') as f1:

#         # print("f1 Keys: %s" % f1.keys()) #<KeysViewHDF5 ['single_stored_object']>
#         # S = f1['single_stored_object'][:] # len: 8 x 64
#         # Assuming 'states' is the key where the states are stored
#         states_refs = f1['single_stored_object'][:]  # Load the array of object references
#         states_tensors = []
#         # Loop over the references, dereference them, and convert to tensors
#         for ref in states_refs:
#             # Dereference the object reference
#             state_data = f1[ref][:]
            
#             # Convert the dereferenced data to a PyTorch tensor
#             state_tensor = torch.tensor(state_data)
#             states_tensors.append(state_tensor)

#         # set_y.append(S) 
#         set_y.append(torch.stack(states_tensors).reshape(8, 64, 64))

# set_y = torch.stack(set_y[900:])
# org_set_y = set_y
# plot_multiple_abs(org_set_y[0], f'GCS_partial_onestep/S_org', cmap='Blues')
org_x = []
with h5py.File('../FNO-NF.jl/data/training-data/cons=1e-5_delta=25_num_sample=10000_theta0=5.jld2', 'r') as f:
        # List all the datasets in the file
        print("Keys: %s" % f.keys())
        # Length of K is 10000. Load K
        K = f['perm'][:]
        print(len(K))
        org_x.append(K) # 1, 10000, 64, 64
        org_x = org_x[0]

# Read the each file s_idx: sample index
for s_idx in range(1, 1001):

    with h5py.File(f'../FNO-NF.jl/scripts/num_obs_2/states_sample_{s_idx}_nobs_2.jld2', 'r') as f1, \
        h5py.File(f'../FNO-NF.jl/scripts/num_obs_2/FIM_eigvec_sample_{s_idx}_nobs_2.jld2', 'r') as f2, \
        h5py.File(f'../FNO-NF.jl/scripts/num_obs_2/FIM_vjp_sample_{s_idx}_nobs_2.jld2', 'r') as f3:

        # print("f1 Keys: %s" % f1.keys()) #<KeysViewHDF5 ['single_stored_object']>
        # S = f1['single_stored_object'][:] # len: 8 x 64
        # Assuming 'states' is the key where the states are stored
        states_refs = f1['single_stored_object'][:]  # Load the array of object references
        states_tensors = []
        # Loop over the references, dereference them, and convert to tensors
        for ref in states_refs:
            # Dereference the object reference
            state_data = f1[ref][:]
            
            # Convert the dereferenced data to a PyTorch tensor
            state_tensor = torch.tensor(state_data)
            states_tensors.append(state_tensor)
        eigvec = f2['single_stored_object'][:] # len: 3 x 64 x 64 ..?
        vjp = f3['single_stored_object'][:] # len: 4096 -> 64 x 64

        # Create input ([K, S_t]_{t=1, ... ,5})
        states_tensors = torch.stack(states_tensors).reshape(8, 64, 64)
        set_x.append([org_x[s_idx], states_tensors[0]])
        set_x.append([org_x[s_idx], states_tensors[1]])
        set_x.append([org_x[s_idx], states_tensors[2]])
        set_x.append([org_x[s_idx], states_tensors[3]])
        set_x.append([org_x[s_idx], states_tensors[4]])

        # Create Output (S_{t+1})
        set_y.append([states_tensors[1].numpy()])
        set_y.append([states_tensors[2].numpy()])
        set_y.append([states_tensors[3].numpy()])
        set_y.append([states_tensors[4].numpy()])
        set_y.append([states_tensors[5].numpy()])

set_x = torch.tensor(set_x[800:900])
set_y = torch.tensor(set_y[800:900])
print("set x", set_x.shape)
set_x = set_x.reshape(-1, batch_size, 2, 64, 64)
set_y = set_y.reshape(-1, batch_size, 1, 64, 64)

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

plot_multiple_abs(set_y[0].squeeze(), f'GCS_partial_onestep/S_masked3', cmap='Blues')
# set_y = set_y.reshape(-1, batch_size, 8, 64, 64)
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
    min_mse, max_mse = [], []
    min_jac, max_jac = [], []

    for iteration in range(num_iterations):
        optimizer.zero_grad()  # Reset gradients
        print("input data", input_data.shape)
        output = model(input_data)
        # output = torch.clamp(output, min=0, max=0.9)
        
        plot_single_abs(input_data.detach().cpu()[0, 0], f'GCS_partial_onestep/iter_{model_type}_{iteration}', cmap='Blues')
        # mask is well operator here
        output = output #* mask[:100].cuda().float()
        # loss = mse_loss(output[:, :, 15:-15], true_data[:, :, 15:-15])
        print("loss type:", model_type, torch.min(output), torch.max(output))
        for batch in [0, 10, 20, 30, 40, 50]:
            plot_single(output[batch][-1].detach().cpu().numpy(), f"GCS_partial_onestep/recent/lr={learning_rate}/S/{model_type}_{batch}_{iteration}")
            plot_single(true_data[batch][-1].detach().cpu().numpy(), f"GCS_partial_onestep/recent/lr={learning_rate}/S/True_{batch}")
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
        plt.savefig(f'GCS_partial_onestep/{model_type}_{num_col}/loss_plot_{learning_rate}_{num_epoch}.png')

        # if iteration % 50 == 0:
        print(f"Iteration {iteration}, Loss: {loss.item()}")
        posterior_set.append(input_data.detach())

        


    # Return the optimized input data (permeability K)
    return posterior_set, losses, min_mse, min_jac, max_mse, max_jac  # Detach from the computational graph



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
    # H(K0)
    zero_X = torch.mean(X[:, 0], dim=0).unsqueeze(dim=0) #[64, 64]
    print("first", zero_X.shape)
    # zero_X = zero_X.unsqueeze(1).repeat(1, 8, 1, 1)  # Reshape [100, 8, 64, 64]
    # print(zero_X.shape)
    for items in range(X.shape[0]):
        X[items, 0] = zero_X
    # zero_X = zero_X.unsqueeze(1).repeat(100, 1, 1, 1)
    # print("zero X", zero_X.shape)
    #######
    posterior_estimate_mse, mse_losses, min_mse, min_jac, max_mse, max_jac = least_squares_posterior_estimation(MSE_model, X, Y_true, "MSE", learning_rate, num_iterations=num_epoch)
    posterior_estimate_jac, jac_losses, min_mse, min_jac, max_mse, max_jac = least_squares_posterior_estimation(JAC_model, X, Y_true, "JAC", learning_rate, num_iterations=num_epoch)

    # Plot loss
    plt.figure(figsize=(10, 6))
    plt.plot(mse_losses, label='MSE Model', color='red', marker='^')
    plt.plot(jac_losses, label='PBI Model', color='blue', marker='o')
    plt.xlabel('Iterations', fontsize=15)
    plt.ylabel('Loss', fontsize=15)
    plt.legend(fontsize=15)
    plt.grid(True)
    plt.savefig(f'GCS_partial_onestep/both_{num_col}/loss_plot_{learning_rate}_{num_epoch}.png')

    # Save or visualize posterior results as needed
    print(f"Posterior estimate for batch {i} completed.")
    ssim_all_mse = 0.
    ssim_all_jac = 0.
    # Plot every 50th epoch
    for t in range(0, num_epoch, 50):
        print(t)
        for b in range(batch_size): # for every test sample
            print("b", b)
            print(len(posterior_estimate_mse[t][b]), X.shape)
            abs_diff_mse = abs(posterior_estimate_mse[t][b][-1].detach().cpu() - X[b][-1].detach().cpu())
            abs_diff_jac = abs(posterior_estimate_jac[t][b][-1].detach().cpu() - X[b][-1].detach().cpu())

            # Create a mask: disregard null space
            mask_S = (Y_true[b][-1] != 0).int()
            masked_X = X[b][-1].detach().cpu()
            masked_X[mask_S == 0] = 20
            if b in [0, 5, 10, 20, 30, 40] and t == 0:
                plot_single_abs(X[b][-1].detach().cpu(), "masked_before_{b}")
                plot_single_abs(masked_X.detach().cpu(), "masked_after_{b}")
            masked_posterior_estimate_mse = posterior_estimate_mse[t][b][-1]
            print("shape", masked_posterior_estimate_mse.detach().cpu().shape)
            masked_posterior_estimate_mse[mask_S == 0] = 20
            masked_posterior_estimate_jac = posterior_estimate_jac[t][b][-1]
            masked_posterior_estimate_jac[mask_S == 0] = 20

            # ssim_mse = ssim(preds=masked_posterior_estimate_mse.detach().cpu().unsqueeze(dim=0).unsqueeze(0), target=masked_X.detach().cpu().unsqueeze(dim=0).unsqueeze(dim=0))
            # ssim_jac = ssim(preds=masked_posterior_estimate_jac.detach().cpu().unsqueeze(dim=0).unsqueeze(0), target=masked_X.detach().cpu().unsqueeze(dim=0).unsqueeze(dim=0))
            pe_mse = posterior_estimate_mse[t][b][-1].detach().cpu().numpy()
            pe_jac = posterior_estimate_jac[t][b][-1].detach().cpu().numpy()
            ssim_mse = ssim(pe_mse, masked_X.numpy(), data_range = pe_mse.max()-pe_mse.min())
            ssim_jac = ssim(pe_jac, masked_X.numpy(), data_range = pe_jac.max()-pe_jac.min())
            path = f'GCS_partial_onestep/recent/lr={learning_rate}/both_{num_col}/posterior_{t}_{i}_{b}'
            print("shape", X[b][-1].shape, zero_X[b].shape, posterior_estimate_mse[t][b][-1].shape)
            
            plot_diff_with_shared_colorbar_all([X[b][-1].detach().cpu(), zero_X[b].detach().cpu(), posterior_estimate_mse[t][b][-1].detach().cpu(), posterior_estimate_jac[t][b][-1].detach().cpu(), abs_diff_mse, abs_diff_jac], t, ssim_mse, ssim_jac, path, cmap='magma')

    t = num_epoch - 1
    print(t)
    mse_mse = 0
    mse_jac = 0
    mse_loss_func = torch.nn.MSELoss()
    for b in range(batch_size): # for every test sample
        print("b", b)
        print(len(posterior_estimate_mse[t][b]), X.shape)
        abs_diff_mse = abs(posterior_estimate_mse[t][b][-1].detach().cpu() - X[b][-1].detach().cpu())
        abs_diff_jac = abs(posterior_estimate_jac[t][b][-1].detach().cpu() - X[b][-1].detach().cpu())

        # Create a mask: disregard null space
        mask_S = (Y_true[b][-1] != 0).int()
        masked_X = X[b][-1].detach().cpu()
        masked_X[mask_S == 0] = 20
        if b in [0, 5, 10, 20, 30, 40] and t == 0:
            plot_single_abs(X[b][-1].detach().cpu(), "masked_before_{b}")
            plot_single_abs(masked_X.detach().cpu(), "masked_after_{b}")
        masked_posterior_estimate_mse = posterior_estimate_mse[t][b][-1]
        print("shape", masked_posterior_estimate_mse.detach().cpu().shape)
        masked_posterior_estimate_mse[mask_S == 0] = 20
        masked_posterior_estimate_jac = posterior_estimate_jac[t][b][-1]
        masked_posterior_estimate_jac[mask_S == 0] = 20

        # ssim_mse = ssim(preds=masked_posterior_estimate_mse.detach().cpu().unsqueeze(dim=0).unsqueeze(0), target=masked_X.detach().cpu().unsqueeze(dim=0).unsqueeze(dim=0))
        # ssim_jac = ssim(preds=masked_posterior_estimate_jac.detach().cpu().unsqueeze(dim=0).unsqueeze(0), target=masked_X.detach().cpu().unsqueeze(dim=0).unsqueeze(dim=0))
        pe_mse = posterior_estimate_mse[t][b][-1].detach().cpu().numpy()
        pe_jac = posterior_estimate_jac[t][b][-1].detach().cpu().numpy()

        true_x = X[b][-1].detach().cpu().numpy()
        # @TODO
        # mse_mse += mse_loss_func(pe_mse, true_x)
        # mse_jac += mse_loss_func(pe_jac, true_x)
        ssim_mse = ssim(pe_mse, masked_X.numpy(), data_range = pe_mse.max()-pe_mse.min())
        ssim_jac = ssim(pe_jac, masked_X.numpy(), data_range = pe_jac.max()-pe_jac.min())
        print(pe_mse.shape, true_x.shape)
        ssim_all_mse += ssim_mse
        ssim_all_jac += ssim_jac
        path = f'GCS_partial_onestep/recent/lr={learning_rate}/both_{num_col}/posterior_{t}_{i}_{b}'
        print("shape", X[b][-1].shape, zero_X[b][-1].shape, posterior_estimate_mse[t][b][-1].shape)
        
        plot_diff_with_shared_colorbar_all([X[b][-1].detach().cpu(), zero_X[b][-1].detach().cpu(), posterior_estimate_mse[t][b][-1].detach().cpu(), posterior_estimate_jac[t][b][-1].detach().cpu(), abs_diff_mse, abs_diff_jac], t, ssim_mse, ssim_jac, path, cmap='magma')
    # for b in range(batch_size): # for every test sample
    #     abs_diff_mse = abs(posterior_estimate_mse[t][b][-1].detach().cpu() - X[b][-1].detach().cpu())
    #     abs_diff_jac = abs(posterior_estimate_jac[t][b][-1].detach().cpu() - X[b][-1].detach().cpu())
    #     # ssim_mse = ssim(preds=posterior_estimate_mse[t][b][-1].detach().cpu().unsqueeze(dim=0).unsqueeze(0), target=X[b][-1].detach().cpu().unsqueeze(dim=0).unsqueeze(dim=0))
    #     # ssim_jac = ssim(preds=posterior_estimate_jac[t][b][-1].detach().cpu().unsqueeze(dim=0).unsqueeze(0), target=X[b][-1].detach().cpu().unsqueeze(dim=0).unsqueeze(dim=0))
    #     pe_mse = posterior_estimate_mse[t][b][-1].detach().cpu().numpy()
    #     pe_jac = posterior_estimate_jac[t][b][-1].detach().cpu().numpy()
    #     ssim_mse = ssim(pe_mse, X[b][-1].detach().cpu().numpy(), data_range = pe_mse.max()-pe_mse.min())
    #     ssim_jac = ssim(pe_jac, X[b][-1].detach().cpu().numpy(), data_range = pe_jac.max()-pe_jac.min())
    #     path = f'GCS_partial_onestep/recent/lr={learning_rate}/both_{num_col}/posterior_{t}_{i}_{b}'
    #     ssim_all_mse += ssim_mse
    #     ssim_all_jac += ssim_jac
    #     print("shape", X[b][-1].shape, zero_X[b][-1].shape, posterior_estimate_mse[t][b][-1].shape)
        
    #     plot_diff_with_shared_colorbar_all([X[b][-1].detach().cpu(), zero_X[b][-1].detach().cpu(), posterior_estimate_mse[t][b][-1].detach().cpu(), posterior_estimate_jac[t][b][-1].detach().cpu(), abs_diff_mse, abs_diff_jac], t, ssim_mse, ssim_jac, path, cmap='magma')
    print("PBI SSIM Full:", ssim_all_jac, "PBI losses", jac_losses[-1], mse_jac)
    print("MSE SSIM Full:", ssim_all_mse, "MSE losses:", mse_losses[-1], mse_mse)

    # plot min max
    plt.figure(figsize=(10, 6))
    plt.plot(min_mse, label=f'MSE Model', color='red', marker='^')
    plt.plot(min_jac, label='PBI Model', color='blue', marker='o')
    plt.xlabel('Iterations', fontsize=15)
    plt.ylabel('Min', fontsize=15)
    plt.legend(fontsize=15)
    plt.grid(True)
    plt.savefig(f'GCS_partial_onestep/recent/lr={learning_rate}/min_{num_epoch}.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(max_mse, label=f'MSE Model', color='red', marker='^')
    plt.plot(max_jac, label='PBI Model', color='blue', marker='o')
    plt.xlabel('Iterations', fontsize=15)
    plt.ylabel('Max', fontsize=15)
    plt.legend(fontsize=15)
    plt.grid(True)
    plt.savefig(f'GCS_partial_onestep/recent/lr={learning_rate}/max_{num_epoch}.png')
    plt.close()

# Full 100.
    # PBI SSIM Full: tensor(56.6365) PBI losses 2.1875030142837204e-05
# MSE SSIM Full: tensor(55.6442) MSE losses: 2.9429453206830658e-05

# PBI SSIM Full: tensor(57.5216) PBI losses 1.4258051123761106e-05
# MSE SSIM Full: tensor(59.3005) MSE losses: 2.1109874069225043e-05

# PBI SSIM Full: tensor(56.2269) PBI losses 1.996083665289916e-05
# MSE SSIM Full: tensor(56.5895) MSE losses: 3.419788117753342e-05

# learning rate=100. 200 epoch
# PBI SSIM Full: tensor(82.7853) PBI losses 2.9922994144726545e-05
# MSE SSIM Full: tensor(84.3231) MSE losses: 5.122043512528762e-05

# learning rate=100, 400 epoch
# PBI SSIM Full: tensor(83.0904) PBI losses 1.997925210162066e-05
# MSE SSIM Full: tensor(84.5203) MSE losses: 3.459892104729079e-05

# before clamp
# PBI SSIM Full: tensor(83.1210) PBI losses 2.2026229999028146e-05
# MSE SSIM Full: tensor(84.9684) MSE losses: 3.198909689672291e-05

# after clamp
# PBI SSIM Full: tensor(83.0168) PBI losses 2.0271818357286975e-05
# MSE SSIM Full: tensor(84.8398) MSE losses: 3.0183911803760566e-05

# after clamp both S and K
# PBI SSIM Full: tensor(47.2440) PBI losses 2.0138128093094565e-05
# MSE SSIM Full: tensor(44.3805) MSE losses: 3.0917133699404076e-05

# after clamp both S and K, run for 200
# PBI SSIM Full: tensor(46.9656) PBI losses 2.0138128093094565e-05
# MSE SSIM Full: tensor(44.2491) MSE losses: 3.0917133699404076e-05

## after changing the package pymetric_msssim: 
# PBI SSIM Full: tensor(68.9674) PBI losses 2.0138128093094565e-05
# MSE SSIM Full: tensor(67.7945) MSE losses: 3.0917133699404076e-05

## after changing the package to sklearn: