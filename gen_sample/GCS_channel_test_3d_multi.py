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
loss_type = "JAC"
set_x, set_y = [], []
batch_size = 10
MSE_output = []
JAC_output = []
MSE_S, JAC_S = [], []
set_vjp, set_eig = [], []
ssim_value, ssim_value_jac = 0., 0.
ssim = StructuralSimilarityIndexMeasure()
num_vec = 10
start_idx = 100
end_idx = 200


# generate dataset with trained FNO

# load MSE FNO
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

MSE_path = f"../test_result/best_model_FNO_GCS_vec_1_MSE.pth"
MSE_model.load_state_dict(torch.load(MSE_path))
MSE_model.eval()

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
# JAC_path = f"../test_result/best_model_FNO_GCS_vec_5_JAC.pth"
JAC_path = f"../test_result/best_model_FNO_GCS_vec_{num_vec}_JAC.pth"
JAC_model.load_state_dict(torch.load(JAC_path))
JAC_model.eval()

# load input data K
with h5py.File('../FNO-NF.jl/data/training-data/cons=1e-5_delta=25_num_sample=10000_theta0=5.jld2', 'r') as f:
    # List all the datasets in the file
    print("Keys: %s" % f.keys())
    # Length of K is 10000. Load K
    K = f['perm'][:]
    print(len(K))
    set_x.append(K) # 1, 10000, 64, 64
    set_x = set_x[0]


# Read the each file s_idx: sample index
for s_idx in range(start_idx, end_idx):

    with h5py.File(f'../FNO-NF.jl/scripts/num_obs_20/states_sample_{s_idx}_nobs_20.jld2', 'r') as f1, \
        h5py.File(f'../FNO-NF.jl/scripts/num_obs_20/FIM_eigvec_sample_{s_idx}_nobs_20.jld2', 'r') as f2, \
        h5py.File(f'../FNO-NF.jl/scripts/num_obs_20/FIM_vjp_sample_{s_idx}_nobs_20.jld2', 'r') as f3:

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
        
        eigvec = f2['single_stored_object'][:] # len: 8 x 20 x 64 x 64
        vjp = f3['single_stored_object'][:] # len: 8 x 20 x 4096
        # print(torch.tensor(eigvec).shape, torch.tensor(vjp).shape)

        # set_y.append(S) 
        set_y.append(torch.stack(states_tensors).reshape(8, 64, 64))
        set_vjp.append(torch.tensor(vjp).reshape(8, 20, 64, 64)[:, :num_vec]) 
        set_eig.append(torch.tensor(eigvec).reshape(8, 20, 64, 64)[:, :num_vec])

set_y = torch.stack(set_y)
org_set_y = set_y
train_vjp = torch.stack(set_vjp).reshape(-1, 64, 64)
train_vjp = train_vjp / torch.norm(train_vjp)
# train_vjp = train_vjp / (10**13)

all_ks = torch.tensor(set_x) # tensor size [10000, 64, 64]
all_ks = all_ks[start_idx-1:end_idx-1] # tensor size [2000, 64, 64]
all_ks = all_ks.unsqueeze(1)  # Now tensor is [2000, 1, 64, 64]
all_ks = all_ks.repeat(1, 8, 1, 1)  # Now tensor is [2000, 8, 64, 64]
all_ks = all_ks.reshape(-1, batch_size, 8, 64, 64) # Now tensor is [20, 100, 8, 64, 64]
print(all_ks.shape)
plot_multiple_abs(all_ks[0, 0], 'GCS_sample/K0', cmap='Blues')
plot_multiple_abs(all_ks[0, 1], 'GCS_sample/K1', cmap='Blues')
plot_multiple_abs(all_ks[-1, -3], 'GCS_sample/K1998', cmap='Blues')
plot_multiple_abs(all_ks[-1, -2], 'GCS_sample/K1999', cmap='Blues')
plot_multiple_abs(all_ks[-1, -1], 'GCS_sample/K2000', cmap='Blues')

set_eig = torch.stack(set_eig)

True_j = train_vjp.float()
print("True J Before", True_j.shape) #True J Before torch.Size([700, 8, 64, 64])
True_j = True_j.reshape(-1, batch_size, 8, num_vec, 64, 64)
print("After True J", True_j.shape) #([7, 100, 8, 64, 64]) -> ([idx, batchsize, 8, 64, 64])
vec = torch.tensor(set_eig)
print("vec", vec.shape)
vec_batch = vec.reshape(-1, batch_size, 8, num_vec, 64, 64)
print("vec", vec_batch.shape)
vec_batch = vec_batch.cuda().float()


set_y = set_y.reshape(-1, batch_size, 8, 64, 64)

num_batch = all_ks.shape[0]
print("num batch", num_batch)

# generate output S with MSE
MSE_model.eval()
with torch.no_grad():
    for i in range(num_batch):
        print("MSE", i)
        X = all_ks[i].cuda().float().unsqueeze(1)
        output = MSE_model(X)
        MSE_output.append(output.squeeze().detach().cpu().numpy())
        print("len", len(MSE_output))
        if i == 0:
            plot_multiple_abs(MSE_output[0][0], 'GCS_sample/MSE0', cmap='Blues')
            plot_multiple_abs(MSE_output[0][1], 'GCS_sample/MSE1', cmap='Blues')
            MSE_abs_diff_train_1 = abs(set_y[0, 0] - MSE_output[0][0])
            MSE_abs_diff_train_2 = abs(set_y[0, 1] - MSE_output[0][1])
        if i == num_batch-1:
            plot_multiple_abs(MSE_output[num_batch-1][-2], 'GCS_sample/MSE1999', cmap='Blues')
            plot_multiple_abs(MSE_output[num_batch-1][-1], 'GCS_sample/MSE2000', cmap='Blues')
            plot_multiple_abs(MSE_output[num_batch-1][-3], 'GCS_sample/MSE1998', cmap='Blues')
            MSE_abs_diff_test_1 = abs(set_y[num_batch-1, -2] - MSE_output[num_batch-1][-2])
            MSE_abs_diff_test_2 = abs(set_y[num_batch-1, -1] - MSE_output[num_batch-1][-1])
            MSE_abs_diff_test_3 = abs(set_y[num_batch-1, -3] - MSE_output[num_batch-1][-3])
        # compute ssim for test samples
        if i > 17:
            # for b in range(batch_size):
            ssim_value += ssim(torch.tensor(MSE_output[i]).unsqueeze(dim=0), set_y[i].unsqueeze(dim=0))
            print("batch index: ", i, ssim_value)
        

# generate output S with JAC
JAC_model.eval()
with torch.no_grad():
    for j in range(num_batch):
        print("JAC", j)
        X_jac = all_ks[j].cuda().float().unsqueeze(1)
        target_vjp = True_j[j].cuda()
        cur_vec_batch = vec_batch[j] # [10, 8, 10, 64, 64]
        print(cur_vec_batch.shape)
        output_jac = JAC_model(X_jac)
        JAC_output.append(output_jac.squeeze().detach().cpu().numpy())
        output, vjp_func = torch.func.vjp(JAC_model, X_jac)
        # vjp_out_list = []
        # for e in range(num_vec):
        #     print("e", e, cur_vec_batch.shape, cur_vec_batch[:, :, e].unsqueeze(dim=1).shape)
        #     vjp_out_onevec = vjp_func(cur_vec_batch[:, :, e].unsqueeze(dim=1))[0] # -> learned vjp #[10, 1, 8, 64, 64]
        #     # vjp_out[:, :, e] = vjp_out_onevec
        #     vjp_out_list.append(vjp_out_onevec)
        #     vjp_out = torch.stack(vjp_out_list, dim=2)

        # vjp_out = torch.func.vmap(vjp_func, in_dims=(2,))(cur_vec_batch)
        vjp_out = torch.func.vmap(lambda vec_batch: vjp_func(vec_batch.unsqueeze(1))[0], in_dims=2)(cur_vec_batch)
 
        print("vjp out", vjp_out.shape) #torch.Size([10, 10, 1, 8, 64, 64])
        # plot_multiple_abs(vjp_out[0].squeeze().detach().cpu(), 'GCS_sample/debug_vjp', cmap='viridis')    
        if j == 0:
            plot_multiple_abs(JAC_output[0][0], 'GCS_sample/JAC0', cmap='Blues')
            plot_multiple_abs(JAC_output[0][1], 'GCS_sample/JAC1', cmap='Blues')
            JAC_abs_diff_train_1 = abs(set_y[0, 0] - JAC_output[0][0])
            JAC_abs_diff_train_2 = abs(set_y[0, 1] - JAC_output[0][1])
        if j == num_batch-1:
            plot_multiple_abs(JAC_output[num_batch-1][-2], 'GCS_sample/JAC1999', cmap='Blues')
            plot_multiple_abs(JAC_output[num_batch-1][-1], 'GCS_sample/JAC2000', cmap='Blues')
            plot_multiple_abs(JAC_output[num_batch-1][-3], 'GCS_sample/JAC1998', cmap='Blues')
            JAC_abs_diff_test_1 = abs(set_y[num_batch-1, -2] - JAC_output[num_batch-1][-2])
            JAC_abs_diff_test_2 = abs(set_y[num_batch-1, -1] - JAC_output[num_batch-1][-1])
            JAC_abs_diff_test_3 = abs(set_y[num_batch-1, -3] - JAC_output[num_batch-1][-3])
            
        # compute ssim for test samples
        if j > 17:
            # for b_jac in range(batch_size):
            ssim_value_jac += ssim(torch.tensor(JAC_output[j]).unsqueeze(dim=0), set_y[j].unsqueeze(dim=0))
            print("batch index: ", j, ssim_value_jac)


# plot difference
plot_share_bar(torch.stack([MSE_abs_diff_train_1, JAC_abs_diff_train_1]).reshape(-1, 64, 64), f'GCS_sample/forward_pred_train_diff1_{num_vec}', cmap='magma')
plot_share_bar(torch.stack([MSE_abs_diff_train_2, JAC_abs_diff_train_2]).reshape(-1, 64, 64), f'GCS_sample/forward_pred_train_diff2_{num_vec}', cmap='magma')
plot_share_bar(torch.stack([MSE_abs_diff_test_1, JAC_abs_diff_test_1]).reshape(-1, 64, 64), f'GCS_sample/forward_pred_test_diff1_{num_vec}', cmap='magma')
plot_share_bar(torch.stack([MSE_abs_diff_test_2, JAC_abs_diff_test_2]).reshape(-1, 64, 64), f'GCS_sample/forward_pred_test_diff2_{num_vec}', cmap='magma')
plot_share_bar(torch.stack([MSE_abs_diff_test_3, JAC_abs_diff_test_3]).reshape(-1, 64, 64), f'GCS_sample/forward_pred_test_diff3_{num_vec}', cmap='magma')

# '''
# forward performance for ood sample
# '''
# num_ood=1
# with h5py.File(f'../FNO-NF.jl/scripts/K_ood{num_ood}.jld2', 'r') as f:
#     # List all the datasets in the file
#     print("Keys: %s" % f.keys())
#     # Accessing a specific dataset
#     K = f['single_stored_object'][:]
#     print(len(K))

# with h5py.File(f'../FNO-NF.jl/scripts/conc{num_ood}.jld2', 'r') as c:
#     # List all the datasets in the file
#     print("Keys: %s" % c.keys())
#     # Accessing a specific dataset
#     S = c['single_stored_object'][:][0]
#     print(len(S))

# K_ood = torch.tensor(K).cuda().float()
# K_ood = K_ood.unsqueeze(0).unsqueeze(1)  # Now tensor is [1, 1, 64, 64]
# K_ood = K_ood.repeat(1, 8, 1, 1)  # Now tensor is [1, 8, 64, 64]
# print("JAC")
# ood_jac = JAC_model(K_ood).detach().cpu().squeeze()
# print("MSE")
# ood_mse = MSE_model(K_ood).detach().cpu().squeeze()
# plot_multiple(ood_jac, f'GCS_sample/ood_jac_{num_ood}', cmap='Blues')
# plot_multiple(ood_mse, f'GCS_sample/ood_mse_{num_ood}', cmap='Blues')
# plot_share_bar(torch.stack([ood_mse, ood_jac]).reshape(-1, 64, 64), f'GCS_sample/ood_diff_{num_ood}', cmap='magma')

# jac_ood_ssim = ssim(torch.tensor(ood_jac).unsqueeze(dim=0), torch.tensor(S).unsqueeze(dim=0))
# mse_ood_ssim = ssim(torch.tensor(ood_mse).unsqueeze(dim=0), torch.tensor(S).unsqueeze(dim=0))

'''
SSIM
'''
print("MSE", ssim_value/2, "JAC", ssim_value_jac/2)
# print("MSE ood", mse_ood_ssim, "JAC ood", jac_ood_ssim)

# save both K and S 
JAC_csv_path = 'GCS_jac.npz'
MSE_csv_path = 'GCS_mse.npz'
input_csv_path = 'GCS_K.npz'

np.savez(MSE_csv_path, generated_S=np.array(MSE_output))
print(f"Saved dataset to {MSE_csv_path}")   

np.savez(JAC_csv_path, generated_S=np.array(JAC_output))
print(f"Saved dataset to {JAC_csv_path}")

np.savez(input_csv_path, input_K=np.array(set_x))
print(f"Saved dataset to {input_csv_path}")
