import torch
import torch.nn as nn
import torch.autograd.functional as F

import numpy as np
import csv
import h5py
from torch.func import vmap, vjp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import seaborn as sGCS
from functorch import vjp, vmap
from modulus.models.fno import FNO
from modulus.models.mlp.fully_connected import FullyConnected


'''
We test stability of trained FNOs with rollout.
'''

def plot_single(true1, path):
    plt.figure(figsize=(10, 10))
    plt.rcParams.update({'font.size': 16})

    plt.imshow(true1, cmap='Blues')
    plt.colorbar(fraction=0.045, pad=0.06)
    # plt.title('True Saturation')

    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return

# plot_ood(true[idx,4], true[idx, 5], one_mse, one_jac, true[idx, 5]-one_mse, true[idx, 5]-one_jac, single_pred_path)
def plot_ood(input, true, mse, jac, mse_diff, jac_diff, path):
    plt.figure(figsize=(20, 10))
    plt.rcParams.update({'font.size': 16})

    vmin, vmax = 0.0, max(abs(mse_diff.max()), abs(jac_diff.max()))

    plt.subplot(2, 4, 1)
    plt.imshow(input.cpu().numpy(), cmap='Blues')
    plt.colorbar(fraction=0.045, pad=0.06)
    plt.title(f'Input: Saturation_{4}')

    plt.subplot(2, 4, 2)
    plt.imshow(true.cpu().numpy(), cmap='Blues')
    plt.colorbar(fraction=0.045, pad=0.06)
    plt.title(f'True: Saturation_{5}')

    plt.subplot(2, 4, 3)
    plt.imshow(mse.cpu().numpy(), cmap='Blues')
    plt.colorbar(fraction=0.045, pad=0.06)
    plt.title(f'MSE: Saturation_{5}')

    plt.subplot(2, 4, 4)
    plt.imshow(jac.cpu().numpy(), cmap='Blues')
    plt.colorbar(fraction=0.045, pad=0.06)
    plt.title(f'GM: Saturation_{5}')

    plt.subplot(2, 4, 7)
    plt.imshow(mse_diff.cpu().numpy(), cmap='inferno', vmin=vmin, vmax=vmax)
    plt.colorbar(fraction=0.045, pad=0.06)
    plt.title(f'MSE Error')

    plt.subplot(2, 4, 8)
    plt.imshow(jac_diff.cpu().numpy(), cmap='inferno', vmin=vmin, vmax=vmax)
    plt.colorbar(fraction=0.045, pad=0.06)
    plt.title(f'GM Error')

    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return

def plot_results(K, sat1, sat2, sat3, sat4, sat5, path, type=None):
    plt.figure(figsize=(30, 5))
    plt.rcParams.update({'font.size': 16})

    vmin, vmax = 0.0, max(abs(sat2.max()), abs(sat3.max()), abs(sat4.max()), abs(sat5.max()))
    if type == "error":
        cmap = 'inferno'
    else:
        cmap = 'Blues'

    if type != "error":
        plt.subplot(1, 6, 1)
        plt.imshow(K.cpu().numpy(), cmap=cmap)
        plt.colorbar(fraction=0.045, pad=0.06)
        plt.title(f'Input: Permeabiltiy')

        plt.subplot(1, 6, 2)
        plt.imshow(sat1.cpu().numpy(), cmap=cmap)
        plt.colorbar(fraction=0.045, pad=0.06)
        plt.title(f'Input: Saturation_{1}')

    plt.subplot(1, 6, 3)
    plt.imshow(sat2.cpu().numpy(), cmap=cmap)
    plt.colorbar(fraction=0.045, pad=0.06)
    plt.title(f'Predicted: Saturation_{2}')

    plt.subplot(1, 6, 4)
    plt.imshow(sat3.cpu().numpy(), cmap=cmap)
    plt.colorbar(fraction=0.045, pad=0.06)
    plt.title(f'Predicted: Saturation_{3}')

    plt.subplot(1, 6, 5)
    plt.imshow(sat4.cpu().numpy(), cmap=cmap)
    plt.colorbar(fraction=0.045, pad=0.06)
    plt.title(f'Predicted: Saturation_{4}')

    plt.subplot(1, 6, 6)
    plt.imshow(sat5.cpu().numpy(), cmap=cmap)
    plt.colorbar(fraction=0.045, pad=0.06)
    plt.title(f'Predicted: Saturation_{5}')

    # Set colorbar to be centered at 0 for error map
    # plt.subplot(1, 6, 3)
    # error1 = true1.cpu().numpy() - pred1.cpu().numpy()
    # vmin, vmax = 0.0, max(abs(error1.min()), abs(error1.max()))
    # plt.imshow(np.abs(error1), cmap='inferno', vmin=vmin, vmax=vmax)
    # plt.colorbar(fraction=0.045, pad=0.06)
    # plt.title('Error')

    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return

# Read datasets
set_x, set_y, set_vjp, set_eig, set_rolling = [], [], [], [], []
true = []
MSE_rollout = []
JAC_rollout = []
idx = 200 # preferably from test dataset.

org_x, set_x, set_y, set_vjp, set_eig, set_rolling = [], [], [], [], [], []

with h5py.File('../FNO-NF.jl/data/training-data/cons=1e-5_delta=25_num_sample=10000_theta0=5.jld2', 'r') as f:
        # List all the datasets in the file
        print("Keys: %s" % f.keys())
        # Length of K is 10000. Load K
        K = f['perm'][:]
        print(len(K))
        org_x.append(K) # 1, 10000, 64, 64
        org_x = org_x[0]

# Read the each file s_idx: sample index
for s_idx in range(1, 250):
    print("s", s_idx)

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

        # Create vjp and eig
        # vjp = torch.tensor(vjp).reshape(8, 20, 64,64)[:, :args.num_vec]
        # eig = torch.tensor(eigvec).reshape(8, 20, 64, 64)[:, :args.num_vec]
        # print("vjp", vjp.shape)
        # set_vjp.append([vjp[0].numpy()]) 
        # set_vjp.append([vjp[1].numpy()]) 
        # set_vjp.append([vjp[2].numpy()]) 
        # set_vjp.append([vjp[3].numpy()]) 
        # set_vjp.append([vjp[4].numpy()]) 

        # set_eig.append([eig[0].numpy()])
        # set_eig.append([eig[1].numpy()])
        # set_eig.append([eig[2].numpy()])
        # set_eig.append([eig[3].numpy()])
        # set_eig.append([eig[4].numpy()])


# Call models
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device: ", device)
torch.tensor()

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

JAC_path = "../test_result/best_model_FNO_GCS_onestep_full epoch_JAC.pth"
# JAC_path = "../test_result/best_model_FNO_GCS_full epoch_JAC.pth"
JAC_model.load_state_dict(torch.load(JAC_path))
JAC_model.eval()

MSE_path = f"test_result/best_model_FNO_GCS_onestep_full epoch_JAC.pth.pth"
# MSE_path = "../test_result/best_model_FNO_GCS_full epoch_MSE.pth"
MSE_model.load_state_dict(torch.load(MSE_path))
MSE_model.eval()
# 3) True
true_path = f'../test_result/GCS/true_seq_{idx}.png'
mse_path = f'../test_result/GCS/mse_seq_{idx}.png'
jac_path = f'../test_result/GCS/jac_seq_{idx}.png'
mse_diff_path = f'../test_result/GCS/mse_seq_diff_{idx}.png'
jac_diff_path = f'../test_result/GCS/jac_seq_diff_{idx}.png'
single_pred_path = f'../test_result/GCS/ood_one_step_pred_{idx}.png'
true = torch.tensor(true).float()
plot_results(true[idx, 0], true[idx, 1], true[idx, 2], true[idx, 3], true[idx, 4], true[idx, 5], true_path)

# Roll out given input [K, S_1]
K = true[idx, 0]
input_mse, input_jac = torch.stack([K, true[idx, 1]]), torch.stack([K, true[idx, 1]])
print("input", input_mse.shape)
# 1) MSE
for i in range(4):
    out_mse = MSE_model(input_mse.unsqueeze(dim=0).cuda())
    out_mse = out_mse.squeeze().detach().cpu()
    MSE_rollout.append(out_mse)
    input_mse = torch.stack([K.cuda(), out_mse.cuda()])
# 2) JAC
for i in range(4):
    out_jac = JAC_model(input_jac.unsqueeze(dim=0).cuda())
    out_jac = out_jac.squeeze().detach().cpu()
    JAC_rollout.append(out_jac)
    input_jac = torch.stack([K.cuda(), out_jac.cuda()])

plot_results(K, true[idx, 1], MSE_rollout[0], MSE_rollout[1], MSE_rollout[2], MSE_rollout[3], mse_path)
plot_results(K, true[idx, 1], JAC_rollout[0], JAC_rollout[1], JAC_rollout[2], JAC_rollout[3], jac_path)

# plot error
plot_results(None, None, abs(true[idx, 2]-MSE_rollout[0]), abs(true[idx, 3]-MSE_rollout[1]), abs(true[idx, 4]-MSE_rollout[2]), abs(true[idx, 5]-MSE_rollout[3]), mse_diff_path, type="error")
plot_results(None, None, abs(true[idx, 2]-JAC_rollout[0]), abs(true[idx, 3]-JAC_rollout[1]), abs(true[idx, 4]-JAC_rollout[2]), abs(true[idx, 5]-JAC_rollout[3]), jac_diff_path, type="error")

'''
test generalizability
'''

input_4th_mse, input_4th_jac = torch.stack([K, true[idx, 4]]), torch.stack([K, true[idx, 4]])
# MSE single step pred
one_mse = MSE_model(input_4th_mse.unsqueeze(dim=0).cuda()).squeeze().detach().cpu()
# JAC single step pred
one_jac = JAC_model(input_4th_jac.unsqueeze(dim=0).cuda()).squeeze().detach().cpu()

plot_ood(true[idx,4], true[idx, 5], one_mse, one_jac, abs(true[idx, 5]-one_mse), abs(true[idx, 5]-one_jac), single_pred_path)
