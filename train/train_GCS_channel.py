import torch
import torch.nn as nn
import torch.autograd.functional as F
import torch.optim as optim
import datetime
import time
import numpy as np
import argparse
import json
import logging
import os
import csv
import h5py
import pandas as pd
import tqdm
import math
from torch.func import vmap, vjp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import seaborn as sGCS
from functorch import vjp, vmap
from torch.utils.data import Subset
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap

from torch.utils.data import Dataset, DataLoader, TensorDataset
from modulus.models.fno import FNO
from modulus.models.mlp.fully_connected import FullyConnected

class Timer:
    def __init__(self):
        self.elapsed_times = []

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.elapsed_time = self.end_time - self.start_time
        self.elapsed_times.append(self.elapsed_time)
        return False


### Dataset ###
class GCSDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    

### Auxiliary Function ###
def save_dataset_to_csv(dataset, prefix):
    k_data = []
    
    for k in dataset:
        k_data.append(k)
    
    k_df = pd.DataFrame(k_data)
    k_df.to_csv(f'{prefix}', index=False)
    print(f"Saved {prefix} dataset to CSV files")
    return

def load_dataset_from_csv(prefix, nx, ny):
    df = pd.read_csv(f'{prefix}')

    print(f'df Length: {len(df)}')
    print(f'df Shape: {df.shape}')
    
    data = [torch.tensor(row.values).reshape(nx, ny) for _, row in df.iterrows()]
    
    return data

def log_likelihood(data, model_output, noise_std):
    # return -0.5 * torch.sum((data - model_output)**2) / (noise_std**2) - \
    #     data.numel() * torch.log(torch.tensor(noise_std))
    return (1/(2*noise_std**2))*torch.sum((data - model_output)**2)

def compute_fim_GCS(simulator, input, T_data, noise_std, nx, ny, forcing, time_step, Re):
    # EGCSure k is a tensor with gradient tracking
    q = input.requires_grad_().cuda()
    fim = torch.zeros((nx*ny, nx*ny))
    
    # # Add noise
    mean = 0.0
    std_dev = 0.1

    # Generate Gaussian noise
    noise = torch.randn(q.size()) * std_dev + mean
    # Solve heat equation
    # w_current = GCS_solver(w_current, f=forcing, T=time_step, Re=Re)
    T_pred = simulator(q, f=forcing, T=time_step, Re=Re)
    T_pred = T_pred + noise.cuda()
    ll = log_likelihood(T_data.cuda(), T_pred, noise_std)
    flat_Jacobian = torch.autograd.grad(inputs=q, outputs=ll, create_graph=True)[0].flatten() # 50 by 50 -> [2500]
    print("flatten", flat_Jacobian.shape)
    flat_Jacobian = flat_Jacobian.reshape(1, -1)
    fim = torch.matmul(flat_Jacobian.T, flat_Jacobian)

    return fim

# Sobolev norm (HS norm)
# where we also compare the numerical derivatives between the output and target
# credit: 
class HsLoss_2d(object):
    def __init__(self, d=2, p=2, k=1, a=None, group=False, size_average=True, reduction=True):
        super(HsLoss_2d, self).__init__()

        #DimeGCSion and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.k = k
        self.balanced = group
        self.reduction = reduction
        self.size_average = size_average

        if a == None:
            a = [1,] * k
        self.a = a

    def rel(self, x, y):
        num_examples = x.size()[0]
        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)
        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)
        return diff_norms/y_norms

    def __call__(self, x, y, a=None):
        nx = x.size()[1]
        ny = x.size()[2]
        k = self.k
        balanced = self.balanced
        a = self.a
        x = x.view(x.shape[0], nx, ny, -1)
        y = y.view(y.shape[0], nx, ny, -1)

        k_x = torch.cat((torch.arange(start=0, end=nx//2, step=1),torch.arange(start=-nx//2, end=0, step=1)), 0).reshape(nx,1).repeat(1,ny)
        k_y = torch.cat((torch.arange(start=0, end=ny//2, step=1),torch.arange(start=-ny//2, end=0, step=1)), 0).reshape(1,ny).repeat(nx,1)
        k_x = torch.abs(k_x).reshape(1,nx,ny,1).to(x.device)
        k_y = torch.abs(k_y).reshape(1,nx,ny,1).to(x.device)

        x = torch.fft.fftn(x, dim=[1, 2])
        y = torch.fft.fftn(y, dim=[1, 2])

        if balanced==False:
            weight = 1
            if k >= 1:
                weight += a[0]**2 * (k_x**2 + k_y**2)
            if k >= 2:
                weight += a[1]**2 * (k_x**4 + 2*k_x**2*k_y**2 + k_y**4)
            weight = torch.sqrt(weight)
            loss = self.rel(x*weight, y*weight)
        else:
            loss = self.rel(x, y)
            if k >= 1:
                weight = a[0] * torch.sqrt(k_x**2 + k_y**2)
                loss += self.rel(x*weight, y*weight)
            if k >= 2:
                weight = a[1] * torch.sqrt(k_x**4 + 2*k_x**2*k_y**2 + k_y**4)
                loss += self.rel(x*weight, y*weight)
            loss = loss / (k+1)

        return loss


### Compute Metric ###
def plot_results(true1, pred1, path):
    plt.figure(figsize=(15, 5))
    plt.rcParams.update({'font.size': 16})

    plt.subplot(1, 3, 1)
    plt.imshow(true1.cpu().numpy(), cmap='Blues')
    plt.colorbar(fraction=0.045, pad=0.06)
    plt.title('True Saturation')

    plt.subplot(1, 3, 2)
    plt.imshow(pred1.cpu().numpy(), cmap='Blues')
    plt.colorbar(fraction=0.045, pad=0.06)
    plt.title('Predicted Saturation')

    # Set colorbar to be centered at 0 for error map
    plt.subplot(1, 3, 3)
    error1 = true1.cpu().numpy() - pred1.cpu().numpy()
    vmin, vmax = 0.0, max(abs(error1.min()), abs(error1.max()))
    plt.imshow(np.abs(error1), cmap='inferno', vmin=vmin, vmax=vmax)
    plt.colorbar(fraction=0.045, pad=0.06)
    plt.title('Error')

    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()

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


def plot_loss_checkpoint(epoch, loss_type, mse_diff, test_diff, jac_diff_list=None):
    # Create loss plot
    print("Create loss plot")
    if epoch < 510:
        mse_diff = np.asarray(mse_diff)
        jac_diff_list = np.asarray(jac_diff_list)
        test_diff = np.asarray(test_diff)
    else: 
        start_epoch = 30
        mse_diff = mse_diff[start_epoch:]
        test_diff = test_diff[start_epoch:]
        jac_diff_list = jac_diff_list[start_epoch:]  # Only if JAC is relevant

    path = f"../plot/Loss/checkpoint/FNO_GCS_vec:{args.num_vec}_{loss_type}_{epoch}.png"
    epochs = np.arange(len(mse_diff))

    fig, ax = plt.subplots()
    ax.plot(epochs, mse_diff, "P-", lw=1.0, ms=4.0, color="coral", label="MSE (Train)")
    ax.plot(epochs, test_diff, "P-", lw=1.0, ms=4.0, color="blue", label="MSE (Test)")
    if args.loss_type == "JAC":
        ax.plot(epochs, jac_diff_list, "P-", lw=1.0, color="slateblue", ms=4.0, label=r"$\|J^Tv - \hat{J}^Tv\|$")
    ax.set_xlabel("Epochs",fontsize=24)
    ax.set_ylabel("Loss", fontsize=24)
    ax.legend()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    return


### Train ###

def main(logger, args, loss_type, dataloader, test_dataloader, True_j, vec, rolling, test_x):
    # Initialization
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device: ", device)

    model = FNO(
        in_channels=1,
        out_channels=1,
        decoder_layer_size=128,
        num_fno_layers=5,
        num_fno_modes=[8, 15, 15],
        padding=3,
        dimension=3,
        latent_channels=64
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-5
    )

    criterion = torch.nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1e-3)
    nx, ny = args.nx, args.ny

    # Gradient-matching and training logic
    if args.loss_type == "Sobolev":
        Sobolev_Loss = HsLoss_2d()
    elif args.loss_type == "Dissipative":
        Sobolev_Loss = HsLoss_2d()
        # DISSIPATIVE REGULARIZATION PARAMETERS
        # below, the number before multiplication by S is the radius in the L2 norm of the function space
        S=args.nx
        radius = 156.25 * S # radius of inner ball
        scale_down = 0.5 # rate at which to linearly scale down inputs
        loss_weight = 0.01 * (S**2) # normalized by L2 norm in function space
        radii = (radius, (525 * S) + radius) # inner and outer radii, in L2 norm of function space
        sampling_fn = sample_uniform_spherical_shell #numsampled is batch size
        target_fn = linear_scale_dissipative_target
        dissloss = nn.MSELoss(reduction='mean')

        modes = 20
        width = 64

        in_dim = 1
        out_dim = 1
    # elif args.loss_type == "JAC":
    # Create vjp_batch
    True_j = torch.tensor(True_j).float()
    print("True J Before", True_j.shape) #True J Before torch.Size([700, 8, 64, 64])
    True_j = True_j.reshape(-1, dataloader.batch_size, 8, args.num_vec, args.nx, args.ny)
    print("After True J", True_j.shape) #([7, 100, 8, 64, 64]) -> ([idx, batchsize, 8, 64, 64])
    vec = torch.tensor(vec)
    print("vec", vec.shape)
    vec_batch = vec.reshape(-1, dataloader.batch_size, 8, args.num_vec, args.nx, args.ny)
    print("vec", vec_batch.shape)
    vec_batch = vec_batch.cuda().float()


    ### Training Loop ###
    elapsed_time_train, mse_diff, jac_diff_list, test_diff = [], [], [], []
    lowest_loss = float('inf')

    print("Beginning training")
    for epoch in range(args.num_epoch):
        start_time = time.time()
        full_loss, full_test_loss, jac_misfit = 0.0, 0.0, 0.0
        idx = 0
        
        for X, Y in dataloader:
            X, Y = X.cuda().float(), Y.cuda().float()
            X = X.unsqueeze(1)
            Y = Y.unsqueeze(1)
            # print("X", "Y", X.shape, Y.shape) #[batchsize, 8, 64, 64]
            
            # MSE 
            if args.loss_type == "MSE":
                output = model(X)
                loss = criterion(output.squeeze(), Y.squeeze()) / torch.norm(Y)
                # target = True_j[idx].cuda()
                # output, vjp_func = torch.func.vjp(model, X)
                # vjp_out_onevec = vjp_func(cur_vec_batch[:, :, e])[0] # -> learned vjp
                # vjp_out = vjp_func(vec_batch[idx])[0]
                # jac_diff = criterion(target, vjp_out)
                # vjp_out = torch.flip(vjp_out, dims=[-1])
                if (epoch == 1) and (idx == 0):
                    plot_multiple_abs(Y[0].squeeze().detach().cpu().numpy(), f"../plot/GCS_channel_vec_{args.num_vec}/training/MSE/true_sat_{epoch}.png")
                    plot_multiple_abs(Y[1].squeeze().detach().cpu().numpy(), f"../plot/GCS_channel_vec_{args.num_vec}/training/MSE/true_sat2_{epoch}.png")
                    # plot_multiple(vec_batch[idx][0].detach().cpu().numpy(), f"../plot/GCS_channel_vec_{args.num_vec}/training/MSE/true_eigvec_{epoch}.png", cmap)
                    # plot_multiple(target[0].detach().cpu().numpy(), f"../plot/GCS_channel_vec_{args.num_vec}/training/MSE/true_vjp_{epoch}.png", cmap)
                    # plot_multiple(target[1].detach().cpu().numpy(), f"../plot/GCS_channel_vec_{args.num_vec}/training/MSE/true_vjp_{epoch}_1.png", cmap)
                if (epoch % 10 == 0) and (idx == 0):
                    print(output.shape)
                    plot_multiple_abs(output[0].squeeze().detach().cpu().numpy(), f"../plot/GCS_channel_vec_{args.num_vec}/training/MSE/learned_sat_{epoch}.png")
                    plot_multiple_abs(output[1].squeeze().detach().cpu().numpy(), f"../plot/GCS_channel_vec_{args.num_vec}/training/MSE/learned_sat2_{epoch}.png")
                    # plot_multiple(vjp_out[0].detach().cpu().numpy(), f"../plot/GCS_channel_vec_{args.num_vec}/training/MSE/learned_vjp_{epoch}.png", cmap)
                    # plot_multiple(vjp_out[1].detach().cpu().numpy(), f"../plot/GCS_channel_vec_{args.num_vec}/training/MSE/learned_vjp_{epoch}_1.png", cmap)
                    plot_multiple_abs(abs(output[0]-Y[0]).squeeze().detach().cpu().numpy(), f"../plot/GCS_channel_vec_{args.num_vec}/training/MSE/diff_sat_{epoch}.png", "magma")
                    plot_multiple_abs(abs(output[1]-Y[1]).squeeze().detach().cpu().numpy(), f"../plot/GCS_channel_vec_{args.num_vec}/training/MSE/diff_sat2_{epoch}.png", "magma")
                    # plot_multiple_abs(abs(target[0]-vjp_out[0]).detach().cpu().numpy(), f"../plot/GCS_channel_vec_{args.num_vec}/training/MSE/diff_vjp_{epoch}.png", "magma")
                    # plot_multiple_abs(abs(target[1]-vjp_out[1]).detach().cpu().numpy(), f"../plot/GCS_channel_vec_{args.num_vec}/training/MSE/diff_vjp_{epoch}_1.png", "magma")
            elif args.loss_type == "Sobolev":
                output = model(X.unsSqueeze(dim=1))
                loss = criterion(output.squeeze(), Y.squeeze()) / torch.norm(Y)
                sob_loss = Sobolev_Loss(output.squeeze(), Y.squeeze())
                loss += sob_loss
            elif args.loss_type == "Dissipative":
                output = model(X.unsqueeze(dim=1))
                loss = Sobolev_Loss(output.squeeze(), Y.squeeze())
                x_diss = torch.tensor(sampling_fn(X.shape[0], radii, (S, S, 2)), dtype=torch.float).to(device)
                y_diss = torch.tensor(target_fn(x_diss, scale_down), dtype=torch.float).to(device)
                out_diss = model(x_diss.reshape(-1, 2, S, S)).reshape(-1, out_dim)
                diss_loss = (1/(S**2)) * loss_weight * dissloss(out_diss, y_diss.reshape(-1, out_dim)) # weighted by 1 / (S**2)
                loss += diss_loss
            else:
            # GM
                print("idx", idx)
                # 1. update vjp and eigenvector
                target_vjp = True_j[idx].cuda()
                target_vjp = target_vjp.unsqueeze(1)  # Shape becomes [5, 1, 8, 15, 64, 64]
                target_vjp = target_vjp.permute(0, 1, 3, 2, 4, 5) # shape [5, 1, 15, 8, 64, 64]
                print("target vjp", target_vjp.shape)
                cur_vec_batch = vec_batch[idx] # 50 x 8 x 3 x 64 x 64
                # vjp_out = torch.zeros(dataloader.batch_size, 8, args.num_vec, 64, 64, requires_grad=True).to('cuda')
                
                # 2. compute MSE and GM loss term
                output, vjp_func = torch.func.vjp(model, X)
                loss = criterion(output.squeeze(), Y.squeeze()) / torch.norm(Y)
                vjp_out_list = []
                for e in range(args.num_vec):
                    print("e", e)
                    vjp_out_onevec = vjp_func(cur_vec_batch[:, :, e].unsqueeze(1))[0] # -> learned vjp
                    # vjp_out[:, :, e] = vjp_out_onevec
                    vjp_out_list.append(vjp_out_onevec)
                    vjp_out = torch.stack(vjp_out_list, dim=2)
                # vjp_out = torch.func.vmap(lambda vec_batch: vjp_func(vec_batch.unsqueeze(1))[0], in_dims=2, chunk_size=2)(cur_vec_batch)
                # vjp_out = vjp_out.permute(1,2,0,3,4,5)
                print("vjp_out", vjp_out.shape)
                # plot_multiple_abs(cur_vec_batch[0, :, e].detach().cpu().numpy(), f"../plot/GCS_channel_vec_{args.num_vec}/debug/true_vec_first_sample_{e}.png")
                plot_multiple_abs(vjp_out[0, : ,0].squeeze().detach().cpu().numpy(), f"../plot/GCS_channel_vec_{args.num_vec}/learned_vjp_first_sample.png", "seismic")
                # plot_multiple_abs(vjp_out_onevec[1].detach().cpu().numpy(), f"../plot/GCS_channel_vec_{args.num_vec}/debug/learned_vjp_second_sample_{e}.png")
    
                if (epoch == 1) and (idx == 0):
                    plot_multiple(X[0,0].squeeze().detach().cpu().numpy(), f"../plot/GCS_channel_vec_{args.num_vec}/training/JAC/K_{epoch}.png")
                    plot_multiple(X[1,0].squeeze().detach().cpu().numpy(), f"../plot/GCS_channel_vec_{args.num_vec}/training/JAC/K2_{epoch}.png")
                    plot_multiple_abs(Y[0].squeeze().detach().cpu().numpy(), f"../plot/GCS_channel_vec_{args.num_vec}/training/JAC/true_sat_{epoch}.png")
                    plot_multiple_abs(Y[1].squeeze().detach().cpu().numpy(), f"../plot/GCS_channel_vec_{args.num_vec}/training/JAC/true_sat2_{epoch}.png")
                    plot_multiple(cur_vec_batch[0, :, 0].squeeze().detach().cpu().numpy(), f"../plot/GCS_channel_vec_{args.num_vec}/training/JAC/true_eigvec_{epoch}.png", cmap)
                    plot_multiple(cur_vec_batch[1, :, 0].squeeze().detach().cpu().numpy(), f"../plot/GCS_channel_vec_{args.num_vec}/training/JAC/true_eigvec2_{epoch}.png", cmap)
                    plot_multiple(target_vjp[0, :, 0].squeeze().detach().cpu().numpy(), f"../plot/GCS_channel_vec_{args.num_vec}/training/JAC/true_vjp_{epoch}.png", cmap)
                    plot_multiple(target_vjp[1, :, 0].squeeze().detach().cpu().numpy(), f"../plot/GCS_channel_vec_{args.num_vec}/training/JAC/true_vjp_{epoch}_1.png", cmap)
                if (epoch % 10 == 0) and (idx == 0):
                    plot_multiple_abs(output[0].squeeze().detach().cpu().numpy(), f"../plot/GCS_channel_vec_{args.num_vec}/training/JAC/learned_sat_{epoch}.png")
                    plot_multiple_abs(output[1].squeeze().detach().cpu().numpy(), f"../plot/GCS_channel_vec_{args.num_vec}/training/JAC/learned_sat2_{epoch}.png")
                    plot_multiple(vjp_out[0, :, 0].squeeze().detach().cpu().numpy(), f"../plot/GCS_channel_vec_{args.num_vec}/training/JAC/learned_vjp_{epoch}.png", cmap)
                    plot_multiple(vjp_out[1, :, 0].squeeze().detach().cpu().numpy(), f"../plot/GCS_channel_vec_{args.num_vec}/training/JAC/learned_vjp_{epoch}_1.png", cmap)
                    plot_multiple_abs(abs(output[0]-Y[0]).squeeze().detach().cpu().numpy(), f"../plot/GCS_channel_vec_{args.num_vec}/training/JAC/diff_sat_{epoch}.png", "magma")
                    plot_multiple_abs(abs(output[1]-Y[1]).squeeze().detach().cpu().numpy(), f"../plot/GCS_channel_vec_{args.num_vec}/training/JAC/diff_sat2_{epoch}.png", "magma")
                    plot_multiple_abs(abs(target_vjp[0, :, 0]-vjp_out[0, :, 0]).squeeze().detach().cpu().numpy(), f"../plot/GCS_channel_vec_{args.num_vec}/training/JAC/diff_vjp_{epoch}.png", "magma")
                    plot_multiple_abs(abs(target_vjp[1, :, 0]-vjp_out[1, :, 0]).squeeze().detach().cpu().numpy(), f"../plot/GCS_channel_vec_{args.num_vec}/training/JAC/diff_vjp_{epoch}_1.png", "magma")
                # print(target.shape, vjp_out.shape, vec_batch[idx].shape)
                jac_diff = criterion(target_vjp, vjp_out)
                jac_misfit += jac_diff.detach().cpu().numpy()
                loss += jac_diff * args.reg_param

                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
            full_loss += loss.item()
            idx += 1
        
        # Save loss
        mse_diff.append(full_loss)
        if args.loss_type == "JAC":
            jac_diff_list.append(jac_misfit)
        # Save time
        end_time = time.time()  
        elapsed_time_train.append(end_time - start_time)
        
        # Validation
        model.eval()
        with torch.no_grad():
            for X_test, Y_test in test_dataloader:
                X_test = X_test.unsqueeze(1)
                Y_test = Y_test.unsqueeze(1)
                X_test, Y_test = X_test.cuda().float(), Y_test.cuda().float()
                output = model(X_test)
                test_loss = criterion(output.squeeze(), Y_test) / torch.norm(Y_test) # relative error
                full_test_loss += test_loss.item()
            test_diff.append(full_test_loss)
        model.train()

        print(f"Epoch: {epoch}, Train Loss: {full_loss:.6f}, JAC misfit: {jac_misfit}, Test Loss: {full_test_loss:.6f}")
        if epoch % 50 == 0:
            torch.save(model.state_dict(), f"../test_result/Checkpoint/FNO_GCS_vec_{args.num_vec}_{loss_type}_{args.nx}_{args.num_train}_{epoch}.pth")
            plot_loss_checkpoint(epoch, loss_type, mse_diff, test_diff, jac_diff_list)
        if full_test_loss < lowest_loss:
            print("saved lowest loss model")
            lowest_loss = full_test_loss
            torch.save(model.state_dict(), f"../test_result/best_model_FNO_GCS_vec_{args.num_vec}_{loss_type}.pth")
            # Save plot
            X_test, Y_test = next(iter(test_dataloader))
            X_test, Y_test = X_test.unsqueeze(1).cuda().float(), Y_test.unsqueeze(1).cuda().float()
            with torch.no_grad():
                Y_pred = model(X_test)
            plot_path = f"../plot/GCS_channel_vec_{args.num_vec}/FNO_GCS_lowest_vec_{args.num_vec}_{loss_type}_True.png"
            plot_multiple_abs(Y_test[0].squeeze().cpu(), plot_path)
            plot_multiple_abs(Y_pred[0].squeeze().detach().cpu(), f"../plot/GCS_channel_vec_{args.num_vec}/FNO_GCS_lowest_{loss_type}_Pred.png")
            plot_multiple_abs(abs(Y_pred[0]-Y_test[0]).squeeze().detach().cpu(), f"../plot/GCS_channel_vec_{args.num_vec}/FNO_GCS_lowest_{loss_type}_diff.png", "magma")
                

        if full_loss < args.threshold:
            print("Stopping early as the loss is below the threshold.")
            break

    print("Finished Computing")

    # Save the model
    torch.save(model.state_dict(), f"../test_result/best_model_FNO_GCS_full epoch_{loss_type}_vec_{args.num_vec}.pth")
    # Save the elapsed times
    with open(f'../test_result/Time/FNO_GCS_{args.loss_type}_{args.nx}_{args.num_train}.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Epoch', 'Elapsed Time (seconds)'])
        for epoch, elapsed_time in enumerate(elapsed_time_train, 1):
            writer.writerow([epoch, elapsed_time])
    # Save the losses
    loss_data = [
        (mse_diff, 'mse_loss'),
        (jac_diff_list, 'jac_loss') if args.loss_type == "JAC" else (None, None),
        (test_diff, 'test_loss')
    ]
    for data, name in loss_data:
        if data:
            with open(f'../test_result/Losses/GCS_{name}_{args.loss_type}_{args.nx}_{args.num_train}_vec_{args.num_vec}.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Epoch', 'Loss'])
                writer.writerows(enumerate(data, 1))
    print("Losses saved to CSV files.")

    # Create loss plot
    print("Create loss plot")
    mse_diff = np.asarray(mse_diff)
    jac_diff_list = np.asarray(jac_diff_list)
    test_diff = np.asarray(test_diff)
    path = f"../plot/Loss/FNO_GCS_channel_{loss_type}_vec_{args.num_vec}.png"

    # Remove the first few epochs (e.g., the first 5 epochs)
    start_epoch = 30
    mse_diff = mse_diff[start_epoch:]
    test_diff = test_diff[start_epoch:]
    jac_diff_list = jac_diff_list[start_epoch:]  # Only if JAC is relevant

    # Create new index for x-axis starting from 0 after removing epochs
    epochs = np.arange(len(mse_diff))

    fig, ax = plt.subplots()
    ax.plot(epochs, mse_diff, "P-", lw=1.0, ms=4.0, color="coral", label="MSE (Train)")
    ax.plot(epochs, test_diff, "P-", lw=1.0, ms=4.0, color="blue", label="MSE (Test)")

    if args.loss_type == "JAC":
        ax.plot(epochs, jac_diff_list, "P-", lw=1.0, color="slateblue", ms=4.0, label=r"$\|J^Tv - \hat{J}^Tv\|$")

    ax.set_xlabel("Epochs", fontsize=24)
    ax.set_ylabel("Loss", fontsize=24)
    ax.legend()
    plt.savefig(path, dpi=150, bbox_inches="tight")

    print("Plot saved.")


    logger.info("%s: %s", "Loss Type", str(loss_type))
    logger.info("%s: %s", "Batch Size", str(args.batch_size))
    logger.info("%s: %s", "Training Loss", str(full_loss/len(dataloader)))
    logger.info("%s: %s", "Final Test Loss", str(full_test_loss/len(test_dataloader)))
    logger.info("%s: %s", "Lowest Test Loss", str(lowest_loss))
    logger.info("%s: %s", "MSE diff", str(mse_diff[-1]))

    return model


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


if __name__ == "__main__":
    # Set device
    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.backends.cudnn.benchmark = True
    print("device: ", device)

    # Set arguments (hyperparameters)
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--num_epoch", type=int, default=200)
    parser.add_argument("--num_train", type=int, default=200)
    parser.add_argument("--num_test", type=int, default=50)
    parser.add_argument("--num_sample", type=int, default=200)
    # parser.add_argument("--num_init", type=int, default=60)
    parser.add_argument("--threshold", type=float, default=1e-8)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--loss_type", default="JAC", choices=["MSE", "JAC", "Sobolev", "Dissipative"])
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--ny", type=int, default=64)
    parser.add_argument("--noise", type=float, default=0.01)
    parser.add_argument("--reg_param", type=float, default=10.0) # 0.1 -> 2
    parser.add_argument("--nu", type=float, default=0.001) # Viscosity
    parser.add_argument("--time_step", type=float, default=0.01) # time step
    parser.add_argument("--num_vec", type=int, default=10)

    args = parser.parse_args()

    # Save initial settings
    start_time = datetime.datetime.now().strftime("%m_%d_%H_%M_%S")
    out_file = os.path.join("../test_result/", f"FNO_GCS_{start_time}.txt")
    logging.basicConfig(filename=out_file, filemode="w", level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(__name__)
    for arg, value in vars(args).items():
        logger.info("%s: %s", arg, value)

    cmap = LinearSegmentedColormap.from_list(
        "cmap_name",
        ["#0000FF", "white", "#FF0000"]
    )


    set_x, set_y, set_vjp, set_eig, set_rolling = [], [], [], [], []

    with h5py.File('../FNO-NF.jl/data/training-data/cons=1e-5_delta=25_num_sample=10000_theta0=5.jld2', 'r') as f:
        # List all the datasets in the file
        print("Keys: %s" % f.keys())
        # Length of K is 10000. Load K
        K = f['perm'][:]
        print(len(K))
        set_x.append(K) # 1, 10000, 64, 64
        set_x = set_x[0]

    # Read the each file s_idx: sample index
    for s_idx in range(1, args.num_train+args.num_test+1):

        with h5py.File(f'../gen_sample/num_obs_20/states_sample_{s_idx}_nobs_20.jld2', 'r') as f1, \
            h5py.File(f'../gen_sample/num_obs_20/FIM_eigvec_sample_{s_idx}_nobs_20.jld2', 'r') as f2, \
            h5py.File(f'../gen_sample/num_obs_20/FIM_vjp_sample_{s_idx}_nobs_20.jld2', 'r') as f3:

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
            set_vjp.append(torch.tensor(vjp).reshape(8, 20, 64, 64)[:, :args.num_vec]) 
            set_eig.append(torch.tensor(eigvec).reshape(8, 20, 64, 64)[:, :args.num_vec])


            # Plot every 200th element
            if s_idx % 200 == 0:
                plot_single_abs(set_x[s_idx-1], f"../plot/GCS_channel_vec_{args.num_vec}/data_permeability:{s_idx}.png")
                print(len(set_y), s_idx)
                plot_multiple_abs(set_y[s_idx-1], f"../plot/GCS_channel_vec_{args.num_vec}/data_saturation:{s_idx}.png")
                if args.loss_type == "JAC":
                    plot_multiple(torch.tensor(set_vjp[s_idx-1][0]), f"../plot/GCS_channel_vec_{args.num_vec}/data_vjp:{s_idx}.png", "seismic")
                    # plot_multiple(set_eig[s_idx-1], f"../plot/GCS_channel_vec_{args.num_vec}/data_eigvec:{s_idx}.png")


    print("len or:", len(set_x), len(set_x[0]), len(set_y), len(set_vjp))
    train_x_raw = torch.tensor(set_x[:args.num_train])
    train_y_raw = torch.stack(set_y[:args.num_train])
    test_x_raw = torch.tensor(set_x[args.num_train:args.num_train+args.num_test])
    test_y_raw = torch.stack(set_y[args.num_train:args.num_train+args.num_test])

    train_x_raw = train_x_raw.unsqueeze(1)  # Now tensor is [25, 1, 64, 64]
    train_x_raw = train_x_raw.repeat(1, 8, 1, 1)  # Now tensor is [25, 8, 64, 64]

    test_x_raw = test_x_raw.unsqueeze(1)  # Now tensor is [25, 1, 64, 64]
    test_x_raw = test_x_raw.repeat(1, 8, 1, 1)  # Now tensor is [25, 8, 64, 64]

    # normalize train_y_raw and train_vjp and set_eig
    def normalize_to_range(x, new_min=-1.0, new_max=1.0):
        """
        Normalize the tensor x to the range [new_min, new_max]
        """
        old_min = torch.min(x)
        old_max = torch.max(x)
        x_norm = (new_max - new_min) * (x - old_min) / (old_max - old_min) + new_min
        return x_norm

    train_vjp = torch.stack(set_vjp[:args.num_train]).reshape(-1, 64, 64)
    print("vjp norm", torch.norm(train_vjp), torch.max(train_vjp))
    train_vjp = train_vjp / torch.norm(train_vjp)
    # train_vjp = train_vjp / (10**13)

    set_eig = torch.stack(set_eig[:args.num_train])
    print("len: ", len(train_vjp), len(set_eig))
    print("len:", len(train_x_raw), len(train_y_raw), len(test_x_raw), len(test_y_raw))

    # Create subsets of the datasets
    train_dataset = CustomDataset(train_x_raw, train_y_raw)
    test_dataset = CustomDataset(test_x_raw, test_y_raw)
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    print("Mini-batch: ", len(train_loader), train_loader.batch_size)

    # train
    main(logger, args, args.loss_type, train_loader, test_loader, train_vjp, set_eig, None, test_x_raw)