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
import matplotlib.colors as colors
from torch.utils.data import Subset

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

def plot_multiple(figures, path, cmap='Blues'):
    fig, axes = plt.subplots(1, 8, figsize=(40, 5))  # Create 1 row and 8 columns
    plt.rcParams.update({'font.size': 16})

    for i, (true1, ax) in enumerate(zip(figures, axes)):
        norm = colors.CenteredNorm()
        im = ax.imshow(true1, cmap=cmap, norm=norm)
        ax.set_title(f'Time step {i+1}')
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.06, norm=norm)

    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_multiple_abs(figures, path, cmap='Blues'):
    fig, axes = plt.subplots(1, 8, figsize=(40, 5))  # Create 1 row and 8 columns
    plt.rcParams.update({'font.size': 16})

    for i, (true1, ax) in enumerate(zip(figures, axes)):
        im = ax.imshow(true1, cmap=cmap)
        ax.set_title(f'Time step {i+1}')
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.06)

    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()


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
# plot_results(Y_test[0,0].cpu(), Y_test[0,1].cpu(), Y_pred[0, 0], Y_pred[0, 1], plot_path)
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

    # plt.subplot(2, 3, 4)
    # plt.imshow(true2.cpu().numpy(), cmap='Blues')
    # plt.colorbar(fraction=0.045, pad=0.06)
    # plt.title('True Saturation')

    # plt.subplot(2, 3, 5)
    # plt.imshow(pred2.cpu().numpy(), cmap='Blues')
    # plt.colorbar(fraction=0.045, pad=0.06)
    # plt.title('Predicted Saturation')

    # # Set colorbar to be centered at 0 for second error map
    # plt.subplot(2, 3, 6)
    # error2 = true2.cpu().numpy() - pred2.cpu().numpy()
    # vmin, vmax = -max(abs(error2.min()), abs(error2.max())), max(abs(error2.min()), abs(error2.max()))
    # plt.imshow(error2, cmap='inferno', vmin=vmin, vmax=vmax)
    # plt.colorbar(fraction=0.045, pad=0.06)
    # plt.title('Error')

    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_single(true1, path):
    plt.figure(figsize=(10, 10))
    plt.rcParams.update({'font.size': 16})

    plt.imshow(true1, cmap='Blues')
    plt.colorbar(fraction=0.045, pad=0.06)
    # plt.title('True Saturation')

    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_data(k, q, T, path):
    plt.rcParams.update({'font.size': 16})
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    im0 = axes[0].imshow(k.squeeze().cpu(), cmap='viridis')
    axes[0].set_title(r"Thermal Conductivity $k$")
    fig.colorbar(im0, ax=axes[0], fraction=0.045, pad=0.06)
    
    im1 = axes[1].imshow(q.cpu(), cmap='inferno')
    axes[1].set_title(r"Heat Source $q$")
    fig.colorbar(im1, ax=axes[1], fraction=0.045, pad=0.06)
    
    im2 = axes[2].imshow(T.cpu().squeeze(), cmap='viridis')
    axes[2].set_title(r"Temperature $T$")
    fig.colorbar(im2, ax=axes[2], fraction=0.045, pad=0.06)
    
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return

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

    path = f"../plot/Loss/checkpoint/FNO_GCS_channel_onestep{loss_type}_{epoch}.png"
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
        in_channels=2,
        out_channels=1,
        decoder_layer_size=128,
        num_fno_layers=6,
        num_fno_modes=[33, 33],
        padding=3,
        dimension=2,
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
    elif args.loss_type == "JAC":
        # Create vec_batch
        True_j = torch.tensor(True_j).float()
        print("True J Before", True_j.shape)
        True_j = True_j.reshape(-1, dataloader.batch_size, args.num_vec, args.nx, args.ny)
        print("After True J", True_j.shape)
        vec = torch.tensor(vec)
        print("vec", vec.shape)
        vec_batch = vec.reshape(-1, dataloader.batch_size, args.num_vec, args.nx, args.ny)
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
            
            # MSE 
            optimizer.zero_grad()
            if args.loss_type == "MSE":
                output = model(X)
                loss = criterion(output.squeeze(), Y.squeeze()) / torch.norm(Y)
                if (epoch == 1) and (idx == 1):
                    plot_single(Y[0].squeeze().detach().cpu().numpy(), f"../plot/GCS_channel_onestep/training/MSE/true_sat_{epoch}.png")
                    plot_single(Y[10].squeeze().detach().cpu().numpy(), f"../plot/GCS_channel_onestep/training/MSE/true_sat2_{epoch}.png")
                    plot_single(Y[15].squeeze().detach().cpu().numpy(), f"../plot/GCS_channel_onestep/training/MSE/true_sat3_{epoch}.png")
                if (epoch % 10 == 0) and (idx == 1):
                    print(Y.shape)
                    plot_single(output[0].squeeze().detach().cpu().numpy(), f"../plot/GCS_channel_onestep/training/MSE/learned_sat_{epoch}.png")
                    plot_single(output[10].squeeze().detach().cpu().numpy(), f"../plot/GCS_channel_onestep/training/MSE/learned_sat2_{epoch}.png")
                    plot_single(output[15].squeeze().detach().cpu().numpy(), f"../plot/GCS_channel_onestep/training/MSE/learned_sat3_{epoch}.png")
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
                target = True_j[idx].cuda()

                '''change: lambda:'''
                # Assume X is a tensor of shape (batch_size, 2), where the first column is K and the second is S.
                K, S = X[:, 0], X[:, 1]
                print("K", K.shape, "S", S.shape)

                # Define a wrapper function that only takes K as input.
                def model_with_fixed_S(K):
                    # Reconstruct the input X using the original S, but replace K with the new K value.
                    modified_X = torch.stack((K, S), dim=1)
                    # Pass this modified input to the model
                    return model(modified_X)

                # Compute the model output and the vjp function with respect to K only
                output, vjp_func = torch.func.vjp(model_with_fixed_S, K)

                loss = criterion(output.squeeze(), Y.squeeze()) / torch.norm(Y)
                vjp_out_list = []
                for e in range(args.num_vec):
                    print("e", e)
                    vjp_out_onevec = vjp_func(vec_batch[idx].unsqueeze(dim=1)[:,:,e])[0] # -> learned vjp
                    # vjp_out[:, :, e] = vjp_out_onevec
                    vjp_out_list.append(vjp_out_onevec)
                    vjp_out = torch.stack(vjp_out_list, dim=2)
                    print("vjp_out", vjp_out.shape)
                
                # vjp_out_org = vjp_func(vec_batch[idx].unsqueeze(dim=1))[0]
                # print("shape", vjp_out_org.shape) #100, 64,64
                # vjp_out = vjp_out_org
                # vjp_out = vjp_out_org[:, 0].squeeze() # choose the pearmeability part
                if (epoch == 1) and (idx == 1):
                    plot_single(Y[0].squeeze().detach().cpu().numpy(), f"../plot/GCS_channel_onestep/training/JAC/true_sat_{epoch}.png")
                    plot_single(Y[10].squeeze().detach().cpu().numpy(), f"../plot/GCS_channel_onestep/training/JAC/true_sat2_{epoch}.png")
                    plot_single(Y[15].squeeze().detach().cpu().numpy(), f"../plot/GCS_channel_onestep/training/JAC/true_sat3_{epoch}.png")
                    plot_single(vec_batch[idx][0][0].detach().cpu().numpy(), f"../plot/GCS_channel_onestep/training/JAC/used_eigvec_{epoch}.png")
                    plot_single(target[0].squeeze().detach().cpu().numpy(), f"../plot/GCS_channel_onestep/training/JAC/true_vjp_{epoch}.png")
                    plot_single(target[4].squeeze().detach().cpu().numpy(), f"../plot/GCS_channel_onestep/training/JAC/true_vjp4_{epoch}.png")
                    plot_single(vjp_out[:][0].squeeze().detach().cpu().numpy(), f"../plot/GCS_channel_onestep/training/JAC/vjp0_{epoch}.png")
                    # plot_single(vjp_out_org[:, 1][0].squeeze().detach().cpu().numpy(), f"../plot/GCS_channel_onestep/training/JAC/vjp1_{epoch}.png")
                if (epoch % 10 == 0) and (idx == 1):
                    print(Y.shape)
                    plot_single(output[0].squeeze().detach().cpu().numpy(), f"../plot/GCS_channel_onestep/training/JAC/learned_sat_{epoch}.png")
                    plot_single(output[10].squeeze().detach().cpu().numpy(), f"../plot/GCS_channel_onestep/training/JAC/learned_sat2_{epoch}.png")
                    plot_single(output[15].squeeze().detach().cpu().numpy(), f"../plot/GCS_channel_onestep/training/JAC/learned_sat3_{epoch}.png")
                    plot_single(vjp_out[:][0].squeeze().detach().cpu().numpy(), f"../plot/GCS_channel_onestep/training/JAC/vjp0_{epoch}.png")
                    # plot_single(vjp_out_org[:, 1][0].squeeze().detach().cpu().numpy(), f"../plot/GCS_channel_onestep/training/JAC/vjp1_{epoch}.png")
                    plot_single(vjp_out[:][4].squeeze().detach().cpu().numpy(), f"../plot/GCS_channel_onestep/training/JAC/vjp4_0_{epoch}.png")
                    # plot_single(vjp_out_org[:, 1][4].squeeze().detach().cpu().numpy(), f"../plot/GCS_channel_onestep/training/JAC/vjp4_1_{epoch}.png")
                
                    # plot_single(vjp_out[0].detach().cpu().numpy(), f"../plot/GCS_channel_onestep/training/JAC/learned_vjp_{epoch}.png")
                # print(target.shape, vjp_out.shape, vec_batch[idx].shape)
                print("target", target.shape, "vjp", vjp_out.shape)
                jac_diff = criterion(target, vjp_out)
                jac_misfit += jac_diff.detach().cpu().numpy() * args.reg_param
                loss += jac_diff * args.reg_param

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
                X_test, Y_test = X_test.cuda().float(), Y_test.cuda().float()
                output = model(X_test)
                test_loss = criterion(output.squeeze(), Y_test) / torch.norm(Y_test)
                full_test_loss += test_loss.item()
            test_diff.append(full_test_loss)
        model.train()

        print(f"Epoch: {epoch}, Train Loss: {full_loss:.6f}, JAC misfit: {jac_misfit}, Test Loss: {full_test_loss:.6f}")
        if epoch % 50 == 0:
            torch.save(model.state_dict(), f"../test_result/Checkpoint/FNO_GCS_onestep_{loss_type}_{args.nx}_{args.num_train}_{epoch}.pth")
            plot_loss_checkpoint(epoch, loss_type, mse_diff, test_diff, jac_diff_list)
        if full_test_loss < lowest_loss:
            print("saved lowest loss model")
            lowest_loss = full_test_loss
            torch.save(model.state_dict(), f"../test_result/best_model_FNO_GCS_onestep_{loss_type}.pth")
            # Save plot
            X_test, Y_test = next(iter(test_dataloader))
            X_test, Y_test = X_test.cuda().float(), Y_test.cuda().float()
            with torch.no_grad():
                Y_pred = model(X_test)
            plot_path = f"../plot/GCS_channel_onestep/FNO_GCS_onestep_lowest_{loss_type}.png"
            plot_results(Y_test[0].squeeze().cpu(), Y_pred[0].squeeze(), plot_path)
                

        if full_loss < args.threshold:
            print("Stopping early as the loss is below the threshold.")
            break

    print("Finished Computing")

    # Save the model
    torch.save(model.state_dict(), f"../test_result/best_model_FNO_GCS_onestep_full epoch_{loss_type}.pth")
    # Save the elapsed times
    with open(f'../test_result/Time/FNO_GCS_onestep_{args.loss_type}_{args.nx}_{args.num_train}.csv', 'w', newline='') as csvfile:
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
            with open(f'../test_result/Losses/GCS_onestep_{name}_{args.loss_type}_{args.nx}_{args.num_train}.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Epoch', 'Loss'])
                writer.writerows(enumerate(data, 1))
    print("Losses saved to CSV files.")

    # Create loss plot
    print("Create loss plot")
    mse_diff = np.asarray(mse_diff)
    jac_diff_list = np.asarray(jac_diff_list)
    test_diff = np.asarray(test_diff)
    path = f"../plot/Loss/FNO_GCS_onestep_{loss_type}.png"

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
    logger.info("%s: %s", "Test Loss", str(full_test_loss/len(test_dataloader)))
    logger.info("%s: %s", "MSE diff", str(mse_diff[-1]))

    return model


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
    parser.add_argument("--num_epoch", type=int, default=800)
    parser.add_argument("--num_train", type=int, default=200)
    parser.add_argument("--num_test", type=int, default=50)
    parser.add_argument("--num_sample", type=int, default=200)
    parser.add_argument("--num_init", type=int, default=60)
    parser.add_argument("--threshold", type=float, default=1e-8)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--loss_type", default="JAC", choices=["MSE", "JAC", "Sobolev", "Dissipative"])
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--ny", type=int, default=64)
    parser.add_argument("--noise", type=float, default=0.01)
    parser.add_argument("--reg_param", type=float, default=10.0)
    parser.add_argument("--nu", type=float, default=0.001) # Viscosity
    parser.add_argument("--time_step", type=float, default=0.01) # time step
    parser.add_argument("--num_vec", type=int, default=3)

    args = parser.parse_args()

    # Save initial settings
    start_time = datetime.datetime.now().strftime("%m_%d_%H_%M_%S")
    out_file = os.path.join("../test_result/", f"FNO_GCS_{start_time}.txt")
    logging.basicConfig(filename=out_file, filemode="w", level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(__name__)
    for arg, value in vars(args).items():
        logger.info("%s: %s", arg, value)

    def plot_series(one, two, three, four, path):
        plt.figure(figsize=(20, 5))
        plt.rcParams.update({'font.size': 16})

        plt.subplot(1, 4, 1)
        plt.imshow(one, cmap='Blues')
        plt.colorbar(fraction=0.045, pad=0.06)
        plt.title('1st year')

        plt.subplot(1, 4, 2)
        plt.imshow(two, cmap='Blues')
        plt.colorbar(fraction=0.045, pad=0.06)
        plt.title('2nd year')

        plt.subplot(1, 4, 3)
        plt.imshow(three, cmap='Blues')
        plt.colorbar(fraction=0.045, pad=0.06)
        plt.title('3rd year')

        plt.subplot(1, 4, 4)
        plt.imshow(four, cmap='Blues')
        plt.colorbar(fraction=0.045, pad=0.06)
        plt.title('4th year')

        plt.tight_layout()
        plt.savefig(path)
        plt.close()

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
            vjp = torch.tensor(vjp).reshape(8, 20, 64,64)[:, :args.num_vec]
            eig = torch.tensor(eigvec).reshape(8, 20, 64, 64)[:, :args.num_vec]
            print("vjp", vjp.shape)
            set_vjp.append([vjp[0].numpy()]) 
            set_vjp.append([vjp[1].numpy()]) 
            set_vjp.append([vjp[2].numpy()]) 
            set_vjp.append([vjp[3].numpy()]) 
            set_vjp.append([vjp[4].numpy()]) 

            set_eig.append([eig[0].numpy()])
            set_eig.append([eig[1].numpy()])
            set_eig.append([eig[2].numpy()])
            set_eig.append([eig[3].numpy()])
            set_eig.append([eig[4].numpy()])

            # Plot every 200th element
            if s_idx % 200 == 0:
                plot_single_abs(set_x[s_idx][0], f"../plot/GCS_channel_onestep/data_permeability:{s_idx}.png")
                print(len(set_y), s_idx)
                plot_multiple_abs(set_y[s_idx], f"../plot/GCS_channel_onestep/data_saturation:{s_idx}.png")
                # if args.loss_type == "JAC":
                    # plot_multiple(set_vjp[s_idx].squeeze(), f"../plot/GCS_channel_onestep/data_vjp:{s_idx}.png")
                    # plot_multiple(set_eig[s_idx].squeeze(), f"../plot/GCS_channel_onestep/data_eigvec:{s_idx}.png")


    print("len or:", len(set_x), len(set_y), len(set_vjp))
    train_x_raw = torch.tensor(set_x[:args.num_train])
    train_y_raw = torch.tensor(set_y[:args.num_train])
    test_x_raw = torch.tensor(set_x[args.num_train:args.num_train+args.num_test])
    test_y_raw = torch.tensor(set_y[args.num_train:args.num_train+args.num_test])

    train_vjp = torch.tensor(set_vjp[:args.num_train]).reshape(-1, 64, 64)
    print("vjp norm", torch.norm(train_vjp))
    train_vjp = train_vjp / torch.norm(train_vjp)

    set_eig = torch.tensor(set_eig[:args.num_train])

    print("len:", len(train_x_raw), len(train_y_raw), len(test_x_raw), len(test_y_raw))

    # Create subsets of the datasets
    train_dataset = CustomDataset(train_x_raw, train_y_raw)
    test_dataset = CustomDataset(test_x_raw, test_y_raw)
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    print("Mini-batch: ", len(train_loader), train_loader.batch_size)

    # train
    main(logger, args, args.loss_type, train_loader, test_loader, train_vjp, set_eig, set_rolling, test_x_raw)