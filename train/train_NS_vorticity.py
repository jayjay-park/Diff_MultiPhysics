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
import pandas as pd
import tqdm
import math
from torch.func import vmap, vjp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from functorch import vjp, vmap
from torch.utils.data import Subset

import sys
sys.path.append('../test')
from generate_NS_org import *
from PINO_NS import *
from baseline import *

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

def generate_dataset(num_samples, num_init, time_step, nx=50, ny=50):
    input, output, init = [], [], []

    L1, L2 = 2*math.pi, 2*math.pi  # Domain size
    Re = 1000  # Reynolds number
    # Define a forcing function (or set to None)
    t = torch.linspace(0, 1, nx + 1, device="cuda")
    t = t[0:-1]
    X, Y = torch.meshgrid(t, t)
    forcing = 0.1 * (torch.sin(2 * math.pi * (X + Y)) + torch.cos(2 * math.pi * (X + Y)))

    # Initialize Navier-Stokes solver
    ns_solver = NavierStokes2d(nx, ny, L1=L1, L2=L2, device="cuda")
    num_iter = int(num_samples/num_init)
    print("num_init: ", num_init)
    print("time step: ", num_iter)
    for s in range(num_init):
        print("gen data for init: ", s)
        
        # Generate initial vorticity field
        random_seed=42 + s
        w = gaussian_random_field_2d((nx, ny), 20, random_seed)
        init.append(w)
        w_current = w.cuda()
        vorticity_data = [w_current.cpu().numpy()]

        # Solve the NS
        for i in range(num_iter):
            w_current = ns_solver(w_current, f=forcing, T=time_step, Re=Re)
            vorticity_data.append(w_current.cpu().numpy())
        
        input.append(vorticity_data[:-1])
        output.append(vorticity_data[1:])
        # print("input length", len(input[0]),"output length", len(output[0]))
        # print("input 1", input[s][1], "output 0", output[s][0])
        # print("input -1", input[s][-1], "output -2", output[s][-2])
        
    return input, output, init



def save_plot(vx_init, vy_init, vx_final, vy_final, filename):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    plt.rcParams.update({'font.size': 14})

    # Plot vx field
    im1 = axes[0, 0].imshow(vx_init.cpu().numpy(), cmap='RdBu')
    axes[0, 0].set_title('Velocity Field vx')
    axes[0, 0].invert_yaxis()
    axes[0, 0].axis('off')
    fig.colorbar(im1, ax=axes[0, 0], orientation='vertical')

    # Plot vy field
    im2 = axes[0, 1].imshow(vy_init.cpu().numpy(), cmap='RdBu')
    axes[0, 1].set_title('Input Velocity Field vy')
    axes[0, 1].invert_yaxis()
    axes[0, 1].axis('off')
    fig.colorbar(im2, ax=axes[0, 1], orientation='vertical')

    # Plot input vorticity field
    im3 = axes[1, 0].imshow(vx_final.cpu().numpy(), cmap='RdBu')
    axes[1, 0].set_title('Output Velocity Field vx')
    axes[1, 0].invert_yaxis()
    axes[1, 0].axis('off')
    fig.colorbar(im3, ax=axes[1, 0], orientation='vertical')

    # Plot output vorticity field
    im4 = axes[1, 1].imshow(vy_final.cpu().numpy(), cmap='RdBu')
    axes[1, 1].set_title('Output Velocity Field vy')
    axes[1, 1].invert_yaxis()
    axes[1, 1].axis('off')
    fig.colorbar(im4, ax=axes[1, 1], orientation='vertical')

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close(fig)

class NSDataset(torch.utils.data.Dataset):
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

def compute_fim_NS(simulator, input, T_data, noise_std, nx, ny, forcing, time_step, Re, input_index, s, num_observations):
    '''
    s: index of training data
    T_data index: [1, ... , 43]
    '''
    # Ensure k is a tensor with gradient tracking
    q = input.requires_grad_().cuda()
    fim = torch.zeros((nx*ny, nx*ny))

    # Generate isotrophic gaussian noise
    for j in range(num_observations):
        normal = torch.randn(nx, ny)
        gaussian_noise = noise_std * normal
        T_pred = simulator(q, f=forcing, T=time_step, Re=Re)
        T_pred = T_pred + gaussian_noise.cuda()
        ll = log_likelihood(T_data.cuda(), T_pred, noise_std)
        flat_Jacobian = torch.autograd.grad(inputs=q, outputs=ll, create_graph=True)[0].flatten() # 50 by 50 -> [2500]
        flat_Jacobian = flat_Jacobian.reshape(1, -1)
        fim += torch.matmul(flat_Jacobian.T, flat_Jacobian).detach().cpu()
        if (j == 9) or (j == 49) or (j == 99):
            plot_single(fim, f"../plot/NS_plot/{num_observations}/fim_{input_index}_{j}_t={s}.png", "viridis")
            plot_single(fim[:100,:100], f"../plot/NS_plot/{num_observations}/fim_sub_{input_index}_{j}_t={s}.png", "viridis")
            plot_single(fim[:,0].reshape(nx, ny), f"../plot/NS_plot/{num_observations}/fim_sub_reshape_{input_index}_{j}_t={s}.png", "viridis")


    return fim

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

    path = f"../plot/Loss/checkpoint/FNO_NS_vort_{loss_type}_{epoch}.png"
    epochs = np.arange(len(mse_diff))

    fig, ax = plt.subplots()
    ax.plot(epochs, mse_diff, "P-", lw=1.0, ms=4.0, color="red", label="MSE (Train)")
    ax.plot(epochs, test_diff, "P-", lw=1.0, ms=4.0, color="blue", label="MSE (Test)")
    if args.loss_type == "JAC":
        ax.plot(epochs, jac_diff_list, "P-", lw=1.0, color="black", ms=4.0, label=r"$\|J^Tv - \hat{J}^Tv\|$")
    ax.set_xlabel("Epochs",fontsize=24)
    ax.set_ylabel("Loss", fontsize=24)
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    return

### Compute Metric ###
# plot_results(Y_test[0,0].cpu(), Y_test[0,1].cpu(), Y_pred[0, 0], Y_pred[0, 1], plot_path)
def plot_results(true1, true2, pred1, pred2, path):
    plt.figure(figsize=(15, 10))
    plt.rcParams.update({'font.size': 16})

    plt.subplot(2, 3, 1)
    plt.imshow(true1.cpu().numpy(), cmap='jet')
    plt.colorbar(fraction=0.045, pad=0.06)
    plt.title('True vorticity')

    plt.subplot(2, 3, 2)
    plt.imshow(pred1.cpu().numpy(), cmap='jet')
    plt.colorbar(fraction=0.045, pad=0.06)
    plt.title('Predicted vorticity')

    plt.subplot(2, 3, 3)
    error1 = true1.cpu().numpy() - pred1.cpu().numpy()
    vmin, vmax = 0.0, max(abs(error1.min()), abs(error1.max()))
    plt.imshow(np.abs(error1), cmap='inferno', vmin=vmin, vmax=vmax)
    plt.colorbar(fraction=0.045, pad=0.06)
    plt.title('Error')

    plt.subplot(2, 3, 4)
    plt.imshow(true2.cpu().numpy(), cmap='jet')
    plt.colorbar(fraction=0.045, pad=0.06)
    plt.title('True Vorticity')

    plt.subplot(2, 3, 5)
    plt.imshow(pred2.cpu().numpy(), cmap='jet')
    plt.colorbar(fraction=0.045, pad=0.06)
    plt.title('Predicted Vorticity')

    plt.subplot(2, 3, 6)
    error2 = true2.cpu().numpy() - pred2.cpu().numpy()
    vmin, vmax = 0.0, max(abs(error2.min()), abs(error2.max()))
    plt.imshow(np.abs(error2), cmap='inferno', vmin=vmin, vmax=vmax)
    plt.colorbar(fraction=0.045, pad=0.06)
    plt.title('Error')

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




### Train ###

def main(logger, args, loss_type, dataloader, test_dataloader, vec, simulator):
    # Initialization
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device: ", device)

    model = FNO(
        in_channels=1,  # Adjusted for vx and vy inputs
        out_channels=1, # Adjusted for wz output
        decoder_layer_size=128,
        num_fno_layers=6,
        num_fno_modes=[32,32],
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
    for first_batch in test_dataloader:
        print(first_batch)
        break  # Stop after printing the first batch

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
        csv_filename = f'../data/true_j_NS_{nx}_{args.num_train}_{args.num_obs}.csv'
        if os.path.exists(csv_filename):
            # Load True_j
            True_j_flat = pd.read_csv(csv_filename).values
            print("len", True_j_flat.shape, len(dataloader)*dataloader.batch_size*nx*ny)
            True_j = torch.tensor(True_j_flat)[:len(dataloader)*dataloader.batch_size, :].reshape(len(dataloader), dataloader.batch_size, nx, ny)
            print(f"Data loaded from {csv_filename}")
        else:
            True_j = torch.zeros(len(dataloader), dataloader.batch_size, nx, ny)
            f = lambda x: simulator(x, f=forcing, T=args.time_step, Re=Re)
            # Iterate over the DataLoader
            index_vec = 0
            print("vec", vec.shape)
            for batch_idx, (batch_data, batch_labels) in enumerate(dataloader):
                for i in range(batch_data.shape[0]):  # Iterate over each sample in the batch
                    # single sample [nx, ny]
                    x = batch_data[i]
                    output, vjp_tru_func = torch.func.vjp(f, x.cuda())
                    print(batch_idx, i, index_vec)
                    vjp = vjp_tru_func(vec[index_vec].cuda())[0].detach().cpu()
                    True_j[batch_idx, i] = vjp
                    if index_vec < 30:
                        plot_single(vjp, f'../plot/NS_plot/FIM/num_obs={args.num_obs}/{args.num_obs}_vjp_{index_vec}.png')
                    index_vec += 1

            # Save True_j to a CSV file
            True_j_flat = True_j.reshape(-1, nx * ny)  # Flatten the last two dimensions
            pd.DataFrame(True_j_flat.numpy()).to_csv(csv_filename, index=False)
            print(f"Data saved to {csv_filename}")
        # Create vec_batch
        True_j = True_j.float()

        # Create vec_batch
        print("before reshape eigvec:", vec.shape)
        vec_batch = vec.reshape(-1, dataloader.batch_size, nx, ny) 
        print("after reshape eigvec:", vec.shape)
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
                output = model(X.unsqueeze(dim=1))
                loss = criterion(output.squeeze(), Y.squeeze()) / torch.norm(Y)
            elif args.loss_type == "Sobolev":
                output = model(X.unsqueeze(dim=1))
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
                output, vjp_func = torch.func.vjp(model, X.unsqueeze(dim=1))
                loss = criterion(output.squeeze(), Y.squeeze()) / torch.norm(Y)
                vjp_out = vjp_func(vec_batch[idx].unsqueeze(dim=1))[0].squeeze()
                plot_single(vec_batch[idx][0].detach().cpu(), f'../plot/NS_plot/vjp_true.png')
                plot_single(vjp_out[0].detach().cpu(), f'../plot/NS_plot/vjp_pred.png')
                jac_diff = criterion(target, vjp_out)
                jac_misfit += jac_diff.detach().cpu().numpy()
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
                output = model(X_test.unsqueeze(dim=1))
                test_loss = criterion(output.squeeze(), Y_test) / torch.norm(Y_test)
                full_test_loss += test_loss.item()
            test_diff.append(full_test_loss)
        model.train()
        
        print(f"Epoch: {epoch}, Train Loss: {full_loss:.6f}, JAC misfit: {jac_misfit}, Test Loss: {full_test_loss:.6f}")
        if epoch % 50 == 0:
            torch.save(model.state_dict(), f"../test_result/Checkpoint/FNO_NS_vort_{loss_type}_{args.nx}_{args.num_train}_{epoch}.pth")
            with torch.no_grad():
                Y_pred = model(first_batch[0].float().cuda().unsqueeze(dim=1))
            plot_loss_checkpoint(epoch, loss_type, mse_diff, test_diff, jac_diff_list)
            plot_path = f"../plot/NS_plot/checkpoint/FNO_NS_vort_{epoch}.png"
            plot_results(first_batch[1][0].squeeze().cpu(), first_batch[1][1].squeeze().cpu(), Y_pred[0].squeeze(), Y_pred[1].squeeze(), plot_path)
                
        if full_test_loss < lowest_loss:
            print("saved lowest loss model")
            lowest_loss = full_test_loss
            torch.save(model.state_dict(), f"../test_result/best_model_FNO_NS_vort_{loss_type}.pth")
            # Save plot
            with torch.no_grad():
                Y_pred = model(first_batch[0].float().cuda().unsqueeze(dim=1))
            plot_path = f"../plot/NS_plot/FNO_NS_vort_lowest_{loss_type}.png"
            plot_results(first_batch[1][0].squeeze().cpu(), first_batch[1][1].squeeze().cpu(), Y_pred[0].squeeze(), Y_pred[1].squeeze(), plot_path)
                
        if full_loss < args.threshold:
            print("Stopping early as the loss is below the threshold.")
            break

    print("Finished Computing")

    # Save the model
    torch.save(model.state_dict(), f"../test_result/best_model_FNO_NS_vort_full epoch_{loss_type}.pth")
    # Save the elapsed times
    with open(f'../test_result/Time/FNO_NS_vort_{args.loss_type}_{args.nx}_{args.num_train}.csv', 'w', newline='') as csvfile:
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
            with open(f'../test_result/Losses/NS_vort_{name}_{args.loss_type}_{args.nx}_{args.num_train}.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Epoch', 'Loss'])
                writer.writerows(enumerate(data, 1))
    print("Losses saved to CSV files.")

    # Create loss plot
    print("Create loss plot")
    plot_loss_checkpoint(epoch, loss_type, mse_diff, test_diff, jac_diff_list)
    print("Plot saved.")


    logger.info("%s: %s", "Loss Type", str(loss_type))
    logger.info("%s: %s", "Batch Size", str(args.batch_size))
    logger.info("%s: %s", "Training Loss", str(full_loss/len(dataloader)))
    logger.info("%s: %s", "Test Loss", str(full_test_loss/len(test_dataloader)))
    logger.info("%s: %s", "MSE diff", str(mse_diff[-1]))

    return model


def plot_single(true1, path, cmap="magma"):
    plt.figure(figsize=(10, 10))
    plt.rcParams.update({'font.size': 16})

    plt.imshow(true1, cmap=cmap)
    plt.colorbar(fraction=0.045, pad=0.06)
    # plt.title('True Saturation')

    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    # Set device
    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.backends.cudnn.benchmark = True
    print("device: ", device)

    # Set arguments: https://github.com/neuraloperator/neuraloperator/blob/main/config/navier_stokes_config.yaml
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--num_epoch", type=int, default=1000)
    parser.add_argument("--num_train", type=int, default=2000) #8000
    parser.add_argument("--num_test", type=int, default=200)
    parser.add_argument("--num_sample", type=int, default=2000) #8000
    parser.add_argument("--num_init", type=int, default=50)
    parser.add_argument("--threshold", type=float, default=1e-8)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--loss_type", default="JAC", choices=["MSE", "JAC", "Sobolev", "Dissipative"])
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--ny", type=int, default=64)
    parser.add_argument("--noise", type=float, default=0.01)
    parser.add_argument("--reg_param", type=float, default=200.0)
    parser.add_argument("--nu", type=float, default=0.001) # Viscosity
    parser.add_argument("--time_step", type=float, default=0.05) # time step
    parser.add_argument("--num_obs", type=float, default=10) # time step

    args = parser.parse_args()

    # Save initial settings
    start_time = datetime.datetime.now().strftime("%m_%d_%H_%M_%S")
    out_file = os.path.join("../test_result/", f"FNO_NS_vort_{start_time}.txt")
    logging.basicConfig(filename=out_file, filemode="w", level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(__name__)
    for arg, value in vars(args).items():
        logger.info("%s: %s", arg, value)

    # Define Simulator
    L1, L2 = 2*math.pi, 2*math.pi  # Domain size
    Re = 1000  # Reynolds number
    # Define a forcing function (or set to None)
    t = torch.linspace(0, 1, args.nx + 1, device="cuda")
    t = t[0:-1]
    X, Y = torch.meshgrid(t, t)
    forcing = 0.1 * (torch.sin(2 * math.pi * (X + Y)) + torch.cos(2 * math.pi * (X + Y)))
    ns_solver = NavierStokes2d(args.nx, args.ny, L1=L1, L2=L2, device="cuda")

    # Generate Training/Test Data
    trainx_file = f'../data/NS_vort/train_x_{args.nx}_{args.ny}_{args.num_train}_{args.num_init}_{args.num_obs}.csv'
    trainy_file = f'../data/NS_vort/train_y_{args.nx}_{args.ny}_{args.num_train}_{args.num_init}_{args.num_obs}.csv'
    testx_file = f'../data/NS_vort/test_x_{args.nx}_{args.ny}_{args.num_test}_{args.num_init}_{args.num_obs}.csv'
    testy_file = f'../data/NS_vort/test_y_{args.nx}_{args.ny}_{args.num_test}_{args.num_init}_{args.num_obs}.csv'
    if not os.path.exists(trainx_file):
        print("Creating Dataset")
        input, output, init = generate_dataset(args.num_train + args.num_test, args.num_init, args.time_step, args.nx, args.ny)
        input = torch.tensor(input).reshape(-1, args.nx*args.ny)
        output = torch.tensor(output).reshape(-1, args.nx*args.ny)
        print("data size", len(input), len(output))

        train_x = NSDataset(input[:args.num_train].numpy())
        train_y = NSDataset(output[:args.num_train].numpy())
        test_x = NSDataset(input[args.num_train:].numpy())
        test_y = NSDataset(output[args.num_train:].numpy())
        # Save datasets to CSV files
        save_dataset_to_csv(train_x, trainx_file)
        save_dataset_to_csv(train_y, trainy_file)
        save_dataset_to_csv(test_x, testx_file)
        save_dataset_to_csv(test_y, testy_file)

    
    print("Loading Dataset")
    sample = load_dataset_from_csv(trainx_file, args.nx, args.ny)
    print("sample", len(sample), sample[0].shape)
    train_x_raw = load_dataset_from_csv(trainx_file, args.nx, args.ny)
    train_y_raw = load_dataset_from_csv(trainy_file, args.nx, args.ny)
    test_x_raw = load_dataset_from_csv(testx_file, args.nx, args.ny)
    test_y_raw = load_dataset_from_csv(testy_file, args.nx, args.ny)

    def normalize_to_range(x, new_min=-1.0, new_max=1.0):
        """
        Normalize the tensor x to the range [new_min, new_max]
        """
        old_min = torch.min(x)
        old_max = torch.max(x)
        x_norm = (new_max - new_min) * (x - old_min) / (old_max - old_min) + new_min
        return x_norm

    # Normalize each sample
    # train_x_raw = torch.stack([normalize_to_range(sample) for sample in train_x_raw])
    # train_y_raw = torch.stack([normalize_to_range(sample) for sample in train_y_raw])
    # test_x_raw = torch.stack([normalize_to_range(sample) for sample in test_x_raw])
    # test_y_raw = torch.stack([normalize_to_range(sample) for sample in test_y_raw])
    # init = [normalize_to_range(torch.tensor(sample)) for sample in init]
    plot_single(train_x_raw[0].reshape(args.nx, args.ny), f'../plot/NS_plot/input.png')
    plot_single(train_x_raw[-1].reshape(args.nx, args.ny), f'../plot/NS_plot/output.png')


    # Randomly sample indices for train and test sets
    # train_indices = np.random.choice(len(train_x_raw), args.num_train, replace=False)
    # test_indices = np.random.choice(len(test_y_raw), args.num_test, replace=False)
    # # Create subsets of the datasets
    train_dataset = CustomDataset(train_x_raw, train_y_raw)
    test_dataset = CustomDataset(test_x_raw, test_y_raw)
    # train_dataset = Subset(train_dataset, train_indices)
    # test_dataset = Subset(test_dataset, test_indices)
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    print("Mini-batch: ", len(train_loader), train_loader.batch_size)

    # compute FIM eigenvector
    if args.loss_type == "JAC":
        csv_filename = f'../data/NS_vort/largest_eigvec_NS_{args.nx}_{args.num_train}_{args.num_obs}.csv'
        if os.path.exists(csv_filename):
            print("Loading largest eigenvector")
            largest_eigenvector = pd.read_csv(csv_filename).values
            largest_eigenvector = torch.tensor(largest_eigenvector)
        else:
            largest_eigenvector = []
            nx, ny = args.nx, args.ny
            noise_std = 1.
            print("Reloaded train: ", train_x_raw[0].shape)
            # Compute FIM
            init_iter = int((args.num_train + args.num_test)/args.num_init)
            init_index = 0
            input_param = init[init_index]
            for s in range(args.num_train):
                print(s, init_index)
                if s == 0:
                    print("s", train_x_raw[s])
                    print("init", init[init_index])
                    # save gradient 
                    grad_vorticity_x = np.gradient(input_param, axis=0)
                    grad_vorticity_y = np.gradient(input_param, axis=1)
                    mag_vorticity = np.sqrt(grad_vorticity_x**2 + grad_vorticity_y**2)
                    plot_single(mag_vorticity, f'../plot/NS_plot/init_grad_{s}.png', "Purples")
                # should be changed to initial state.
                if (s % (init_iter) == 0) and (s != 0):
                    init_index += 1
                    input_param = init[init_index]
                    # save gradient 
                    grad_vorticity_x = np.gradient(input_param, axis=0)
                    grad_vorticity_y = np.gradient(input_param, axis=1)
                    mag_vorticity = np.sqrt(grad_vorticity_x**2 + grad_vorticity_y**2)
                    plot_single(mag_vorticity, f'../plot/NS_plot/init_grad_{s}.png', "Purples")
                    print("s-1", train_x_raw[s-1])
                    print("s", train_x_raw[s])
                    print("s+1", train_x_raw[s+1])
                    print("init", init[init_index])
                    
                fim = compute_fim_NS(ns_solver, input_param, train_y_raw[s], noise_std, nx, ny, forcing, args.time_step, Re, init_index, s, num_observations=args.num_obs).detach().cpu()
                # Analyze the FIM
                eigenvalues, eigenvec = torch.linalg.eigh(fim.cuda())
                largest_eigenvector.append(eigenvec[0].detach().cpu())
                if s < 30:
                    print("eigval: ", eigenvalues)
                    print("shape", train_x_raw[s].shape)
                    plot_single(train_x_raw[s].detach().cpu(), f'../plot/NS_plot/FIM/num_obs={args.num_obs}/{args.num_obs}_state_{s}.png', "Purples")
                    plot_single(eigenvec[0].detach().cpu().reshape(args.nx, args.ny), f'../plot/NS_plot/FIM/num_obs={args.num_obs}/{args.num_obs}_eigenvec0_{s}.png', "Purples")
                    plot_single(eigenvec[1].detach().cpu().reshape(args.nx, args.ny), f'../plot/NS_plot/FIM/num_obs={args.num_obs}/{args.num_obs}_eigenvec1_{s}.png', "Purples")
                    plot_single(eigenvec[2].detach().cpu().reshape(args.nx, args.ny), f'../plot/NS_plot/FIM/num_obs={args.num_obs}/{args.num_obs}_eigenvec2_{s}.png', "Purples")
                    plot_single(eigenvalues.detach().cpu().reshape(args.nx, args.ny), f'../plot/NS_plot/FIM/num_obs={args.num_obs}/{args.num_obs}_eigenvalues_{s}.png', "Purples")
            largest_eigenvector = torch.stack(largest_eigenvector)
            pd.DataFrame(largest_eigenvector.numpy()).to_csv(csv_filename, index=False)
            print(f"Data saved to {csv_filename}")
                
        # print("shape", eigenvalues.shape, eigenvec.shape) -> torch.Size([2500]) torch.Size([2500, 2500])
        # Get the eigenvector corresponding to the largest eigenvalue
        # Assuming eigenvalues and eigenvectors are already computed
        # eigenvalues: A tensor of eigenvalues
        # eigenvec: A matrix where each column corresponds to an eigenvector

        # # Sort eigenvalues in descending order and get their indices
        # sorted_indices = torch.argsort(eigenvalues, descending=True)

        # # Iterate over sorted eigenvalues to find the largest one with a non-zero eigenvector
        # largest_eigenvector = None
        # for idx in sorted_indices:
        #     print("candidate idx", idx)
        #     candidate_eigenvector = eigenvec[:, idx]
            
        #     # Check if the eigenvector is non-zero
        #     if torch.any(candidate_eigenvector != 0):
        #         largest_eigenvector = candidate_eigenvector
        #         break
        # # Handle the case where no non-zero eigenvector is found
        # if largest_eigenvector is None:
        #     raise ValueError("No non-zero eigenvector found.")

        # # idx = torch.argmax(eigenvalues)
        # # largest_eigenvector = eigenvec[:, idx]
        # largest_eigenvector = largest_eigenvector.reshape(args.nx, args.ny)

        # print("Largest Eigenvalue and index:", eigenvalues[idx], idx)
        # print("Corresponding Eigenvector:", largest_eigenvector)
        # print("Eigenvector shape", largest_eigenvector.shape)
        # print("eigenvalue: ", eigenvalues)
        # print("eigenvector: ", eigenvec)
        print("largest eigenvector shape: ", largest_eigenvector.shape)
        largest_eigenvector = largest_eigenvector.reshape(-1, args.nx, args.ny)
    else:
        largest_eigenvector = None
    for data in enumerate(train_loader):
        print("from loader", data)
    # train
    main(logger, args, args.loss_type, train_loader, test_loader, largest_eigenvector, ns_solver)