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
import math
from torch.func import vmap, vjp
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import seaborn as sns
from functorch import vjp, vmap
from torch.utils.data import Subset

from torch.utils.data import DataLoader, TensorDataset
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

### Heat Equation ###
# def solve_heat_equation(k, q, nx=50, ny=50, num_iterations=1000):
#     dx = dy = 1.0 / (nx - 1)
#     T = torch.zeros((nx, ny), device=k.device)  # Initialize with boundary temperature
#     T[0, :] = T[-1, :] = T[:, 0] = T[:, -1] = 0
#     k = k * 0.1
    
#     for _ in range(num_iterations):
#         T_old = T.clone()
#         T[1:-1, 1:-1] = (
#             k[1:-1, 1:-1] * (T_old[2:, 1:-1] / k[2:, 1:-1] + T_old[:-2, 1:-1] / k[:-2, 1:-1] + 
#                              T_old[1:-1, 2:] / k[1:-1, 2:] + T_old[1:-1, :-2] / k[1:-1, :-2])
#             + dx * dy * q[1:-1, 1:-1]  # Changed sign to positive
#         ) / (k[1:-1, 1:-1] * (1/k[2:, 1:-1] + 1/k[:-2, 1:-1] + 1/k[1:-1, 2:] + 1/k[1:-1, :-2]))
        
#         # Boundary conditions (Dirichlet)
#         T[0, :] = T[-1, :] = T[:, 0] = T[:, -1] = 0
    
#     return T


# def solve_heat_equation(k, q, nx=50, ny=50, num_iterations=1000):
#     dx = dy = 1.0 / (nx - 1)
#     T = torch.zeros((nx, ny), device=k.device)  # Initialize temperature with boundary temperature
#     # Boundary conditions (Dirichlet)
#     T[0, :] = 100  # Left boundary
#     T[-1, :] = 200  # Right boundary
#     T[:, 0] = 150  # Bottom boundary
#     T[:, -1] = 250  # Top boundary
    
#     # Modify k to introduce stiffness
#     k[:nx//2, :ny//2] = 1.0  # Low conductivity in the bottom-left quadrant
#     k[nx//2:, ny//2:] = 1000.0  # High conductivity in the top-right quadrant
    
#     for _ in range(num_iterations):
#         T_old = T.clone()
#         T[1:-1, 1:-1] = (
#             k[1:-1, 1:-1] * (
#                 T_old[2:, 1:-1] / k[2:, 1:-1] + 
#                 T_old[:-2, 1:-1] / k[:-2, 1:-1] + 
#                 T_old[1:-1, 2:] / k[1:-1, 2:] + 
#                 T_old[1:-1, :-2] / k[1:-1, :-2]
#             )
#             + dx * dy * q[1:-1, 1:-1]
#         ) / (
#             k[1:-1, 1:-1] * (
#                 1/k[2:, 1:-1] + 1/k[:-2, 1:-1] + 
#                 1/k[1:-1, 2:] + 1/k[1:-1, :-2]
#             )
#         )

#     return T

def solve_heat_equation(k, q, nx=50, ny=50, num_iterations=1000):
    dx = dy = 1.0 / (nx - 1)
    T = torch.zeros((nx, ny), device=k.device)  # Initialize temperature with boundary temperature
    # Boundary conditions (Dirichlet)
    T[0, :] = 100  # Left boundary
    T[-1, :] = 200  # Right boundary
    T[:, 0] = 150  # Bottom boundary
    T[:, -1] = 250  # Top boundary
    
    # Modify k to introduce extreme stiffness
    k[:nx//3, :ny//3] = 0.1      # Very low conductivity in the bottom-left region
    k[nx//3:2*nx//3, ny//3:2*ny//3] = 10.0  # Moderate conductivity in the middle region
    k[2*nx//3:, 2*ny//3:] = 10000.0  # Extremely high conductivity in the top-right region
    k[:nx//3, 2*ny//3:] = 0.01  # Extremely low conductivity in the top-left region
    k[2*nx//3:, :ny//3] = 5000.0  # Very high conductivity in the bottom-right region
    
    for _ in range(num_iterations):
        T_old = T.clone()
        T[1:-1, 1:-1] = (
            k[1:-1, 1:-1] * (
                T_old[2:, 1:-1] / k[2:, 1:-1] + 
                T_old[:-2, 1:-1] / k[:-2, 1:-1] + 
                T_old[1:-1, 2:] / k[1:-1, 2:] + 
                T_old[1:-1, :-2] / k[1:-1, :-2]
            )
            + dx * dy * q[1:-1, 1:-1]
        ) / (
            k[1:-1, 1:-1] * (
                1/k[2:, 1:-1] + 1/k[:-2, 1:-1] + 
                1/k[1:-1, 2:] + 1/k[1:-1, :-2]
            )
        )

    return T

# def solve_heat_equation(k, q, nx=50, ny=50, num_iterations=1000):
#     dx = dy = 1.0 / (nx - 1)
#     T = torch.zeros((nx, ny), device=k.device)  # Initialize temperature with boundary temperature
#     # Boundary conditions (Dirichlet)
#     T[0, :] = 100  # Left boundary
#     T[-1, :] = 200  # Right boundary
#     T[:, 0] = 150  # Bottom boundary
#     T[:, -1] = 250  # Top boundary

#     blocks = [
#     (8, 20, 8, 20, 10000.0),   # Block 1 with high conductivity
#     (22, 34, 10, 22, 500.0),  # Block 2 with medium-high conductivity
#     (36, 48, 24, 36, 100.0),  # Block 3 with medium-low conductivity
#     (44, 56, 44, 56, 1000.0),   # Block 4 with low conductivity
#     (12, 28, 30, 46, 5000.0),   # Block 5 with different conductivity
#     (52, 64, 20, 32, 8000.0),  # Block 6 with different conductivity
#     (0, 8, 50, 64, 50.0),    # Block 7 with different conductivity
#     (40, 48, 0, 8, 300.0)     # Block 8 with different conductivity
#     ]

    
#     # Initialize conductivity with default low value
#     k[:] = 0.01

#     # Apply each block's conductivity
#     for (x1, x2, y1, y2, k_val) in blocks:
#         k[x1:x2, y1:y2] = k_val

#     for _ in range(num_iterations):
#         T_old = T.clone()
#         T[1:-1, 1:-1] = (
#             k[1:-1, 1:-1] * (
#                 T_old[2:, 1:-1] / k[2:, 1:-1] + 
#                 T_old[:-2, 1:-1] / k[:-2, 1:-1] + 
#                 T_old[1:-1, 2:] / k[1:-1, 2:] + 
#                 T_old[1:-1, :-2] / k[1:-1, :-2]
#             )
#             + dx * dy * q[1:-1, 1:-1]
#         ) / (
#             k[1:-1, 1:-1] * (
#                 1/k[2:, 1:-1] + 1/k[:-2, 1:-1] + 
#                 1/k[1:-1, 2:] + 1/k[1:-1, :-2]
#             )
#         )

#     return T




### Dataset ###
def create_q_function(nx, ny, noise_level=0.1, pattern='sinusoidal', freq_mean=5., freq_std=1.):
    # Create a grid of x and y coordinates
    x = torch.linspace(0, 1, nx)
    y = torch.linspace(0, 1, ny)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    # Different patterns for q(x, y)
    if pattern == 'sinusoidal':
        # Sample frequencies from a Gaussian distribution
        freq_x = torch.normal(mean=torch.tensor(freq_mean), std=torch.tensor(freq_std))
        freq_y = torch.normal(mean=torch.tensor(freq_mean - 2), std=torch.tensor(freq_std))
        q = 3000 * (torch.sin(freq_x * X) * torch.sin(freq_y * Y) + torch.cos(freq_x * Y) * torch.cos((freq_y-1) * X))
    elif pattern == 'gaussian':
        q = 3000 * torch.exp(-((X-0.5)**2 + (Y-0.5)**2) / (2 * 0.1**2))
    elif pattern == 'random_noise':
        q = 3000 * torch.randn(nx, ny)
    else:
        raise ValueError("Unknown pattern type")
    
    # Add noise
    noise = noise_level * torch.randn_like(q)
    q_noisy = q + noise
    
    return q_noisy

def create_q_function_inference(nx, ny, fx_mean, fy_mean, noise_level=0.1, pattern='sinusoidal', freq_std=1.):
    # Create a grid of x and y coordinates
    x = torch.linspace(0, 1, nx)
    y = torch.linspace(0, 1, ny)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    # Different patterns for q(x, y)
    if pattern == 'sinusoidal':
        # Sample frequencies from a Gaussian distribution
        freq_x = torch.normal(mean=torch.tensor(fx_mean), std=torch.tensor(freq_std))
        freq_y = torch.normal(mean=torch.tensor(fy_mean), std=torch.tensor(freq_std))
        q = 3000 * (torch.sin(freq_x * X) * torch.sin(freq_y * Y) + torch.cos(freq_x * Y) * torch.cos((freq_y-1) * X))
    elif pattern == 'gaussian':
        q = 3000 * torch.exp(-((X-0.5)**2 + (Y-0.5)**2) / (2 * 0.1**2))
    elif pattern == 'random_noise':
        q = 3000 * torch.randn(nx, ny)
    else:
        raise ValueError("Unknown pattern type")
    
    # Add noise
    noise = noise_level * torch.randn_like(q)
    q_noisy = q + noise
    
    return q_noisy

def generate_dataset(num_samples, nx=50, ny=50, pattern='sinusoidal'):
    input = []
    output = []
    for s in range(num_samples):
        print(s)
        
        # constant k
        k = torch.ones(nx, ny)
        # Generate q with the selected pattern
        q = create_q_function(nx, ny, noise_level=0.1, pattern=pattern)
        # Solve the heat equation
        T = solve_heat_equation(k, q, nx, ny)
        
        input.append(q)
        output.append(T)
        
        if s == 0:
            plot_path = f"../plot/Heat_plot/Heat_q_{pattern}.png"
            plot_data(k, q, T, plot_path)
        elif s == 1:
            plot_path = f"../plot/Heat_plot/Heat_q_{pattern}(1).png"
            plot_data(k, q, T, plot_path)

    return input, output



class HeatDataset(torch.utils.data.Dataset):
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
        k_data.append(k.flatten().tolist())
    
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
    return -0.5 * torch.sum((data - model_output)**2) / (noise_std**2) - \
        data.numel() * torch.log(torch.tensor(noise_std))

def compute_fim_for_2d_heat(solve_heat_equation, q, T_data, noise_std, nx=50, ny=50):
    # Ensure k is a tensor with gradient tracking
    q = torch.tensor(q, requires_grad=True).cuda()
    k = torch.ones(nx, ny).cuda()
    fim = torch.zeros((nx*ny, nx*ny))
    
    # Add noise
    mean = 0.0
    std_dev = 0.05

    # Generate Gaussian noise
    noise = torch.randn(q.size()) * std_dev + mean
    # Solve heat equation
    T_pred = solve_heat_equation(k, q, nx, ny) + noise.cuda()
    ll = log_likelihood(T_data.cuda(), T_pred.cuda(), noise_std)
    flat_Jacobian = torch.autograd.grad(inputs=q, outputs=ll, create_graph=True)[0].flatten() # 50 by 50 -> [2500]
    flat_Jacobian = flat_Jacobian.reshape(1, -1)
    fim = torch.matmul(flat_Jacobian.T, flat_Jacobian)

    return fim

### Compute Metric ###
def plot_results(k, T_true, T_pred, path):
    plt.rcParams.update({'font.size': 16})
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    im0 = axes[0].imshow(k.squeeze().cpu(), cmap='inferno')
    axes[0].set_title(r"Heat Source $q$")
    fig.colorbar(im0, ax=axes[0], fraction=0.045, pad=0.06)
    
    im1 = axes[1].imshow(T_true.cpu(), cmap='viridis')
    axes[1].set_title("True Temperature (T)")
    fig.colorbar(im1, ax=axes[1], fraction=0.045, pad=0.06)
    
    im2 = axes[2].imshow(T_pred.cpu().squeeze(), cmap='viridis')
    axes[2].set_title("Predicted Temperature (T)")
    fig.colorbar(im2, ax=axes[2], fraction=0.045, pad=0.06)
    
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return

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
def main(logger, args, loss_type, dataloader, test_dataloader, vec):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device: ", device)

    model = FNO(
        in_channels=1,
        out_channels=1,
        decoder_layer_size=128,
        num_fno_layers=6,
        num_fno_modes=24,
        padding=3,
        dimension=2,
        latent_channels=64
    ).to('cuda')


    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-5
    )

    criterion = torch.nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1e-3)
    nx, ny = args.nx, args.ny

    ### Gradient-matching ###
    if args.loss_type == "JAC":
        csv_filename = f'../data/true_j_{nx}_{ny}_{args.num_train}.csv'
        if os.path.exists(csv_filename):
            # Load True_j
            True_j_flat = pd.read_csv(csv_filename).values
            print("len", True_j_flat.shape, len(dataloader)*dataloader.batch_size*nx*ny)
            True_j = torch.tensor(True_j_flat)[:len(dataloader)*dataloader.batch_size, :].reshape(len(dataloader), dataloader.batch_size, nx, ny)
            print(f"Data loaded from {csv_filename}")
        else:
            True_j = torch.zeros(len(dataloader), dataloader.batch_size, nx, ny)
            k = torch.ones(nx, ny)
            f = lambda x: solve_heat_equation(k.cuda(), x, nx, ny)
            # Iterate over the DataLoader
            for batch_idx, (batch_data, batch_labels) in enumerate(dataloader):
                for i in range(batch_data.shape[0]):  # Iterate over each sample in the batch
                    # single sample [nx, ny]
                    x = batch_data[i]
                    output, vjp_tru_func = torch.func.vjp(f, x.cuda())
                    print(batch_idx, i)
                    True_j[batch_idx, i] = vjp_tru_func(vec)[0].detach().cpu()
                    print(True_j[batch_idx, i])

            # Save True_j to a CSV file
            True_j_flat = True_j.reshape(-1, nx * ny)  # Flatten the last two dimensions
            pd.DataFrame(True_j_flat.numpy()).to_csv(csv_filename, index=False)
            print(f"Data saved to {csv_filename}")
        # Create vec_batch
        True_j = True_j.float()
        vec_batch = vec.unsqueeze(0).repeat(dataloader.batch_size, 1, 1)
        vec_batch = vec_batch.cuda().float()
        

    ### Training Loop ###
    elapsed_time_train, mse_diff, jac_diff_list, test_diff = [], [], [], []
    lowest_loss = float('inf')

    print("Beginning training")
    for epoch in range(args.num_epoch):
        start_time = time.time()
        full_loss, full_test_loss, jac_misfit = 0.0, 0.0, 0.0
        idx = 0
        
        for k, T in dataloader:
            k, T = k.unsqueeze(dim=1).to(device).float(), T.to(device).float()
            
            # MSE 
            optimizer.zero_grad()
            output = model(k)
            print("output", output)
            loss = criterion(output.squeeze(), T) / torch.norm(T)

            # GM
            if args.loss_type == "JAC":
                target = True_j[idx].cuda()
                output, vjp_func = torch.func.vjp(model, k)
                print("vjp", output)
                vjp_out = vjp_func(vec_batch.unsqueeze(dim=1))[0].squeeze()
                jac_diff = criterion(target, vjp_out)
                jac_misfit += jac_diff.detach().cpu().numpy()
                loss += jac_diff * args.reg_param # / torch.norm(target)
                print("jac_diff", jac_diff)

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
            for k, T in test_dataloader:
                k, T = k.to(device).float(), T.to(device).float()
                output = model(k.unsqueeze(dim=1))
                test_loss = criterion(output.squeeze(), T) / torch.norm(T)
                full_test_loss += test_loss.item()
            test_diff.append(full_test_loss)
        model.train()
        
        print(f"Epoch: {epoch}, Train Loss: {full_loss:.6f}, JAC misfit: {jac_misfit}, Test Loss: {full_test_loss:.6f}")
        if epoch % 50 == 0:
            torch.save(model.state_dict(), f"../test_result/Checkpoint/FNO_Heat_{loss_type}_{args.nx}_{args.num_train}_{epoch}.pth")
        if full_test_loss < lowest_loss:
            print("saved lowest loss model")
            lowest_loss = full_test_loss
            torch.save(model.state_dict(), f"../test_result/best_model_FNO_Heat_{loss_type}.pth")
            # Save plot
            k, T = next(iter(test_dataloader))
            k, T = k.unsqueeze(dim=1).to(device).float(), T.to(device).float()
            with torch.no_grad():
                T_pred = model(k)
            plot_path = f"../plot/Heat_plot/FNO_Heat_lowest_{loss_type}.png"
            plot_results(k[0], T[0], T_pred[0], plot_path)
                
        if full_loss < args.threshold:
            print("Stopping early as the loss is below the threshold.")
            break

    print("Finished Computing")

    # Save the model
    torch.save(model.state_dict(), f"../test_result/best_model_FNO_Heat_full epoch_{loss_type}.pth")
    # Save the elapsed times
    with open(f'../test_result/Time/FNO_HEAT_{args.loss_type}_{args.nx}_{args.num_train}.csv', 'w', newline='') as csvfile:
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
            with open(f'../test_result/Losses/{name}_{args.loss_type}_{args.nx}_{args.num_train}.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Epoch', 'Loss'])
                writer.writerows(enumerate(data, 1))
    print("Losses saved to CSV files.")

    print("Creating plot...")
    plt.rcParams.update({'font.size': 14})
    k, T = next(iter(test_dataloader))
    k, T = k.unsqueeze(dim=1).to(device).float(), T.to(device).float()
    with torch.no_grad():
        T_pred = model(k)
    plot_path = f"../plot/Heat_plot/FNO_Heat_{loss_type}.png"
    plot_results(k[0], T[0], T_pred[0], plot_path)

    # Create loss plot
    print("Create loss plot")
    mse_diff = np.asarray(mse_diff)
    jac_diff_list = np.asarray(jac_diff_list)
    test_diff = np.asarray(test_diff)
    path = f"../plot/Loss/FNO_Heat_{loss_type}.png"

    fig, ax = plt.subplots()
    ax.plot(mse_diff, "P-", lw=2.0, ms=6.0, color="coral", label="MSE (Train)")
    ax.plot(test_diff, "P-", lw=2.0, ms=6.0, color="indianred", label="MSE (Test)")
    if args.loss_type == "JAC":
        ax.plot(jac_diff_list, "P-", lw=2.0, color="slateblue", ms=6.0, label=r"$\|J^Tv - \hat{J}^Tv\|$")
    ax.set_xlabel("Epochs",fontsize=24)
    ax.xaxis.set_tick_params(labelsize=24)
    ax.yaxis.set_tick_params(labelsize=24)
    ax.legend(fontsize=24)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(path, bbox_inches ='tight', pad_inches = 0.1)


    logger.info("%s: %s", "Loss Type", str(loss_type))
    logger.info("%s: %s", "Batch Size", str(args.batch_size))
    logger.info("%s: %s", "Training Loss", str(full_loss/len(dataloader)))
    logger.info("%s: %s", "Test Loss", str(full_test_loss/len(test_dataloader)))
    logger.info("%s: %s", "MSE diff", str(mse_diff[-1]))

if __name__ == "__main__":
    # Set device
    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.backends.cudnn.benchmark = True
    print("device: ", device)

    # Set arguments (hyperparameters)
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--num_epoch", type=int, default=500)
    parser.add_argument("--num_train", type=int, default=100)
    parser.add_argument("--num_test", type=int, default=100)
    parser.add_argument("--num_sample", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=1e-8)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--loss_type", default="JAC", choices=["MSE", "JAC"])
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--ny", type=int, default=64)
    parser.add_argument("--noise", type=float, default=0.01)
    parser.add_argument("--reg_param", type=float, default=500.0)

    args = parser.parse_args()

    # Save initial settings
    start_time = datetime.datetime.now().strftime("%m_%d_%H_%M_%S")
    out_file = os.path.join("../test_result/", f"FNO_Heat_norm_{start_time}.txt")
    logging.basicConfig(filename=out_file, filemode="w", level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(__name__)
    for arg, value in vars(args).items():
        logger.info("%s: %s", arg, value)

    # Generate Training/Test Data
    # nx, ny = 60 is sinusoidal data
    trainx_file = f'../data/train_x_{args.nx}_{args.ny}_{args.num_train}_stiff.csv'
    trainy_file = f'../data/train_y_{args.nx}_{args.ny}_{args.num_train}_stiff.csv'
    testx_file = f'../data/test_x_{args.nx}_{args.ny}_{args.num_test}_stiff.csv'
    testy_file = f'../data/test_y_{args.nx}_{args.ny}_{args.num_test}_stiff.csv'
    if os.path.exists(trainx_file):
        print("Loading Dataset")
        train_x = HeatDataset(load_dataset_from_csv(trainx_file, args.nx, args.ny))
        train_y = HeatDataset(load_dataset_from_csv(trainy_file, args.nx, args.ny))
        test_x = HeatDataset(load_dataset_from_csv(testx_file, args.nx, args.ny))
        test_y = HeatDataset(load_dataset_from_csv(testy_file, args.nx, args.ny))
    else:
        print("Creating Dataset")
        input, output = generate_dataset(args.num_train + args.num_test, args.nx, args.ny)
        train_x = HeatDataset(input[:args.num_train])
        train_y = HeatDataset(output[:args.num_train])
        test_x = HeatDataset(input[args.num_train:])
        test_y = HeatDataset(output[args.num_train:])
        # Save datasets to CSV files
        save_dataset_to_csv(train_x, trainx_file)
        save_dataset_to_csv(train_y, trainy_file)
        save_dataset_to_csv(test_x, testx_file)
        save_dataset_to_csv(test_y, testy_file)

    
    print(train_x[0], train_x[1], len(train_x)) # 2
    print(train_y[0], train_y[1], len(train_y)) # 2


    # Randomly sample indices for train and test sets
    train_indices = np.random.choice(len(train_x), args.num_train, replace=False)
    test_indices = np.random.choice(len(test_y), args.num_test, replace=False)
    # Create subsets of the datasets
    train_dataset = CustomDataset(train_x, train_y)
    test_dataset = CustomDataset(test_x, test_y)
    train_dataset = Subset(train_dataset, train_indices)
    test_dataset = Subset(test_dataset, test_indices)
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    print("Mini-batch: ", len(train_loader), train_loader.batch_size)

    # compute FIM eigenvector
    if args.loss_type == "JAC":
        nx, ny = args.nx, args.ny
        noise_std = 0.01  # Adjust as needed
        print("Reloaded train: ", train_x[0].shape)
        fim = compute_fim_for_2d_heat(solve_heat_equation, train_x[0].cuda(), train_y[0].cuda(), noise_std, nx, ny).detach().cpu()
        # Compute FIM
        for s in range(args.num_sample - 1):
            print(s)
            # k = torch.exp(torch.randn(nx, ny)).cuda()  # Log-normal distribution for k
            fim += compute_fim_for_2d_heat(solve_heat_equation, train_x[s], train_y[s], noise_std, nx, ny).detach().cpu()
        fim /= args.num_sample

        # Analyze the FIM
        eigenvalues, eigenvec = torch.linalg.eigh(fim.cuda())
        # print("shape", eigenvalues.shape, eigenvec.shape) -> torch.Size([2500]) torch.Size([2500, 2500])
        # Get the eigenvector corresponding to the largest eigenvalue
        # Assuming eigenvalues and eigenvectors are already computed
        # eigenvalues: A tensor of eigenvalues
        # eigenvec: A matrix where each column corresponds to an eigenvector

        # Sort eigenvalues in descending order and get their indices
        sorted_indices = torch.argsort(eigenvalues, descending=True)

        # Iterate over sorted eigenvalues to find the largest one with a non-zero eigenvector
        largest_eigenvector = None
        for idx in sorted_indices:
            print("candidate idx", idx)
            candidate_eigenvector = eigenvec[:, idx]
            
            # Check if the eigenvector is non-zero
            if torch.any(candidate_eigenvector != 0):
                largest_eigenvector = candidate_eigenvector
                break
        # Handle the case where no non-zero eigenvector is found
        if largest_eigenvector is None:
            raise ValueError("No non-zero eigenvector found.")

        # idx = torch.argmax(eigenvalues)
        # largest_eigenvector = eigenvec[:, idx]
        largest_eigenvector = largest_eigenvector.reshape(args.nx, args.ny)

        print("Largest Eigenvalue and index:", eigenvalues[idx], idx)
        print("Corresponding Eigenvector:", largest_eigenvector)
        print("Eigenvector shape", largest_eigenvector.shape)
        print("eigenvalue: ", eigenvalues)
        print("eigenvector: ", eigenvec)
    else:
        largest_eigenvector = None

    # train
    main(logger, args, args.loss_type, train_loader, test_loader, largest_eigenvector)