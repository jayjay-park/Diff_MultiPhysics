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
import seaborn as sns
from functorch import vjp, vmap
from torch.utils.data import Subset
from generate_NS_org import *

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

def generate_dataset(simulator, num_samples=500, n_steps=200):
    dataset = []
    for s in range(int(num_samples)):
        print("initial condition :", s)
        # Generate random initial conditions
        freq_x = torch.normal(mean=4.0, std=0.3, size=(1,), device='cuda').item()
        freq_y = torch.normal(mean=2.0, std=0.5, size=(1,), device='cuda').item()
        phase_x = torch.normal(mean=0.0, std=1., size=(1,), device='cuda').item()
        phase_y = torch.normal(mean=0.0, std=1., size=(1,), device='cuda').item()

        # Initial Condition (vortex) using sinusoidal function with random parameters
        vx = -torch.sin(freq_y * torch.pi * simulator.yy + phase_y)
        vy = torch.sin(freq_x * torch.pi * simulator.xx + phase_x)
        vx_init = vx.detach()
        vy_init = vy.detach()

        # Simulate for n_steps
        with torch.no_grad():
          vx_final, vy_final, wz, seq_val = simulate(simulator, vx, vy, n_steps)

        # Store the data
        dataset.append({
            'vx': seq_val[:, 0].cpu().numpy(),
            'vy': seq_val[:, 1].cpu().numpy(),
        })
        torch.cuda.empty_cache()

        if s == 0 or s == 1:
            # Save the plots of vx, vy, input, and output vorticity fields
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
            plt.savefig(f'NS_dataset_{s}.png', dpi=300)
            plt.close()
            torch.cuda.empty_cache()

    return dataset



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


class NavierStokesDataset(Dataset):
    def __init__(self, file_path):
        # Load the dataset from the .npy file
        self.dataset = np.load(file_path, allow_pickle=True)
        
    def __len__(self):
        # Return the number of samples
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # Get the data sample at the given index
        init_vx, init_vy, vx, vy, wz = self.dataset[idx]
        
        # Convert numpy arrays to torch tensors
        init_vx = torch.tensor(init_vx, dtype=torch.float32)
        init_vy = torch.tensor(init_vy, dtype=torch.float32)
        vx = torch.tensor(vx, dtype=torch.float32)
        vy = torch.tensor(vy, dtype=torch.float32)
        wz = torch.tensor(wz, dtype=torch.float32)
        
        return init_vx, init_vy, vx, vy, wz


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

def compute_fim_NS(simulator, q, T_data, noise_std, nx=50, ny=50):
    # Ensure k is a tensor with gradient tracking
    q = q.requires_grad_().cuda()
    vx = q[0]
    vy = q[1]
    print("vx shape", vx.shape)
    fim = torch.zeros((2*nx*ny, 2*nx*ny))
    
    # Add noise
    mean = 0.0
    std_dev = 0.1

    # Generate Gaussian noise
    noise = torch.randn(q.size()) * std_dev + mean
    # Solve heat equation
    out_vx, out_vy = simulator(vx, vy)
    T_pred = torch.stack([out_vx, out_vy])
    T_pred = T_pred + noise.cuda()
    ll = log_likelihood(T_data.cuda(), T_pred.cuda(), noise_std)
    flat_Jacobian = torch.autograd.grad(inputs=q, outputs=ll, create_graph=True)[0].flatten() # 50 by 50 -> [2500]
    print("flatten", flat_Jacobian.shape)
    flat_Jacobian = flat_Jacobian.reshape(1, -1)
    fim = torch.matmul(flat_Jacobian.T, flat_Jacobian)

    return fim

### Compute Metric ###
def plot_results(vx, vy, wz, wz_pred, path):
    plt.figure(figsize=(15, 10))
    plt.rcParams.update({'font.size': 16})

    plt.subplot(2, 3, 1)
    plt.imshow(vx.cpu().numpy(), cmap='inferno')
    plt.colorbar(fraction=0.045, pad=0.06)
    plt.title('True vx')

    plt.subplot(2, 3, 2)
    plt.imshow(wz.cpu().numpy(), cmap='inferno')
    plt.colorbar(fraction=0.045, pad=0.06)
    plt.title('Predicted vx')

    plt.subplot(2, 3, 3)
    plt.imshow(wz.cpu().numpy() - vy.cpu().numpy(), cmap='viridis')
    plt.colorbar(fraction=0.045, pad=0.06)
    plt.title('Error')

    plt.subplot(2, 3, 4)
    plt.imshow(vy.cpu().numpy(), cmap='inferno')
    plt.colorbar(fraction=0.045, pad=0.06)
    plt.title('True vy')

    plt.subplot(2, 3, 5)
    plt.imshow(wz_pred.cpu().numpy(), cmap='inferno')
    plt.colorbar(fraction=0.045, pad=0.06)
    plt.title('Predicted vy')

    plt.subplot(2, 3, 6)
    plt.imshow(vy.cpu().numpy() - wz_pred.cpu().numpy(), cmap='viridis')
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
        in_channels=2,  # Adjusted for vx and vy inputs
        out_channels=2, # Adjusted for wz output
        decoder_layer_size=128,
        num_fno_layers=6,
        num_fno_modes=20,
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
    if args.loss_type == "JAC":
        csv_filename = f'../data/true_j_NS_{nx}_{args.num_train}.csv'
        if os.path.exists(csv_filename):
            # Load True_j
            True_j_flat = pd.read_csv(csv_filename).values
            print("len", True_j_flat.shape, len(dataloader)*2*dataloader.batch_size*nx*ny)
            True_j = torch.tensor(True_j_flat)[:len(dataloader)*dataloader.batch_size*2, :].reshape(len(dataloader), dataloader.batch_size, 2, nx, ny)
            print(f"Data loaded from {csv_filename}")
        else:
            True_j = torch.zeros(len(dataloader), dataloader.batch_size, 2, nx, ny)
            f = lambda vxvy : torch.stack([simulator(vxvy[0], vxvy[1])[0], simulator(vxvy[0], vxvy[1])[1]])
            # Iterate over the DataLoader
            for batch_idx, (batch_data, batch_labels) in enumerate(dataloader):
                for i in range(batch_data.shape[0]):  # Iterate over each sample in the batch
                    # single sample [2, nx, ny]
                    x = batch_data[i].cuda()
                    print("x", x.shape, vec.shape)
                    output, vjp_tru_func = torch.func.vjp(f, x)
                    print(batch_idx, i)
                    True_j[batch_idx, i] = vjp_tru_func(vec)[0].detach().cpu()
                    print(True_j[batch_idx, i])

        # # Save True_j to a CSV file
        # True_j_flat = True_j.reshape(-1, 2, nx * ny)  # Flatten the last two dimensions
        # pd.DataFrame(True_j_flat.numpy()).to_csv(csv_filename, index=False)
        # print(f"Data saved to {csv_filename}")
        # Create vec_batch
        True_j = True_j.float()
        vec_batch = vec.unsqueeze(0).repeat(dataloader.batch_size, 1, 1, 1)
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
            X, Y = X.cuda(), Y.cuda()
            # X, Y = X/torch.norm(X), Y/torch.norm(Y)
            vx_init, vy_init = X[:, 0], X[:, 1]
            vx_out, vy_out = Y[:, 0], Y[:, 1]
            
            # MSE 
            optimizer.zero_grad()
            if args.loss_type == "MSE":
                output = model(X)
                loss = criterion(output.squeeze(), Y.squeeze()) / torch.norm(Y)
            else:
            # GM
                target = True_j[idx].cuda()
                output, vjp_func = torch.func.vjp(model, X)
                loss = criterion(output.squeeze(), Y.squeeze()) / torch.norm(Y)
                vjp_out = vjp_func(vec_batch)[0].squeeze()
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
                X_test, Y_test = X_test.cuda(), Y_test.cuda()
                vx_init_t, vy_init_t = X_test[:, 0], X_test[:, 1]
                vx_out_t, vy_out_t = Y_test[:, 0], Y_test[:, 1]
                output = model(X_test)
                test_loss = criterion(output.squeeze(), Y_test) / torch.norm(Y_test)
                full_test_loss += test_loss.item()
            test_diff.append(full_test_loss)
        model.train()
        
        print(f"Epoch: {epoch}, Train Loss: {full_loss:.6f}, JAC misfit: {jac_misfit}, Test Loss: {full_test_loss:.6f}")
        if epoch % 50 == 0:
            torch.save(model.state_dict(), f"../test_result/Checkpoint/FNO_NS_{loss_type}_{args.nx}_{args.num_train}_{epoch}.pth")
        if full_test_loss < lowest_loss:
            print("saved lowest loss model")
            lowest_loss = full_test_loss
            torch.save(model.state_dict(), f"../test_result/best_model_FNO_NS_{loss_type}.pth")
            # Save plot
            X_test, Y_test = next(iter(test_dataloader))
            X_test, Y_test = X_test.cuda(), Y_test.cuda()
            vx_init_t, vy_init_t = X_test[0, 0], X_test[0, 1]
            vx_out_t, vy_out_t = Y_test[0, 0], Y_test[0, 1]
            with torch.no_grad():
                Y_pred = model(X_test)
            plot_path = f"../plot/NS_plot/FNO_NS_lowest_{loss_type}.png"
            plot_results(vx_out_t, vy_out_t, Y_pred[0, 0], Y_pred[0, 1], plot_path)
                
        if full_loss < args.threshold:
            print("Stopping early as the loss is below the threshold.")
            break

    print("Finished Computing")

    # Save the model
    torch.save(model.state_dict(), f"../test_result/best_model_FNO_NS_full epoch_{loss_type}.pth")
    # Save the elapsed times
    with open(f'../test_result/Time/FNO_NS_{args.loss_type}_{args.nx}_{args.num_train}.csv', 'w', newline='') as csvfile:
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
            with open(f'../test_result/Losses/NS_{name}_{args.loss_type}_{args.nx}_{args.num_train}.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Epoch', 'Loss'])
                writer.writerows(enumerate(data, 1))
    print("Losses saved to CSV files.")

    # Create loss plot
    print("Create loss plot")
    mse_diff = np.asarray(mse_diff)
    jac_diff_list = np.asarray(jac_diff_list)
    test_diff = np.asarray(test_diff)
    path = f"../plot/Loss/FNO_NS_{loss_type}.png"

    fig, ax = plt.subplots()
    ax.plot(mse_diff, "P-", lw=2.0, ms=6.0, color="coral", label="MSE (Train)")
    ax.plot(test_diff, "P-", lw=2.0, ms=6.0, color="indianred", label="MSE (Test)")
    if args.loss_type == "JAC":
        ax.plot(jac_diff_list, "P-", lw=2.0, color="slateblue", ms=6.0, label=r"$\|J^Tv - \hat{J}^Tv\|$")
    ax.set_xlabel("Epochs",fontsize=24)
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
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--num_epoch", type=int, default=1000)
    parser.add_argument("--num_train", type=int, default=2000)
    parser.add_argument("--num_test", type=int, default=400)
    parser.add_argument("--num_sample", type=int, default=2000)
    parser.add_argument("--threshold", type=float, default=1e-8)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--loss_type", default="JAC", choices=["MSE", "JAC"])
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--ny", type=int, default=64)
    parser.add_argument("--noise", type=float, default=0.01)
    parser.add_argument("--reg_param", type=float, default=50.0)
    parser.add_argument("--nu", type=float, default=0.001) # Viscosity
    parser.add_argument("--dt", type=float, default=0.001) # time step

    args = parser.parse_args()

    # Save initial settings
    start_time = datetime.datetime.now().strftime("%m_%d_%H_%M_%S")
    out_file = os.path.join("../test_result/", f"FNO_NS_{start_time}.txt")
    logging.basicConfig(filename=out_file, filemode="w", level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(__name__)
    for arg, value in vars(args).items():
        logger.info("%s: %s", arg, value)

    # Initialize
    N = args.nx  # Grid size
    L = 1.0  # Domain length
    dt = args.dt
    n_steps = int(torch.ceil(torch.tensor((0.3 / dt) + 1)).item())  # Number of time steps to simulate
    num_samples = (args.num_train + args.num_test)/n_steps  # Number of samples of initial condition to generate
    simulator = NavierStokesSimulator(N, L, dt, args.nu).cuda()
    datafile = f'../data/NS_{args.nx}_{args.ny}_{args.num_train}_{args.num_test}_NS1.npy'

    if os.path.exists(datafile):
        print("Loading Dataset")
        dataset = NavierStokesDataset(datafile)
    else:
        print("Creating Dataset")

        # Generate dataset
        dataset = generate_dataset(simulator, num_samples, n_steps)

        # Save the dataset
        np.save(datafile, dataset)

        print(f"Dataset generated with {num_samples} samples of initial condition.")
        print(f"Each sample contains:")
        print(f"  - input_vx: shape {dataset[0]['vx'].shape}")
        print(f"  - input_vy: shape {dataset[0]['vy'].shape}")



    # Load the dataset
    dataset = np.load(datafile, allow_pickle=True)
    print(dataset[0])
    inputs, outputs = [], []

    for num in range(int(num_samples)):
        one_series = dataset[num]
        # Extract input and output data
        one_series_vx = torch.tensor(one_series['vx'], dtype=torch.float32) # 200, 40, 40
        one_series_vy = torch.tensor(one_series['vy'], dtype=torch.float32)
        one_series_fields = torch.stack([one_series_vx, one_series_vy]).reshape(-1, 2, args.nx, args.ny)
        one_series_input = one_series_fields[:-1]
        one_series_output = one_series_fields[1:]
        print("one_series_output shape", one_series_output.shape)
        inputs.append(one_series_input)
        outputs.append(one_series_output)

    inputs = torch.stack(inputs).reshape(-1, 2, args.nx, args.ny)
    outputs = torch.stack(outputs).reshape(-1, 2, args.nx, args.ny)
    print(inputs.shape, outputs.shape)
    inputs_train = inputs[:args.num_train]
    outputs_train = outputs[:args.num_train]
    inputs_test = inputs[args.num_train:args.num_train + args.num_test]
    outputs_test = outputs[args.num_train:args.num_train + args.num_test]

    # Create TensorDatasets
    train_dataset = TensorDataset(inputs_train, outputs_train)
    test_dataset = TensorDataset(inputs_test, outputs_test)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Testing samples: {len(test_dataset)}")
    print(f"Input shape: {inputs.shape[1:]}")
    print(f"Output shape: {outputs.shape[1:]}")
    print("Mini-batch: ", len(train_loader), train_loader.batch_size)

    # compute FIM eigenvector
    if args.loss_type == "JAC":
        nx, ny = args.nx, args.ny
        noise_std = 0.01
        # fim = compute_fim_for_2d_heat(simulator, inputs_train[0].cuda(), outputs_train[0].cuda(), noise_std, nx, ny).detach().cpu()
        fim = compute_fim_NS(simulator, inputs_train[0].cuda(), outputs_train[0].cuda(), noise_std, nx=50, ny=50).detach().cpu()
        # Compute FIM
        for s in range(args.num_sample - 1):
            print("FIM: ", s)
            # k = torch.exp(torch.randn(nx, ny)).cuda()  # Log-normal distribution for k
            fim += compute_fim_NS(simulator, inputs_train[s], outputs_train[s], noise_std, nx, ny).detach().cpu()
        fim /= args.num_sample

        # Analyze the FIM
        eigenvalues, eigenvec = torch.linalg.eigh(fim.cuda())

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
        largest_eigenvector = largest_eigenvector.reshape(2, args.nx, args.ny)

        print("Largest Eigenvalue and index:", eigenvalues[idx], idx)
        print("Corresponding Eigenvector:", largest_eigenvector)
        print("Eigenvector shape", largest_eigenvector.shape)
        print("eigenvalue: ", eigenvalues)
        print("eigenvector: ", eigenvec)
    else:
        largest_eigenvector = None

    # train
    main(logger, args, args.loss_type, train_loader, test_loader, largest_eigenvector, simulator)