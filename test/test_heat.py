import torch
import torch.nn as nn
import torch.autograd.functional as F
import torch.optim as optim
import torchdiffeq
import datetime
import numpy as np
import argparse
import json
import logging
import os
import csv
import pandas as pd
import math
from torch.func import vmap, vjp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import seaborn as sns

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
def solve_heat_equation(k, q, nx=50, ny=50, num_iterations=1000):
    dx = dy = 1.0 / (nx - 1)
    T = torch.zeros((nx, ny), device=device)
    
    for _ in range(num_iterations):
        T_old = T.clone()
        T[1:-1, 1:-1] = (
            k[1:-1, 1:-1] * (T_old[2:, 1:-1] / k[2:, 1:-1] + T_old[:-2, 1:-1] / k[:-2, 1:-1] + 
                             T_old[1:-1, 2:] / k[1:-1, 2:] + T_old[1:-1, :-2] / k[1:-1, :-2])
            - dx * dy * q[1:-1, 1:-1]
        ) / (k[1:-1, 1:-1] * (1/k[2:, 1:-1] + 1/k[:-2, 1:-1] + 1/k[1:-1, 2:] + 1/k[1:-1, :-2]))
        
        # Boundary conditions (Dirichlet)
        T[0, :] = T[-1, :] = T[:, 0] = T[:, -1] = 0
    
    return T

### Dataset ###
def generate_dataset(num_samples, nx=50, ny=50):
    dataset = []
    for _ in range(num_samples):
        # Log-normal distribution for k (common in heat transfer problems)
        k = torch.exp(torch.randn(nx, ny, device=device))
        q = torch.ones((nx, ny), device=device) * 100  # Constant heat source term
        T = solve_heat_equation(k, q)
        dataset.append((k, T))
    return dataset

class HeatDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


### Compute Metric ###
def compute_mse(model, dataloader):
    mse = 0.0
    with torch.no_grad():
        for k, T in dataloader:
            k, T = k.to(device), T.to(device)
            output = model(k)
            mse += F.mse_loss(output, T).item()
    return mse / len(dataloader)

def plot_results(k, T_true, T_pred, path):
    plt.rcParams.update({'font.size': 14})
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    im0 = axes[0].imshow(k.squeeze().cpu(), cmap='viridis')
    axes[0].set_title("Input: Log-Thermal Conductivity (k)")
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

def main(logger, args, loss_type, dataloader, test_dataloader):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device: ", device)

    model = FNO(
        in_channels=1,
        out_channels=1,
        num_fno_modes=7,
        padding=3,
        dimension=2,
        latent_channels=32
    ).to('cuda')


    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-5
    )

    criterion = torch.nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1e-3)

    ### Training Loop ###
    # timer = Timer()
    elapsed_time_train = []
    mse_diff = []

    print("Beginning training")
    for epoch in range(args.num_epoch):
        # start_time = time.time()
        full_loss, full_test_loss = 0.0, 0.0
        
        for k, T in dataloader:
            k, T = k.unsqueeze(dim=1).to(device).float(), T.to(device).float()
            
            optimizer.zero_grad()
            output = model(k)
            loss = criterion(output, T)
            
            loss.backward()
            optimizer.step()
            
            full_loss += loss.item()
        
        mse_diff.append(full_loss / len(dataloader))
        
        # Validation
        model.eval()
        with torch.no_grad():
            for k, T in test_dataloader:
                k, T = k.to(device).float(), T.to(device).float()
                output = model(k.unsqueeze(dim=1))
                test_loss = criterion(output, T)
                full_test_loss += test_loss.item()
        model.train()
        
        print(f"Epoch: {epoch}, Train Loss: {full_loss/len(dataloader):.6f}, Test Loss: {full_test_loss/len(test_dataloader):.6f}")
        
        if full_loss < args.threshold:
            print("Stopping early as the loss is below the threshold.")
            break

    print("Finished Computing")

    # Save the model
    torch.save(model.state_dict(), f"../test_result/best_model_FNO_Heat_{loss_type}.pth")

    print("Creating plot...")
    plt.rcParams.update({'font.size': 14})
    k, T = next(iter(test_dataloader))
    print("k", k)
    print("T", T)
    k, T = k.unsqueeze(dim=1).to(device).float(), T.to(device).float()
    with torch.no_grad():
        T_pred = model(k)
    plot_path = f"../plot/Heat_plot/FNO_Heat_{loss_type}.png"
    plot_results(k[0], T[0], T_pred[0], plot_path)

    print("Create loss plot")
    mse_diff = np.asarray(mse_diff)
    path = f"../plot/Loss/FNO_Heat_{loss_type}.png"

    fig, ax = plt.subplots()
    ax.plot(mse_diff[10:], "P-", lw=2.0, ms=5.0, label="MSE")
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
    parser.add_argument("--num_epoch", type=int, default=5000)
    parser.add_argument("--num_train", type=int, default=1000)
    parser.add_argument("--num_test", type=int, default=200)
    parser.add_argument("--threshold", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--loss_type", default="MSE", choices=["MSE"])
    parser.add_argument("--nx", type=int, default=50)
    parser.add_argument("--ny", type=int, default=50)
    parser.add_argument("--noise", type=float, default=0.01)

    args = parser.parse_args()

    # Save initial settings
    start_time = datetime.datetime.now().strftime("%m_%d_%H_%M_%S")
    out_file = os.path.join("../test_result/", f"FNO_Heat_{start_time}.txt")
    logging.basicConfig(filename=out_file, filemode="w", level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(__name__)
    for arg, value in vars(args).items():
        logger.info("%s: %s", arg, value)

    # Generate Training/Test Data
    print("Creating Dataset")
    # dataset = generate_dataset(args.num_train + args.num_test, args.nx, args.ny)
    # train_dataset = HeatDataset(dataset[:args.num_train])
    # test_dataset = HeatDataset(dataset[args.num_train:])


    def save_dataset_to_csv(dataset, prefix):
        k_data = []
        T_data = []
        
        for k, T in dataset:
            k_data.append(k.flatten().tolist())
            T_data.append(T.flatten().tolist())
        
        k_df = pd.DataFrame(k_data)
        T_df = pd.DataFrame(T_data)
        
        k_df.to_csv(f'{prefix}_k.csv', index=False)
        T_df.to_csv(f'{prefix}_T.csv', index=False)
        
        print(f"Saved {prefix} dataset to CSV files")
    
    def load_dataset_from_csv(prefix, nx, ny):
        k_df = pd.read_csv(f'{prefix}_k.csv')
        T_df = pd.read_csv(f'{prefix}_T.csv')
        
        k_data = [torch.tensor(row.values).reshape(nx, ny) for _, row in k_df.iterrows()]
        T_data = [torch.tensor(row.values).reshape(nx, ny) for _, row in T_df.iterrows()]
        
        return list(zip(k_data, T_data))

    train_dataset = HeatDataset(load_dataset_from_csv('../data/train', args.nx, args.ny))
    test_dataset = HeatDataset(load_dataset_from_csv('../data/test', args.nx, args.ny))

    # Save datasets to CSV files
    # save_dataset_to_csv(train_dataset, '../data/train')
    # save_dataset_to_csv(test_dataset, '../data/test')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    print("Mini-batch: ", len(train_loader), train_loader.batch_size)

    # train
    main(logger, args, args.loss_type, train_loader, test_loader)

    # def load_dataset_from_csv(prefix, nx, ny):
    # k_df = pd.read_csv(f'{prefix}_k.csv')
    # T_df = pd.read_csv(f'{prefix}_T.csv')
    
    # k_data = [torch.tensor(row.values).reshape(nx, ny) for _, row in k_df.iterrows()]
    # T_data = [torch.tensor(row.values).reshape(nx, ny) for _, row in T_df.iterrows()]
    
    # return list(zip(k_data, T_data))



    # train_dataset = HeatDataset(load_dataset_from_csv('../data/train', args.nx, args.ny))
    # test_dataset = HeatDataset(load_dataset_from_csv('../data/test', args.nx, args.ny))