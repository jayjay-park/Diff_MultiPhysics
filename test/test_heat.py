import torch
import torch.nn as nn
import torch.autograd.functional as F
import torch.optim as optim
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
#     T = torch.zeros((nx, ny))
    
#     for _ in range(num_iterations):
#         T_old = T.clone()
#         T[1:-1, 1:-1] = (
#             k[1:-1, 1:-1] * (T_old[2:, 1:-1] / k[2:, 1:-1] + T_old[:-2, 1:-1] / k[:-2, 1:-1] + 
#                              T_old[1:-1, 2:] / k[1:-1, 2:] + T_old[1:-1, :-2] / k[1:-1, :-2])
#             - dx * dy * q[1:-1, 1:-1]
#         ) / (k[1:-1, 1:-1] * (1/k[2:, 1:-1] + 1/k[:-2, 1:-1] + 1/k[1:-1, 2:] + 1/k[1:-1, :-2]))
        
#         # Boundary conditions (Dirichlet)
#         T[0, :] = T[-1, :] = T[:, 0] = T[:, -1] = 0
    
#     return T

def solve_heat_equation(k, q, nx=50, ny=50, num_iterations=1000):
    dx = dy = 1.0 / (nx - 1)
    T = torch.zeros((nx, ny), device=k.device)  # Initialize with boundary temperature
    T[0, :] = T[-1, :] = T[:, 0] = T[:, -1] = 0
    
    for _ in range(num_iterations):
        T_old = T.clone()
        T[1:-1, 1:-1] = (
            k[1:-1, 1:-1] * (T_old[2:, 1:-1] / k[2:, 1:-1] + T_old[:-2, 1:-1] / k[:-2, 1:-1] + 
                             T_old[1:-1, 2:] / k[1:-1, 2:] + T_old[1:-1, :-2] / k[1:-1, :-2])
            + dx * dy * q[1:-1, 1:-1]  # Changed sign to positive
        ) / (k[1:-1, 1:-1] * (1/k[2:, 1:-1] + 1/k[:-2, 1:-1] + 1/k[1:-1, 2:] + 1/k[1:-1, :-2]))
        
        # Boundary conditions (Dirichlet)
        T[0, :] = T[-1, :] = T[:, 0] = T[:, -1] = 0
    
    return T

### Dataset ###
def create_q_function(nx, ny, noise_level=0.1):
    # Create a grid of x and y coordinates
    x = torch.linspace(0, 1, nx)
    y = torch.linspace(0, 1, ny)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    # Define q(x, y) as per the given function
    q = 3000 * (torch.sin(5*X) * torch.sin(3*Y) + torch.cos(5*Y) * torch.cos(2*X))
    
    # Add noise
    noise = noise_level * torch.randn_like(q)
    q_noisy = q + noise
    
    return q_noisy

def generate_dataset(num_samples, nx=50, ny=50):
    input = []
    output = []
    for s in range(num_samples):
        print(s)
        # Log-normal distribution for k (common in heat transfer problems)
        # k = torch.exp(torch.randn(nx, ny, device=device))
        k = torch.ones(nx, ny)
        # 1. constant heat source
        # q = torch.ones(nx, ny) * 7000
        # 2. q(x, y)=3000 (sin(5x) sin(3y) + cos(5y)cos(2x))
        # q = create_q_function(nx, ny, noise_level=0.5)
        # 3. normal
        q = torch.randn(nx, ny)
        T = solve_heat_equation(k, q)
        # dataset.append([q, T])
        input.append(q)
        output.append(T)
        if s == 0:
            plot_path = f"../plot/Heat_plot/Heat_q_3.png"
            plot_data(k, q, T, plot_path)
        elif s == 1:
            plot_path = f"../plot/Heat_plot/Heat_q_3(1).png"
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
        # csv_filename = f'../data/true_j_{nx}_{ny}_200.csv'
        if os.path.exists(csv_filename):
            # Load True_j
            True_j_flat = pd.read_csv(csv_filename).values
            print("len", True_j_flat.shape, len(dataloader)*dataloader.batch_size*nx*ny)
            True_j = torch.tensor(True_j_flat)[:len(dataloader)*dataloader.batch_size, :].reshape(len(dataloader), dataloader.batch_size, nx, ny)
            print(f"Data loaded from {csv_filename}")
        else:
            True_j = torch.zeros(len(dataloader), dataloader.batch_size, nx, ny)
            # q = torch.ones((nx, ny)) * 100 
            k = torch.ones(nx, ny)
            f = lambda x: solve_heat_equation(k.cuda(), x)
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
    elapsed_time_train = []
    mse_diff = []
    lowest_loss = 10000000

    print("Beginning training")
    for epoch in range(args.num_epoch):
        # start_time = time.time()
        full_loss, full_test_loss, jac_misfit = 0.0, 0.0, 0.0
        idx = 0
        
        for k, T in dataloader:
            k, T = k.unsqueeze(dim=1).to(device).float(), T.to(device).float()
            
            # MSE 
            optimizer.zero_grad()
            output = model(k)
            loss = criterion(output.squeeze(), T) #/ torch.norm(T)

            # GM
            if args.loss_type == "JAC":
                target = True_j[idx].cuda()
                output, vjp_func = torch.func.vjp(model, k)
                vjp_out = vjp_func(vec_batch.unsqueeze(dim=1))[0].squeeze()
                jac_diff = criterion(target, vjp_out)
                jac_misfit += jac_diff.detach().cpu().numpy()
                loss += jac_diff * args.reg_param # / torch.norm(target)
                print("jac_diff", jac_diff)

            loss.backward(retain_graph=True)
            optimizer.step()
            full_loss += loss.item()
            idx += 1
        
        mse_diff.append(full_loss)
        
        # Validation
        model.eval()
        with torch.no_grad():
            for k, T in test_dataloader:
                k, T = k.to(device).float(), T.to(device).float()
                output = model(k.unsqueeze(dim=1))
                test_loss = criterion(output.squeeze(), T)
                full_test_loss += test_loss.item()
        model.train()
        
        print(f"Epoch: {epoch}, Train Loss: {full_loss:.6f}, JAC misfit: {jac_misfit}, Test Loss: {full_test_loss:.6f}")
        
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

    print("Creating plot...")
    plt.rcParams.update({'font.size': 14})
    k, T = next(iter(test_dataloader))
    k, T = k.unsqueeze(dim=1).to(device).float(), T.to(device).float()
    with torch.no_grad():
        T_pred = model(k)
    plot_path = f"../plot/Heat_plot/FNO_Heat_{loss_type}.png"
    plot_results(k[0], T[0], T_pred[0], plot_path)

    print("Create loss plot")
    mse_diff = np.asarray(mse_diff)
    path = f"../plot/Loss/FNO_Heat_{loss_type}.png"

    fig, ax = plt.subplots()
    ax.plot(mse_diff, "P-", lw=2.0, ms=5.0, label="MSE")
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
    parser.add_argument("--num_train", type=int, default=1000)
    parser.add_argument("--num_test", type=int, default=600)
    parser.add_argument("--threshold", type=float, default=1e-8)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--loss_type", default="MSE", choices=["MSE", "JAC"])
    parser.add_argument("--nx", type=int, default=50)
    parser.add_argument("--ny", type=int, default=50)
    parser.add_argument("--noise", type=float, default=0.01)
    parser.add_argument("--reg_param", type=float, default=20.0)
    parser.add_argument("--num_sample", type=int, default=1000)
    args = parser.parse_args()

    # Save initial settings
    start_time = datetime.datetime.now().strftime("%m_%d_%H_%M_%S")
    out_file = os.path.join("../test_result/", f"FNO_Heat_norm_{start_time}.txt")
    logging.basicConfig(filename=out_file, filemode="w", level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(__name__)
    for arg, value in vars(args).items():
        logger.info("%s: %s", arg, value)

    # Generate Training/Test Data
    trainx_file = f'../data/train_x_{args.nx}_{args.ny}_{args.num_train}.csv'
    trainy_file = f'../data/train_y_{args.nx}_{args.ny}_{args.num_train}.csv'
    testx_file = f'../data/test_x_{args.nx}_{args.ny}_{args.num_test}.csv'
    testy_file = f'../data/test_y_{args.nx}_{args.ny}_{args.num_test}.csv'
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
        nx, ny = 50, 50
        noise_std = 0.01  # Adjust as needed
        # train = [torch.stack(data) for data in train_x]  # Assuming `data` can be converted to a tensor
        # train = torch.stack(train)
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
        largest_eigenvector = largest_eigenvector.reshape(nx, ny)

        print("Largest Eigenvalue and index:", eigenvalues[idx], idx)
        print("Corresponding Eigenvector:", largest_eigenvector)
        print("Eigenvector shape", largest_eigenvector.shape)
        print("eigenvalue: ", eigenvalues)
        print("eigenvector: ", eigenvec)
    else:
        largest_eigenvector = None

    # train
    main(logger, args, args.loss_type, train_loader, test_loader, largest_eigenvector)