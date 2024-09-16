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
from PINO_NS import *

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
    input, output= [], []

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
        w_current = w.cuda()
        vorticity_data = [w_current.cpu().numpy()]

        # Solve the NS
        for i in range(num_iter):
            w_current = ns_solver(w_current, f=forcing, T=time_step, Re=Re)
            vorticity_data.append(w_current.cpu().numpy())
        
        input.append(vorticity_data[:-1])
        output.append(vorticity_data[1:])
        
        if s == 0:
            plot_vorticity(vorticity_data[0], s, title=f"Vorticity Field at Time Step {i + 1}")
        elif s == 1:
            plot_vorticity(vorticity_data[0], s, title=f"Vorticity Field at Time Step {i + 1}")

    return input, output



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

def compute_fim_NS(simulator, input, T_data, noise_std, nx, ny, forcing, time_step, Re):
    # Ensure k is a tensor with gradient tracking
    q = input.requires_grad_().cuda()
    fim = torch.zeros((nx*ny, nx*ny))
    
    # # Add noise
    mean = 0.0
    std_dev = 0.1

    # Generate Gaussian noise
    noise = torch.randn(q.size()) * std_dev + mean
    # Solve heat equation
    # w_current = ns_solver(w_current, f=forcing, T=time_step, Re=Re)
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

        #Dimension and Lp-norm type are postive
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
def plot_results(true1, true2, pred1, pred2, path):
    plt.figure(figsize=(15, 10))
    plt.rcParams.update({'font.size': 16})

    plt.subplot(2, 3, 1)
    plt.imshow(true1.cpu().numpy(), cmap='inferno')
    plt.colorbar(fraction=0.045, pad=0.06)
    plt.title('True vorticity')

    plt.subplot(2, 3, 2)
    plt.imshow(pred1.cpu().numpy(), cmap='inferno')
    plt.colorbar(fraction=0.045, pad=0.06)
    plt.title('Predicted vorticity')

    plt.subplot(2, 3, 3)
    plt.imshow(true1.cpu().numpy() - pred1.cpu().numpy(), cmap='viridis')
    plt.colorbar(fraction=0.045, pad=0.06)
    plt.title('Error')

    plt.subplot(2, 3, 4)
    plt.imshow(true2.cpu().numpy(), cmap='inferno')
    plt.colorbar(fraction=0.045, pad=0.06)
    plt.title('True Vorticity')

    plt.subplot(2, 3, 5)
    plt.imshow(pred2.cpu().numpy(), cmap='inferno')
    plt.colorbar(fraction=0.045, pad=0.06)
    plt.title('Predicted Vorticity')

    plt.subplot(2, 3, 6)
    plt.imshow(true2.cpu().numpy() - pred2.cpu().numpy(), cmap='viridis')
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
# shape is the tuple shape of each instance
def sample_uniform_spherical_shell(npoints: int, radii: float, shape: tuple):
    ndim = np.prod(shape)
    inner_radius, outer_radius = radii
    pts = []
    for i in range(npoints):
        # uniformly sample radius
        samp_radius = np.random.uniform(inner_radius, outer_radius)
        vec = np.random.randn(ndim) # ref: https://mathworld.wolfram.com/SpherePointPicking.html
        vec /= np.linalg.norm(vec, axis=0)
        pts.append(np.reshape(samp_radius*vec, shape))

    return np.array(pts)

# Partitions of unity - input is real number, output is in interval [0,1]
"""
norm_of_x: real number input
shift: x-coord of 0.5 point in graph of function
scale: larger numbers make a steeper descent at shift x-coord
"""
def sigmoid_partition_unity(norm_of_x, shift, scale):
    return 1/(1 + torch.exp(scale * (norm_of_x - shift)))

# Dissipative functions - input is point x in state space (practically, subset of R^n)
"""
inputs: input point in state space
scale: real number 0 < scale < 1 that scales down input x
"""
def linear_scale_dissipative_target(inputs, scale):
    return scale * inputs


def main(logger, args, loss_type, dataloader, test_dataloader, vec, simulator):
    # Initialization
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device: ", device)

    model = FNO(
        in_channels=1,  # Adjusted for vx and vy inputs
        out_channels=1, # Adjusted for wz output
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
        csv_filename = f'../data/true_j_NS_{nx}_{args.num_train}.csv'
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
            for batch_idx, (batch_data, batch_labels) in enumerate(dataloader):
                for i in range(batch_data.shape[0]):  # Iterate over each sample in the batch
                    # single sample [nx, ny]
                    x = batch_data[i]
                    output, vjp_tru_func = torch.func.vjp(f, x.cuda())
                    print(batch_idx, i)
                    True_j[batch_idx, i] = vjp_tru_func(vec)[0].detach().cpu()

            # Save True_j to a CSV file
            True_j_flat = True_j.reshape(-1, nx * ny)  # Flatten the last two dimensions
            pd.DataFrame(True_j_flat.numpy()).to_csv(csv_filename, index=False)
            print(f"Data saved to {csv_filename}")
        # Create vec_batch
        True_j = True_j.float()
        vec_batch = vec.unsqueeze(0).repeat(dataloader.batch_size, 1, 1)
        vec_batch = vec_batch.cuda().float()

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
                X_test, Y_test = X_test.cuda().float(), Y_test.cuda().float()
                output = model(X_test.unsqueeze(dim=1))
                test_loss = criterion(output.squeeze(), Y_test) / torch.norm(Y_test)
                full_test_loss += test_loss.item()
            test_diff.append(full_test_loss)
        model.train()
        
        print(f"Epoch: {epoch}, Train Loss: {full_loss:.6f}, JAC misfit: {jac_misfit}, Test Loss: {full_test_loss:.6f}")
        if epoch % 50 == 0:
            torch.save(model.state_dict(), f"../test_result/Checkpoint/FNO_NS_vort_{loss_type}_{args.nx}_{args.num_train}_{epoch}.pth")
        if full_test_loss < lowest_loss:
            print("saved lowest loss model")
            lowest_loss = full_test_loss
            torch.save(model.state_dict(), f"../test_result/best_model_FNO_NS_vort_{loss_type}.pth")
            # Save plot
            X_test, Y_test = next(iter(test_dataloader))
            X_test, Y_test = X_test.cuda().float(), Y_test.cuda().float()
            print("shape", X_test.shape, Y_test.shape)
            with torch.no_grad():
                Y_pred = model(X_test.unsqueeze(dim=1))
            plot_path = f"../plot/NS_plot/FNO_NS_vort_lowest_{loss_type}.png"
            plot_results(Y_test[0].squeeze().cpu(), Y_test[1].squeeze().cpu(), Y_pred[0].squeeze(), Y_pred[1].squeeze(), plot_path)
                
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
    mse_diff = np.asarray(mse_diff)
    jac_diff_list = np.asarray(jac_diff_list)
    test_diff = np.asarray(test_diff)
    path = f"../plot/Loss/FNO_NS_vort_{loss_type}.png"

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
    parser.add_argument("--num_train", type=int, default=1000)
    parser.add_argument("--num_test", type=int, default=200)
    parser.add_argument("--num_sample", type=int, default=1000)
    parser.add_argument("--num_init", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=1e-8)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--loss_type", default="MSE", choices=["MSE", "JAC", "Sobolev", "Dissipative"])
    parser.add_argument("--nx", type=int, default=100)
    parser.add_argument("--ny", type=int, default=100)
    parser.add_argument("--noise", type=float, default=0.01)
    parser.add_argument("--reg_param", type=float, default=50.0)
    parser.add_argument("--nu", type=float, default=0.001) # Viscosity
    parser.add_argument("--time_step", type=float, default=0.01) # time step

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
    trainx_file = f'../data/NS_vort/train_x_{args.nx}_{args.ny}_{args.num_train}.csv'
    trainy_file = f'../data/NS_vort/train_y_{args.nx}_{args.ny}_{args.num_train}.csv'
    testx_file = f'../data/NS_vort/test_x_{args.nx}_{args.ny}_{args.num_test}.csv'
    testy_file = f'../data/NS_vort/test_y_{args.nx}_{args.ny}_{args.num_test}.csv'
    if not os.path.exists(trainx_file):
        print("Creating Dataset")
        input, output = generate_dataset(args.num_train + args.num_test, args.num_init, args.time_step, args.nx, args.ny)
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
    # train_x = NSDataset(train_x_raw)
    # train_y = NSDataset(train_y_raw)
    # test_x = NSDataset(test_x_raw)
    # test_y = NSDataset(test_y_raw)



    # Randomly sample indices for train and test sets
    train_indices = np.random.choice(len(train_x_raw), args.num_train, replace=False)
    test_indices = np.random.choice(len(test_y_raw), args.num_test, replace=False)
    # Create subsets of the datasets
    train_dataset = CustomDataset(train_x_raw, train_y_raw)
    test_dataset = CustomDataset(test_x_raw, test_y_raw)
    train_dataset = Subset(train_dataset, train_indices)
    test_dataset = Subset(test_dataset, test_indices)
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    print("Mini-batch: ", len(train_loader), train_loader.batch_size)

    # compute FIM eigenvector
    if args.loss_type == "JAC":
        nx, ny = args.nx, args.ny
        noise_std = 0.01
        print("Reloaded train: ", train_x_raw[0].shape)
        fim = compute_fim_NS(ns_solver, train_x_raw[0].cuda(), train_y_raw[0].cuda(), noise_std, nx, ny, forcing, args.time_step, Re).detach().cpu()
        # Compute FIM
        for s in range(args.num_sample - 1):
            print(s)
            # k = torch.exp(torch.randn(nx, ny)).cuda()  # Log-normal distribution for k
            fim += compute_fim_NS(ns_solver, train_x_raw[s], train_y_raw[s], noise_std, nx, ny, forcing, args.time_step, Re).detach().cpu()
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
    main(logger, args, args.loss_type, train_loader, test_loader, largest_eigenvector, ns_solver)