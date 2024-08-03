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
import math
from torch.func import vmap, vjp
from matplotlib.pyplot import *
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

def FIM_noise(params, S_K, C, setting, init, eta, gamma, delta = 0.001):

    listX = []
    dx, dt, c, n, T = setting
    log_likelihood = lambda simulated: 0.5*torch.norm(simulated - C)**2
    # This jacobian is not differentiated by parameter...
    jacobian = jacrev(log_likelihood, argnums=0)(simulated_C) # 127
    jacobian = jacobian.reshape(-1, 1)
    FIM = torch.mm(jacobian, jacobian.T) 
    # average ...

    return FIM

### Equation ###
def lorenz(t, u, params=[10.0,28.0,8/3]):
    """ Lorenz chaotic differential equation: du/dt = f(t, u)
    t: time T to evaluate system
    u: state vector [x, y, z] 
    return: new state vector in shape of [3]"""

    du = torch.stack([
            params[0] * (u[1] - u[0]),
            u[0] * (params[1] - u[2]) - u[1],
            (u[0] * u[1]) - (params[2] * u[2])
        ])
    return du

class Lorenz63(nn.Module):
    def __init__(self, sigma, rho, beta):
        super().__init__()
        self.sigma = nn.Parameter(torch.tensor(sigma, dtype=torch.float32))
        self.rho = nn.Parameter(torch.tensor(rho, dtype=torch.float32))
        self.beta = nn.Parameter(torch.tensor(beta, dtype=torch.float32))

    def forward(self, t, state):
        x, y, z = state
        dx = self.sigma * (y - x)
        dy = x * (self.rho - z) - y
        dz = x * y - self.beta * z
        return torch.stack([dx, dy, dz])

def log_likelihood(data, model_output, noise_std):
    print(data)
    print(model_output)
    print(noise_std)
    return -0.5 * torch.sum((data - model_output)**2) / (noise_std**2) - \
           data.shape[0] * torch.log(torch.tensor(noise_std))

def compute_fim_wrt_input(model, initial_state, t, data, noise_std):
    # Ensure initial_state is a tensor with gradient tracking
    initial_state = torch.tensor(initial_state, requires_grad=True)
    
    fim = torch.zeros((len(initial_state), len(initial_state)))
    
    y = torchdiffeq.odeint(model, initial_state, t, method='rk4', rtol=1e-8)
    ll = log_likelihood(data, y, noise_std)

    for i in range(len(initial_state)):
        grad_i = torch.autograd.grad(ll, initial_state, create_graph=True)[0][i]
        for j in range(i, len(initial_state)):
            grad_j = torch.autograd.grad(grad_i, initial_state, retain_graph=True)[0][j]
            fim[i, j] = grad_j.item()
            fim[j, i] = fim[i, j]

    return fim

def compute_fim(model, initial_state, t, data, noise_std):
    params = list(model.parameters())
    fim = torch.zeros((len(params), len(params)))
    y = torchdiffeq.odeint(model, initial_state, t, method='rk4', rtol=1e-8)
    ll = log_likelihood(data, y, noise_std)

    for i in range(len(params)):
        grad_i = torch.autograd.grad(ll, params[i], create_graph=True)[0]
        for j in range(i, len(params)):
            grad_j = torch.autograd.grad(grad_i, params[j], retain_graph=True)[0]
            fim[i, j] = grad_j.item()
            fim[j, i] = fim[i, j]

    return fim

### Dataset ###
def create_data(dyn_info, n_train, n_test, n_val, n_trans):
    dyn, dim, time_step = dyn_info
    # Adjust total time to account for the validation set
    tot_time = time_step * (n_train + n_test + n_val + n_trans + 1)
    t_eval_point = torch.arange(0, tot_time, time_step)

    # Generate trajectory using the dynamical system
    # lorenz(t, u, params=[10.0,28.0,8/3])
    traj = torchdiffeq.odeint(dyn, torch.randn(dim), t_eval_point, method='rk4', rtol=1e-8)
    traj = traj[n_trans:]  # Discard transient part

    # Create training dataset
    X_train = traj[:n_train]
    Y_train = traj[1:n_train + 1]
    
    # Shift trajectory for validation dataset
    traj = traj[n_train:]
    X_val = traj[:n_val]
    Y_val = traj[1:n_val + 1]

    # Shift trajectory for test dataset
    traj = traj[n_val:]
    X_test = traj[:n_test]
    Y_test = traj[1:n_test + 1]

    return [X_train, Y_train, X_val, Y_val, X_test, Y_test]

### Compute Metric ###
def rk4(x, f, dt):
    k1 = f(0, x)
    k2 = f(0, x + dt*k1/2)
    k3 = f(0, x + dt*k2/2)
    k4 = f(0, x + dt*k3)
    return x + dt/6*(k1 + 2*k2 + 2*k3 + k4)

def lyap_exps(dyn_sys_info, ds_name, traj, iters, batch_size):
    model, dim, time_step = dyn_sys_info
    LE = torch.zeros(dim).to(device)
    traj_gpu = traj.to(device)
    if model == lorenz:
        f = lambda x: rk4(x, model, time_step)
        Jac = torch.vmap(torch.func.jacrev(f))(traj_gpu)
    else:
        f = model
        # traj_in_batch = traj_gpu.reshape(-1, 1, dim, 1)
        traj_data = TensorDataset(traj_gpu)
        traj_loader = DataLoader(traj_data, batch_size=batch_size, shuffle=False)
        Jac = torch.randn(traj_gpu.shape[0], dim, dim).cuda()
        i = 0

        for traj in traj_loader:

            jac = torch.func.jacrev(model)
            x = traj[0].unsqueeze(dim=2).to('cuda')
            cur_model_J = jac(x)
            squeezed_J = cur_model_J[:, :, 0, :, :, 0]
            learned_J = [squeezed_J[in_out_pair, :, in_out_pair, :] for in_out_pair in range(batch_size)]
            learned_J = torch.stack(learned_J, dim=0).cuda()
            Jac[i:i+batch_size] = learned_J
            i +=batch_size
        print(Jac)

    Q = torch.rand(dim,dim).to(device)
    eye_cuda = torch.eye(dim).to(device)
    for i in range(iters):
        if i > 0 and i % 1000 == 0:
            print("Iteration: ", i, ", LE[0]: ", LE[0].detach().cpu().numpy()/i/time_step)
        Q = torch.matmul(Jac[i], Q)
        Q, R = torch.linalg.qr(Q)
        LE += torch.log(abs(torch.diag(R)))
    return LE/iters/time_step

def model_size(model):
    # Adapted from https://discuss.pytorch.org/t/finding-model-size/130275/11
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.3f}MB'.format(size_all_mb))
    return size_all_mb

def plot_attractor(model, dyn_info, time, path):
    # generate true orbit and learned orbit
    dyn, dim, time_step = dyn_info
    tran_orbit = torchdiffeq.odeint(dyn, torch.randn(dim), torch.arange(0, 5, time_step), method='rk4', rtol=1e-8)
    true_o = torchdiffeq.odeint(dyn, tran_orbit[-1], torch.arange(0, time, time_step), method='rk4', rtol=1e-8)

    learned_o = torch.zeros(time*int(1/time_step), dim)
    x0 = tran_orbit[-1]
    for t in range(time*int(1/time_step)):
        learned_o[t] = x0
        new_x = model(x0.reshape(1, dim, 1).cuda())
        x0 = new_x.squeeze()
    learned_o = learned_o.detach().cpu().numpy()

    # create plot of attractor with initial point starting from 
    fig, axs = subplots(2, 3, figsize=(24,12))
    cmap = cm.plasma
    num_row, num_col = axs.shape

    for x in range(num_row):
        for y in range(num_col):
            orbit = true_o if x == 0 else learned_o
            if y == 0:
                axs[x,y].plot(orbit[0, 0], orbit[0, 1], '+', markersize=35, color=cmap.colors[0])
                axs[x,y].scatter(orbit[:, 0], orbit[:, 1], c=orbit[:, 2], s = 6, cmap='plasma', alpha=0.5)
                axs[x,y].set_xlabel("X")
                axs[x,y].set_ylabel("Y")
            elif y == 1:
                axs[x,y].plot(orbit[0, 0], orbit[0, 2], '+', markersize=35, color=cmap.colors[0])
                axs[x,y].scatter(orbit[:, 0], orbit[:, 2], c=orbit[:, 2], s = 6, cmap='plasma', alpha=0.5)
                axs[x,y].set_xlabel("X")
                axs[x,y].set_ylabel("Z")
            else:
                axs[x,y].plot(orbit[0, 1], orbit[0, 2], '+', markersize=35, color=cmap.colors[0])
                axs[x,y].scatter(orbit[:, 1], orbit[:, 2], c=orbit[:, 2], s = 6, cmap='plasma', alpha=0.5)
                axs[x,y].set_xlabel("Y")
                axs[x,y].set_ylabel("Z")
        
            axs[x,y].tick_params(labelsize=42)
            axs[x,y].xaxis.label.set_size(42)
            axs[x,y].yaxis.label.set_size(42)
    tight_layout()
    fig.savefig(path, format='png', dpi=400, bbox_inches ='tight', pad_inches = 0.1)
    return

def plot_likelihood_contours(model, initial_state, t, data, noise_std, true_params, fim, estimated_params, param_indices=(0, 1)):
    # Define grid
    n_points = 50
    param_range = 0.5  # Range around true parameters to plot
    p1, p2 = param_indices
    p1_range = np.linspace(true_params[p1] - param_range, true_params[p1] + param_range, n_points)
    p2_range = np.linspace(true_params[p2] - param_range, true_params[p2] + param_range, n_points)
    P1, P2 = np.meshgrid(p1_range, p2_range)

    # Compute log-likelihood for each point
    log_liks = np.zeros_like(P1)
    for i in range(n_points):
        for j in range(n_points):
            model.sigma.data = torch.tensor(P1[i, j] if p1 == 0 else true_params[0])
            model.rho.data = torch.tensor(P2[i, j] if p2 == 1 else true_params[1])
            model.beta.data = torch.tensor(P1[i, j] if p1 == 2 else (P2[i, j] if p2 == 2 else true_params[2]))
            
            y = torchdiffeq.odeint(model, initial_state, t, method='rk4')
            log_liks[i, j] = log_likelihood(data, y, noise_std).item()

    # Plot likelihood
    rcParams.update({'font.size': 14})
    figure(figsize=(10, 8))
    contour = contour(P1, P2, log_liks, levels=10)
    colorbar(contour, label='Log-likelihood')
    
    # Plot true parameters
    plot(true_params[p1], true_params[p2], 'r*', markersize=15, label='True parameters')
    
    # Plot estimated parameters
    scatter(estimated_params[:, p1], estimated_params[:, p2], alpha=0.5, s=60, label='Estimated parameters')
    
    # Plot FIM ellipse
    cov = torch.inverse(fim)
    sub_cov = cov[[p1, p2]][:, [p1, p2]]
    eigenvalues, eigenvectors = torch.linalg.eigh(sub_cov)
    
    # Ensure eigenvalues are positive
    eigenvalues = torch.abs(eigenvalues)
    angle = np.degrees(np.arctan2(eigenvectors[1, 0].item(), eigenvectors[0, 0].item()))
    # 95% CI
    width, height = 2 * torch.sqrt(eigenvalues) * np.sqrt(5.991)  # 95% confidence ..?
    ellipse = Ellipse(xy=(true_params[p1], true_params[p2]), width=width.item(), height=height.item(), 
                      angle=angle, facecolor='none', edgecolor='r', linestyle='--', label='95% CI (FIM)')
    gca().add_patch(ellipse)
    
    # Plot eigenvectors
    for i in range(2):
        eigen_vector = eigenvectors[:, i]
        print(eigen_vector[0].item() * torch.sqrt(eigenvalues[i]).item(), 
            eigen_vector[1].item() * torch.sqrt(eigenvalues[i]).item())
        print(eigen_vector, eigenvalues)
        arrow(true_params[p1], true_params[p2], 
                  eigen_vector[0].item() * eigenvalues[i].item() * 5000, 
                  eigen_vector[1].item() * eigenvalues[i].item() * 5000, 
                  color='black', alpha=0.5, width=0.0005, head_width=0.005, 
                  length_includes_head=True, label=f'Eigenvector' if i == 0 else '')

    param_names = ['σ', 'ρ', 'β']
    xlabel(f'${param_names[p1]}$')
    ylabel(f'${param_names[p2]}$')
    title(f'Log-likelihood contours and Fisher Information Ellipse\n{param_names[p1]} vs {param_names[p2]}')
    legend()
    tight_layout()
    fig.savefig('Contour_'+str(param_indices)+'.png')
    return


def main(logger, args, loss_type, dataloader, test_dataloader, cotangent, batch_size):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device: ", device)

    model = FNO(
        in_channels=3,
        out_channels=3,
        num_fno_modes=3,
        padding=4,
        dimension=1,
        latent_channels=64
    ).to('cuda')

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-5
    )

    criterion = torch.nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1e-3)

    ### Training Loop ###
    n_store, k  = 100, 0
    time_step = 0.01
    reg_param = 2.0
    jac_diff_train, jac_diff_test = torch.empty(n_store+1), torch.empty(n_store+1)
    print("Computing analytical Jacobian")
    t = torch.linspace(0, time_step, 2).cuda()
    threshold = 0.005
    f = lambda x: torchdiffeq.odeint(lorenz, x, t, method="rk4")[1]
    torch.cuda.empty_cache()
    timer = Timer()
    elapsed_time_train = []
    jac_diff = []
    mse_diff = []
    test_loss_store = []
    lowest_loss = 1000000

    if loss_type == "JAC":
        # len_train = len(dataloader) * dataloader.batch_size
        cotangent_batch = cotangent.view(1, 3, 1).repeat(batch_size, 1, 1).cuda()
        print(cotangent_batch.shape, cotangent_batch[-1], cotangent)
        True_j = torch.zeros(train_list[0].shape[0], 3)
        for j in range(train_list[0].shape[0]):
            x = train_list[0][j]
            output, vjp_tru_func = vjp(f, x)
            True_j[j] = vjp_tru_func(cotangent)[0]
        True_J = True_j.reshape(len(dataloader), dataloader.batch_size, 3).cuda()
    
    print("Beginning training")
    for epoch in range(args.num_epoch):
        start_time = time.time()
        full_loss, full_test_loss = 0.0, 0.0
        idx = 0
        mse = 0.
        jac = 0.
        for data in dataloader:
            optimizer.zero_grad()
            y_true = data[1].to('cuda')
            y_pred = model(data[0].unsqueeze(dim=2).to('cuda'))

            # MSE Loss
            loss_mse = criterion(y_pred.view(batch_size, -1), y_true.view(batch_size, -1))
            loss = loss_mse / torch.norm(y_true, p=2)
            mse += loss.detach().cpu().numpy()
            
            if loss_type == "JAC":
                with timer:
                    x = data[0].unsqueeze(dim=2).to('cuda')
                    output, vjp_func = vjp(model, x)
                    vjp_out = vjp_func(cotangent_batch)[0].squeeze()

                    jac_norm_diff = criterion(True_J[idx], vjp_out)
                    jac += jac_norm_diff.detach().cpu().numpy()
                    loss += (jac_norm_diff / torch.norm(True_J[idx]))*reg_param
    
            full_loss += loss
            idx += 1
            end_time = time.time()  
            optimizer.step()
            
        mse_diff.append(mse)
        jac_diff.append(jac)
        full_loss.backward(retain_graph=True)
        optimizer.step()
        
        for test_data in test_dataloader:
            y_test_true = test_data[1].to('cuda')
            y_test_pred = model(test_data[0].unsqueeze(dim=2).to('cuda'))
            test_loss = criterion(y_test_pred.view(batch_size, -1), y_test_true.view(batch_size, -1))
            full_test_loss += test_loss.detach().cpu().numpy()

        test_loss_store.append(full_test_loss)
        print("epoch: ", epoch, "loss: ", full_loss.item(), "test loss: ", full_test_loss.item())

        if full_test_loss < threshold:
            print("Stopping early as the loss is below the threshold.")
            break
        
        if full_test_loss < lowest_loss:
            print("saved lowest loss model")
            lowest_loss = full_test_loss
            torch.save(model.state_dict(), f"../test_result/best_model_FNO_Lorenz_{loss_type}.pth")

    if loss_type == "JAC":
        with open('../test_result/Time/Modulus_FNO_elapsed_times_Jacobian.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Epoch', 'Elapsed Time (seconds)'])
            for epoch, elapsed_time in enumerate(timer.elapsed_times, 1):
                writer.writerow([epoch, elapsed_time])
    with open('../test_result/Time/Modulus_FNO_epoch_times.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Epoch', 'Elapsed Time (seconds)'])
        for epoch, elapsed_time in enumerate(elapsed_time_train, 1):
            writer.writerow([epoch, elapsed_time])


    print("Creating plot...")
    phase_path = f"../plot/Phase_plot/FNO_Lorenz_{loss_type}.png"
    plot_attractor(model, [lorenz, 3, 0.01], 50, phase_path)

    print("Create loss plot")
    jac_diff = np.asarray(jac_diff)
    mse_diff = np.asarray(mse_diff)
    test_loss_store = np.asarray(test_loss_store)
    path = f"../plot/Loss/FNO_Lorenz_{loss_type}.png"
    test_path = f"../plot/Loss/FNO_Lorenz_test_{loss_type}.png"

    fig, ax = subplots()
    if loss_type == "JAC":
        ax.plot(jac_diff[10:], "P-", lw=2.0, ms=5.0, label=r"$\|J^Tv - \hat{J}^Tv\|$")
    ax.plot(mse_diff[10:], "P-", lw=2.0, ms=5.0, label="MSE")
    ax.set_xlabel("Epochs",fontsize=24)
    ax.xaxis.set_tick_params(labelsize=24)
    ax.yaxis.set_tick_params(labelsize=24)
    ax.legend(fontsize=24)
    ax.grid(True)
    tight_layout()
    fig.savefig(path, bbox_inches ='tight', pad_inches = 0.1)
    close()

    fig_test, axt = subplots()
    axt.plot(test_loss_store[10:], "P-", lw=2.0, ms=5.0, label="MSE")
    axt.set_xlabel("Epochs",fontsize=24)
    axt.xaxis.set_tick_params(labelsize=24)
    axt.yaxis.set_tick_params(labelsize=24)
    axt.legend(fontsize=24)
    axt.grid(True)
    tight_layout()
    fig_test.savefig(test_path, bbox_inches ='tight', pad_inches = 0.1)
    close()

    # compute LE
    torch.cuda.empty_cache()
    dim = 3
    init = torch.randn(dim)
    true_traj = torchdiffeq.odeint(lorenz, torch.randn(dim), torch.arange(0, 50, 0.01), method='rk4', rtol=1e-8)

    init_point = torch.randn(dim)
    learned_traj = torch.empty_like(true_traj).cuda()
    learned_traj[0] = init_point
    print(learned_traj.shape)
    for i in range(1, len(learned_traj)):
        out = model(learned_traj[i-1].reshape(1, dim, 1).cuda()).reshape(dim,-1)
        print(out)
        learned_traj[i] = out.squeeze()

    print("shape", learned_traj.shape)
    
    # print("Computing LEs of NN...")
    # learned_LE = lyap_exps([model, dim, 0.01], "lorenz", learned_traj, true_traj.shape[0], batch_size).detach().cpu().numpy()
    # print("Computing true LEs...")
    # True_LE = lyap_exps([lorenz, dim, 0.01], "lorenz", true_traj, true_traj.shape[0], batch_size).detach().cpu().numpy()

    print("Computing rest of metrics...")
    True_mean = torch.mean(true_traj, dim = 0)
    Learned_mean = torch.mean(learned_traj, dim = 0)
    True_var = torch.var(true_traj, dim = 0)
    Learned_var = torch.var(learned_traj, dim=0)

    logger.info("%s: %s", "Loss Type", str(loss_type))
    logger.info("%s: %s", "Batch Size", str(batch_size))
    logger.info("%s: %s", "Training Loss", str(full_loss))
    logger.info("%s: %s", "Test Loss", str(full_test_loss))
    # logger.info("%s: %s", "Learned LE", str(learned_LE))
    # logger.info("%s: %s", "True LE", str(True_LE))
    logger.info("%s: %s", "Learned mean", str(Learned_mean))
    logger.info("%s: %s", "True mean", str(True_mean))
    logger.info("%s: %s", "MSE diff", str(mse_diff[-1]))
    logger.info("%s: %s", "JAC diff", str(jac_diff[-1]))



if __name__ == "__main__":

    # Set device
    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.backends.cudnn.benchmark = True
    print("device: ", device)

    # Set arguments (hyperparameters)
    DYNSYS_MAP = {'Lorenz': [lorenz, 3]}

    parser = argparse.ArgumentParser()
    parser.add_argument("--time_step", type=float, default=0.01) #0.25
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--num_epoch", type=int, default=2000)
    # parser.add_argument("--integration_time", type=int, default=0) #100
    parser.add_argument("--num_train", type=int, default=4000) #3000
    parser.add_argument("--num_test", type=int, default=1000) #3000
    parser.add_argument("--num_trans", type=int, default=200)
    parser.add_argument("--iters", type=int, default=6000)
    parser.add_argument("--threshold", type=float, default=0.)
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--loss_type", default="JAC", choices=["JAC", "MSE"])
    parser.add_argument("--reg_param", type=float, default=0.5) 
    parser.add_argument("--num_init", type=int, default=5)
    parser.add_argument("--c", type=float, default=0.)
    parser.add_argument("--noise", type=float, default=0.01)
    parser.add_argument("--dim", type=int, default=3, choices = [127, 200, 1024]) 
    parser.add_argument("--T", type=int, default=201, choices = [5, 11, 51, 101, 201, 301, 501, 1001, 1501])
    parser.add_argument("--optim_name", default="AdamW", choices=["AdamW", "Adam", "RMSprop", "SGD"])
    parser.add_argument("--cotangent", default="FIM", choices=["ones", "rand", "QR", "FIM", "score"])
    parser.add_argument("--dyn_sys", default="Lorenz", choices=DYNSYS_MAP.keys())

    args = parser.parse_args()
    true_params = [10.0, 28.0, 8/3]
    dyn_sys_func = Lorenz63(*true_params)
    dim = args.dim
    dyn_sys_info = [dyn_sys_func, args.dyn_sys, dim]

    # Save initial settings
    start_time = datetime.datetime.now().strftime("%m_%d_%H_%M_%S")
    out_file = os.path.join("../test_result/", f"FNO_Lorenz_{start_time}.txt")
    logging.basicConfig(filename=out_file, filemode="w", level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(__name__)
    for arg, value in vars(args).items():
        logger.info("%s: %s", arg, value)

    # Generate Training/Test Data
    print("Creating Dataset")
    num_init = args.num_init
    train_x, train_y, test_x, test_y = [], [], [], []
    for i in range(num_init):

        n_train= int(args.num_train/num_init)
        n_test = int(args.num_test/num_init)
        n_trans= args.num_trans
        time_step = 0.01
        dim = 3
        initial_state = torch.randn(dim)

        # Adjust total time, Generate traj
        tot_time = time_step * (n_train + n_test + n_trans + 1)
        t_eval_point = torch.arange(0, tot_time, time_step)
        
        true_model = Lorenz63(*true_params)
        traj = torchdiffeq.odeint(true_model, initial_state, t_eval_point, method='rk4', rtol=1e-8)

        # Create training dataset
        org_traj = traj[n_trans:]
        X_train = org_traj[:n_train]
        Y_train = org_traj[1:n_train + 1]

        # Shift trajectory for test dataset
        updated_traj = org_traj[n_train:]
        X_test = updated_traj[:n_test]
        Y_test = updated_traj[1:n_test + 1]
        dataset = [X_train, Y_train, X_test, Y_test]
        print("test", X_test, Y_test)
        print("dataset", dataset[0].shape)

        # Add noise
        noise = torch.normal(mean=0.0, std=args.noise, size=(3,))
        train_x.append(dataset[0]* (1 + noise))
        train_y.append(dataset[1]* (1 + noise))
        test_x.append(dataset[2]* (1 + noise))
        test_y.append(dataset[3]* (1 + noise))
        print("Added noise")

    train_list = [torch.stack(train_x).reshape(-1, 3).float(), torch.stack(train_y).reshape(-1, 3).float()]
    test_list = [torch.stack(test_x).reshape(-1, 3).float(), torch.stack(test_y).reshape(-1, 3).float()]
    print("train shape: ", train_list[0].shape)

    # compute FIM of train_x
    t = torch.linspace(0, 10, 1000)
    fim = compute_fim(true_model, initial_state, t, train_list[0][:t.shape[0]], args.noise)
    # Note that the eigenvalues and eigenvectors can be complex even if the input matrix has real values. If you are sure that your matrix will have real eigenvalues and eigenvectors, you can use torch.linalg.eigvals to get only the eigenvalues, or torch.linalg.eigh for Hermitian (symmetric if real) matrices, which guarantees real eigenvalues and orthogonal eigenvectors:
    eigenvalues, eigenvectors = torch.linalg.eigh(fim)
    print("fim", fim)
    print("Eigenvalues:", eigenvalues)
    print("Eigenvectors:", eigenvectors)

    train_data = TensorDataset(*train_list)
    test_data = TensorDataset(*test_list)
    dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    print("Mini-batch: ", len(dataloader), dataloader.batch_size)

    # train
    main(logger, args, args.loss_type, dataloader, test_dataloader, eigenvectors[0], batch_size=args.batch_size)