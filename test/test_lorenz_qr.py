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

# mpirun -n 2 python test_....

from torch.utils.data import DataLoader, TensorDataset
from modulus.models.fno import FNO
from modulus.models.mlp.fully_connected import FullyConnected
# from modulus.launch.logging import LaunchLogger
# from modulus.launch.utils.checkpoint import save_checkpoint

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

def main(logger, loss_type):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device: ", device)

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

    ### Dataset ###
    def create_data(dyn_info, n_train, n_test, n_val, n_trans):
        dyn, dim, time_step = dyn_info
        # Adjust total time to account for the validation set
        tot_time = time_step * (n_train + n_test + n_val + n_trans + 1)
        t_eval_point = torch.arange(0, tot_time, time_step)

        # Generate trajectory using the dynamical system
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

    print("Creating Dataset")
    n_train = 4000
    n_test = 3000
    batch_size = 200
    dim = 3
    r = 3
    dataset = create_data([lorenz, 3, 0.01], n_train=n_train, n_test=n_test, n_val=0, n_trans=0)
    train_list = [dataset[0], dataset[1]]
    val_list = [dataset[2], dataset[3]]
    test_list = [dataset[4], dataset[5]]

    train_data = TensorDataset(*train_list)
    test_data = TensorDataset(*test_list)
    dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    print("Mini-batch: ", len(dataloader), dataloader.batch_size)

    model = FNO(
        in_channels=3,
        out_channels=3,
        num_fno_modes=4,
        padding=5,
        dimension=1,
        latent_channels=128
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
    num_epochs = 5000
    time_step = 0.01
    reg_param = 2.0
    jac_diff_train, jac_diff_test = torch.empty(n_store+1), torch.empty(n_store+1)
    print("Computing analytical Jacobian")
    t = torch.linspace(0, time_step, 2).cuda()
    threshold = 0.00005
    f = lambda x: torchdiffeq.odeint(lorenz, x, t, method="rk4")[1]
    torch.cuda.empty_cache()
    timer = Timer()
    elapsed_time_train = []
    jac_diff = []
    mse_diff = []

    x = data[0].unsqueeze(dim=2).to('cuda')
    output, vjp_func = vjp(model, x)
    Q_list_summed = Q_list[idx].sum(dim=-1, keepdim=True) #[200, 3, 1]
    vjp_out = vjp_func(Q_list_summed)[0] #[200, 3, 1]


    if loss_type == "JAC":


        # randm = torch.eye(*(dim, r))
        traj, vjp_lorenz = vjp(lorenz, 0., x)
        True_J = vjp_lorenz(randm)[1]


        True_j = torch.zeros(n_train, dim, r)
        Q_list = torch.zeros(n_train, dim, r)
        for j in range(n_train):
            # create random matrix by n x r
            randm = torch.randn(dim, r)
            approx = torch.mm(True_J_full[j].T, randm)
            Q, R = torch.linalg.qr(approx)
            True_j[j] = torch.mm(True_J_full[j].T, Q)
            Q_list[j] = Q
        True_J = True_j.reshape(len(dataloader), dataloader.batch_size, dim, r).cuda()
        Q_list = True_j.reshape(len(dataloader), dataloader.batch_size, dim, r).cuda()

        print("Sanity Check: \n", True_j[0], True_j[batch_size], True_j[2*batch_size], True_j[3*batch_size])
        print("True: ", True_J[0:4, 0])
    
    print("Beginning training")
    for epoch in range(num_epochs):
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
                    Q_list_summed = Q_list[idx].sum(dim=-1, keepdim=True) #[200, 3, 1]
                    vjp_out = vjp_func(Q_list_summed)[0] #[200, 3, 1]

                    jac_norm_diff = criterion(True_J[idx], vjp_out)
                    jac += jac_norm_diff.detach().cpu().numpy()
                    loss += (jac_norm_diff / torch.norm(True_J[idx]))*reg_param
    
            full_loss += loss
            idx += 1
            end_time = time.time()  
            elapsed_time_train.append(end_time - start_time)
            
        mse_diff.append(mse)
        jac_diff.append(jac)
        print(mse, jac)
        full_loss.backward()
        optimizer.step()
        
        for test_data in test_dataloader:
            y_test_true = test_data[1].to('cuda')
            y_test_pred = model(test_data[0].unsqueeze(dim=2).to('cuda'))
            test_loss = criterion(y_test_pred.view(batch_size, -1), y_test_true.view(batch_size, -1))
            full_test_loss += test_loss
        
        print("epoch: ", epoch, "loss: ", full_loss.item(), "test loss: ", full_test_loss.item())

        if full_loss < threshold:
            print("Stopping early as the loss is below the threshold.")
            break
        

    print("Finished Computing")
    model_size = model_size(model)
    # Save the model
    torch.save(model.state_dict(), f"../test_result/best_model_FNO_{loss_type}.pth")

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
    phase_path = f"../plot/Phase_plot/FNO_Modulus_{loss_type}.png"
    plot_attractor(model, [lorenz, 3, 0.01], 50, phase_path)

    print("Create loss plot")
    jac_diff = np.asarray(jac_diff)
    print(jac_diff.shape)
    mse_diff = np.asarray(mse_diff)
    path = f"../plot/Loss/FNO_Modulus_{loss_type}.png"

    fig, ax = subplots()
    ax.plot(jac_diff[10:], "P-", lw=2.0, ms=5.0, label=r"$\|J^Tv - \hat{J}^Tv\|$")
    ax.plot(mse_diff[10:], "P-", lw=2.0, ms=5.0, label="MSE")
    ax.set_xlabel("Epochs",fontsize=24)
    ax.xaxis.set_tick_params(labelsize=24)
    ax.yaxis.set_tick_params(labelsize=24)
    ax.legend(fontsize=24)
    ax.grid(True)
    tight_layout()
    savefig(path, bbox_inches ='tight', pad_inches = 0.1)

    logger.info("%s: %s", "Model Size", str(model_size))
    logger.info("%s: %s", "Loss Type", str(loss_type))
    logger.info("%s: %s", "Batch Size", str(batch_size))
    logger.info("%s: %s", "Training Loss", str(full_loss))

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
    
    print("Computing LEs of NN...")
    learned_LE = lyap_exps([model, dim, 0.01], "lorenz", learned_traj, true_traj.shape[0], batch_size).detach().cpu().numpy()
    print("Computing true LEs...")
    True_LE = lyap_exps([lorenz, dim, 0.01], "lorenz", true_traj, true_traj.shape[0], batch_size).detach().cpu().numpy()

    print("Computing rest of metrics...")
    True_mean = torch.mean(true_traj, dim = 0)
    Learned_mean = torch.mean(learned_traj, dim = 0)
    True_var = torch.var(true_traj, dim = 0)
    Learned_var = torch.var(learned_traj, dim=0)


    logger.info("%s: %s", "Learned LE", str(learned_LE))
    logger.info("%s: %s", "True LE", str(True_LE))
    logger.info("%s: %s", "Learned mean", str(Learned_mean))
    logger.info("%s: %s", "True mean", str(True_mean))
    logger.info("%s: %s", "MSE diff", str(mse_diff[-1]))
    logger.info("%s: %s", "JAC diff", str(jac_diff[-1]))
if __name__ == "__main__":

    start_time = datetime.datetime.now().strftime("%m_%d_%H_%M_%S")
    out_file = os.path.join("../test_result/", f"FNO_JAC_{start_time}.txt")
    logging.basicConfig(filename=out_file, level=logging.INFO, format="%(message)s")
    logger = logging.getLogger()

    # call main
    main(logger, "JAC")
    # MSE | epoch:  1599 loss:  4.928406633553095e-05 test loss:  0.0015732439933344722
    # JAC | epoch:  2724 loss:  4.7205154260154814e-05 test loss:  0.00018358735542278737
