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
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import axes3d
from neuralop.datasets import load_darcy_flow_small
from torch.utils.data import DataLoader, Subset

# mpirun -n 2 python test_....
# pip install warp-lang

from torch.utils.data import DataLoader, TensorDataset
from modulus.models.fno import FNO
from modulus.models.mlp.fully_connected import FullyConnected
from modulus.datapipes.benchmarks.darcy import Darcy2D
# from modulus.launch.utils.checkpoint import save_checkpoint

from torch import FloatTensor
from modulus.launch.logging import LaunchLogger


class LimitedDarcy2DIterator:
    def __init__(self, darcy2d_instance, num_batches):
        """
        Wrapper to limit the number of batches produced by Darcy2D instance.
        
        Parameters
        ----------
        darcy2d_instance : Darcy2D
            An instance of the Darcy2D data generator.
        num_batches : int
            The number of batches to generate.
        """
        self.darcy2d_instance = darcy2d_instance
        self.num_batches = num_batches

    def __iter__(self):
        """
        Yields
        ------
        Iterator[Tuple[Tensor, Tensor]]
            Limited iterator that returns a specified number of batches of (permeability, darcy pressure)
            fields of size [batch, resolution, resolution].
        """
        self._counter = 0
        self._darcy_iter = iter(self.darcy2d_instance)
        return self

    def __next__(self):
        if self._counter < self.num_batches:
            self._counter += 1
            return next(self._darcy_iter)
        else:
            raise StopIteration

    def __len__(self):
        return self.num_batches

class GridValidator:
    '''
    Adapted from https://github.com/NVIDIA/modulus/blob/main/examples/cfd/darcy_fno/validator.py
    '''
    """Grid Validator

    The validator compares model output and target, inverts normalisation and plots a sample

    Parameters
    ----------
    loss_fun : MSELoss
        loss function for assessing validation error
    norm : Dict, optional
        mean and standard deviation for each channel to normalise input and target
    font_size : float, optional
        font size used in figures

    """

    def __init__(
        self,
        loss_fun,
        norm: dict = {"permeability": (0.0, 1.0), "darcy": (0.0, 1.0)},
        font_size: float = 28.0,
    ):
        self.norm = norm
        self.criterion = loss_fun
        self.font_size = font_size
        self.headers = ("permeability", "truth", "prediction", "relative error")

    def compare(
        self,
        invar: FloatTensor,
        target: FloatTensor,
        prediction: FloatTensor,
        step: int,
        logger: LaunchLogger,
    ) -> float:
        """compares model output, target and plots everything

        Parameters
        ----------
        invar : FloatTensor
            input to model
        target : FloatTensor
            ground truth
        prediction : FloatTensor
            model output
        step : int
            iteration counter
        logger : LaunchLogger
            logger to which figure is passed

        Returns
        -------
        float
            validation error
        """
        loss = self.criterion(prediction, target)
        norm = self.norm

        # pick first sample from batch
        invar = invar * norm["permeability"][1] + norm["permeability"][0]
        target = target * norm["darcy"][1] + norm["darcy"][0]
        prediction = prediction * norm["darcy"][1] + norm["darcy"][0]
        invar = invar.cpu().numpy()[0, -1, :, :]
        target = target.cpu().numpy()[0, 0, :, :]
        prediction = prediction.detach().cpu().numpy()[0, 0, :, :]

        plt.close("all")
        plt.rcParams.update({"font.size": self.font_size})
        fig, ax = plt.subplots(1, 4, figsize=(15 * 4, 15), sharey=True)
        im = []
        im.append(ax[0].imshow(invar))
        im.append(ax[1].imshow(target))
        im.append(ax[2].imshow(prediction))
        im.append(ax[3].imshow((prediction - target) / norm["darcy"][1]))

        for ii in range(len(im)):
            fig.colorbar(im[ii], ax=ax[ii], location="bottom", fraction=0.046, pad=0.04)
            ax[ii].set_title(self.headers[ii])

        logger.log_figure(figure=fig, artifact_file=f"validation_step_{step:03d}.png")

        return loss

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
                # print("shape", traj)
                jac = torch.func.jacrev(f)
                x = traj[0].unsqueeze(dim=2).to('cuda')
                batchsize = x.shape[0]
                cur_model_J = jac(x)
                squeezed_J = cur_model_J[:, :, 0, :, :, 0]
                non_zero_indices = torch.nonzero(squeezed_J)
                non_zero_values = squeezed_J[non_zero_indices[:, 0], non_zero_indices[:, 1], non_zero_indices[:, 2], non_zero_indices[:, 3]]
                learned_J = non_zero_values.reshape(batchsize, 3, 3)
                Jac[i:i+batchsize] = learned_J
                i +=batchsize
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

    def plot_solution(test_loaders, data_processor, path):
        rcParams.update({'font.size': 12})

        test_samples = test_loaders[32].dataset
        fig = figure(figsize=(10, 10), constrained_layout=True)
        gs = gridspec.GridSpec(3, 8, figure=fig, width_ratios=[1, 0.05, 1, 0.05, 1, 0.05, 1, 0.05], height_ratios=[1, 1, 1])

        for index in range(3):
            data = test_samples[index]
            x = data['x'] # input x: [1, 3, 3]
            y = data['y'] # Ground-truth
            out = model(x.unsqueeze(0).to('cuda')) # Model prediction

            # Plot Input x
            ax = fig.add_subplot(gs[index, 0])
            cax = ax.imshow(x[0], cmap='gray')
            fig.colorbar(cax, ax=ax, fraction=0.05, pad=0.04)
            if index == 0: 
                ax.set_title('Input x')
            ax.set_xticks([])
            ax.set_yticks([])

            # Plot Ground-truth y
            ax = fig.add_subplot(gs[index, 2])
            cax = ax.imshow(y.squeeze())
            fig.colorbar(cax, ax=ax, fraction=0.05, pad=0.04)
            if index == 0: 
                ax.set_title('Ground-truth y')
            ax.set_xticks([])
            ax.set_yticks([])

            # Plot Model prediction
            ax = fig.add_subplot(gs[index, 4])
            cax = ax.imshow(out.squeeze().detach().cpu().numpy())
            fig.colorbar(cax, ax=ax, fraction=0.05, pad=0.04)
            if index == 0: 
                ax.set_title('Model prediction')
            ax.set_xticks([])
            ax.set_yticks([])

            # Plot Difference
            ax = fig.add_subplot(gs[index, 6])
            diff = abs(y.squeeze() - out.squeeze().detach().cpu().numpy())
            print(torch.tensor(diff).shape)
            cax = ax.imshow(diff, cmap='gray_r')
            fig.colorbar(cax, ax=ax, fraction=0.05, pad=0.04)
            if index == 0: 
                ax.set_title('Difference')
            ax.set_xticks([])
            ax.set_yticks([])

        fig.suptitle('Inputs, ground-truth, prediction, and difference')
        fig.savefig(path, format='png', dpi=400, bbox_inches='tight', pad_inches=0.1)
        return

    def plot_solution_darcy(test_loaders, path,
     mean_darcy, std_darcy):
        rcParams.update({'font.size': 12})

        print(len(test_loaders)) # 200
        
        fig = figure(figsize=(10, 10), constrained_layout=True)
        gs = gridspec.GridSpec(3, 8, figure=fig, width_ratios=[1, 0.05, 1, 0.05, 1, 0.05, 1, 0.05], height_ratios=[1, 1, 1])

        for index in range(3):

            # data = test_loaders[index]
            sample = test_loaders[index]
            x_orig = sample['permeability']
            x = sample['permeability'][index, 0, :, :] # [32, 1, 64, 64]
            print("x:", x)
            y = sample['darcy'][index, 0, :, :] # Ground-truth
            out = model(x_orig.to('cuda')) # Model prediction
            out = out[index, 0, :, :]

            # Plot Input x
            ax = fig.add_subplot(gs[index, 0])
            cax = ax.imshow(x.squeeze().detach().cpu().numpy(), cmap='gray')
            fig.colorbar(cax, ax=ax, fraction=0.05, pad=0.04)
            if index == 0: 
                ax.set_title('Input: Permeability')
            ax.set_xticks([])
            ax.set_yticks([])

            # Plot Ground-truth y
            ax = fig.add_subplot(gs[index, 2])
            cax = ax.imshow(y.squeeze().detach().cpu().numpy())
            fig.colorbar(cax, ax=ax, fraction=0.05, pad=0.04)
            if index == 0: 
                ax.set_title('True Pressure Field')
            ax.set_xticks([])
            ax.set_yticks([])

            # Plot Model prediction
            ax = fig.add_subplot(gs[index, 4])
            cax = ax.imshow(out.squeeze().detach().cpu().numpy())
            fig.colorbar(cax, ax=ax, fraction=0.05, pad=0.04)
            if index == 0: 
                ax.set_title('Predicted Pressure Field')
            ax.set_xticks([])
            ax.set_yticks([])

            # Plot Difference
            ax = fig.add_subplot(gs[index, 6])
            diff = y.squeeze().detach().cpu().numpy() - out.squeeze().detach().cpu().numpy()
            cax = ax.imshow(diff, cmap='gray')
            fig.colorbar(cax, ax=ax, fraction=0.05, pad=0.04)
            if index == 0: 
                ax.set_title('Difference')
            ax.set_xticks([])
            ax.set_yticks([])

        print("path", path)
        fig.suptitle('Inputs, ground-truth, prediction, and difference')
        fig.savefig(path, format='png', dpi=400, bbox_inches='tight', pad_inches=0.1)
        close(fig)
        return


    print("Creating Dataset")
    # dataloader, val_dataloader, data_processor = load_darcy_flow_small(
    #     n_train=800, batch_size=50,
    #     test_resolutions=[16, 32], n_tests=[50, 50], test_batch_sizes=[4, 2],
    #     )

    # train_dataset = dataloader.dataset

    # input_keys = [Key("coeff", scale=(7.48360e00, 4.49996e00))]
    # output_keys = [Key("sol", scale=(5.74634e-03, 3.88433e-03))]

    # normalizer
    # permeability_mean = 1.25
    # permeability_std_dev = 0.75
    # darcy_mean = 0.0452
    # darcy_std_dev = 0.0279
    permeability_mean = 7.4836
    permeability_std_dev = 4.49996
    darcy_mean = 0.000574634
    darcy_std_dev = 0.000388433

    # training
    resolution = 64
    batch_size = 4
    rec_results_freq = 8
    max_pseudo_epochs=256
    pseudo_epoch_sample_size=2048
    num_train = 5
    num_test = 5

    normaliser = {
        "permeability": (permeability_mean, permeability_std_dev),
        "darcy": (darcy_mean, darcy_std_dev),
    }
    darcy_dataloader = Darcy2D(
        resolution=resolution,
        batch_size=batch_size,
        normaliser=normaliser,
    )

    # Define a function to plot a single image
    def plot_field(field, title, ax):
        ax.imshow(field, cmap='viridis')
        ax.set_title(title)
        ax.axis('off')

    # for i in range(5):
    #     # Get the next batch
    #     batch = next(iter(darcy_dataloader))
    #     true = batch["darcy"]
    #     pred = batch["permeability"]
    #     phase_path_train = f"../plot/Phase_plot/FNO_Darcy_{loss_type}_{i}.png"

    #     # Assuming batch size is 64, take the first instance for simplicity
    #     true_instance = true[0, 0].cpu().numpy()
    #     pred_instance = pred[0, 0].cpu().numpy()

    #     # Plotting the fields
    #     fig, axs = subplots(1, 2, figsize=(12, 6))
    #     plot_field(true_instance, 'Darcy Pressure Field', axs[0])
    #     plot_field(pred_instance, 'Permeability Field', axs[1])
    #     fig.savefig(phase_path_train, format='png', dpi=400, bbox_inches='tight', pad_inches=0.1)

    all_dataloader = []
    dataloader = []
    val_dataloader = []
    all_num = num_train + num_test
    for i in range(all_num):
        instance = next(iter(darcy_dataloader))
        p = instance["permeability"].detach()
        d = instance["darcy"].detach()
        if i < num_train:
            print("train", p[0,0])
            dataloader.append({"permeability": p, "darcy": d})
        else:
            print("test", p[0,0])
            val_dataloader.append({"permeability": p, "darcy": d})

        phase_path_train = f"../plot/Phase_plot/FNO_Darcy_{loss_type}_data{i}.png"
        # take the first instance for simplicity
        true_instance = p[0, 0].cpu().numpy()
        pred_instance = d[0, 0].cpu().numpy()
        # Plotting the fields
        fig, axs = subplots(1, 2, figsize=(12, 6))
        plot_field(true_instance, 'Pemeability', axs[0])
        plot_field(pred_instance, 'Pressure Field', axs[1])
        fig.savefig(phase_path_train, format='png', dpi=400, bbox_inches='tight', pad_inches=0.1)

    # val_dataloader = all_dataloader[num_train:]
    # dataloader = all_dataloader[:num_train]

    print("val", val_dataloader[0])
    print("data", dataloader[0])

    print("Mini-batch: ", len(dataloader))

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
    n_store, k  = 100, 0
    num_epochs = 500
    time_step = 0.01
    jac_diff_train, jac_diff_test = torch.empty(n_store+1), torch.empty(n_store+1)

    t = torch.linspace(0, time_step, 2).cuda()
    threshold = 0.
    f = lambda x: torchdiffeq.odeint(lorenz, x, t, method="rk4")[1]
    torch.cuda.empty_cache()
    timer = Timer()
    elapsed_time_train = []

    if loss_type == "JAC":
        print("Computing analytical Jacobian")
        True_j = torch.zeros(n_train, 3)
        for j in range(n_train):
            x = train_list[0][j]
            cotangent = torch.ones_like(x)
            output, vjp_tru_func = vjp(f, x)
            res = vjp_tru_func(cotangent)[0]
            True_j[j] = res
        True_J = True_j.reshape(len(dataloader), dataloader.batch_size, 3).cuda()

        print("Sanity Check: \n", True_j[0], True_j[batch_size], True_j[2*batch_size], True_j[3*batch_size])
        print("True: ", True_J[0:4, 0])
    
    print("Beginning training")
    for epoch in range(num_epochs):
        start_time = time.time()
        full_loss, full_test_loss = 0.0, 0.0
        idx = 0
        for data in dataloader:
            # x: [4, 1, 16, 16] y: [4, 1, 16, 16]
            # x: [64, 1, 256, 256] y: [64, 1, 256, 256]
            x = data['permeability'].to('cuda')
            y_true = data['darcy'].to('cuda')
            optimizer.zero_grad()
            y_pred = model(x)

            # MSE Loss
            loss_mse = criterion(y_pred, y_true)
            loss = loss_mse / torch.norm(y_true, p=2)
            
            if loss_type == "JAC":
                with timer:
                    x = data[0].unsqueeze(dim=2).to('cuda')
                    output, vjp_func = vjp(model, x)

                    cotangent = torch.ones_like(x)
                    vjp_out = vjp_func(cotangent)[0].squeeze()

                    jac_norm_diff = criterion(True_J[idx], vjp_out)
                    reg_param = 500.0
                    loss += (jac_norm_diff / torch.norm(True_J[idx]))*reg_param
                        
            full_loss += loss
            idx += 1
            end_time = time.time()  
            elapsed_time_train.append(end_time - start_time)
            
        rel_err = torch.norm(y_pred - y_true) / torch.norm(y_true)
        print(epoch, "relative error:", rel_err)
        full_loss.backward()
        optimizer.step()

        # for test_data in val_dataloader[16]:
        #     y_test_true = test_data['y'].to('cuda')
        #     x_test = test_data['x'].to('cuda')
        #     y_test_pred = model(x_test)
        #     test_loss = criterion(y_test_pred, y_test_true)
        #     full_test_loss += test_loss
        
        # print("epoch: ", epoch, "loss: ", full_loss.item(), "test loss: ", full_test_loss.item())

        if full_loss < threshold:
            print("Stopping early as the loss is below the threshold.")
            break

    # validation step
    # total_loss = 0.0
    # validation_iters = 10
    # for _, batch in zip(range(validation_iters), dataloader):
    #     val_loss = validator.compare(
    #         batch["permeability"],
    #         batch["darcy"],
    #         forward_eval(batch["permeability"]),
    #         pseudo_epoch,
    #         logger,
    #     )
    #     total_loss += val_loss
    # print("validation error:", total_loss / validation_iters)

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
    phase_path = f"../plot/Phase_plot/FNO_Darcy_{loss_type}.png"
    phase_path_train = f"../plot/Phase_plot/FNO_Darcy_{loss_type}_train.png"
    # plot_solution(val_dataloader, data_processor, phase_path)
    plot_solution_darcy(val_dataloader, phase_path, darcy_mean, darcy_std_dev)
    plot_solution_darcy(dataloader, phase_path_train, darcy_mean, darcy_std_dev)

    # compute LE

    print("Computing rest of metrics...")
    True_mean = torch.mean(true_traj, dim = 0)
    Learned_mean = torch.mean(learned_traj, dim = 0)
    True_var = torch.var(true_traj, dim = 0)
    Learned_var = torch.var(learned_traj, dim=0)

    logger.info("%s: %s", "Model Size", str(model_size))
    logger.info("%s: %s", "Loss Type", str(loss_type))
    logger.info("%s: %s", "Batch Size", str(batch_size))
    logger.info("%s: %s", "Training Loss", str(full_loss))
    logger.info("%s: %s", "Learned LE", str(learned_LE))
    logger.info("%s: %s", "True LE", str(True_LE))
    logger.info("%s: %s", "Learned mean", str(Learned_mean))
    logger.info("%s: %s", "True mean", str(True_mean))

if __name__ == "__main__":

    start_time = datetime.datetime.now().strftime("%m_%d_%H_%M_%S")
    out_file = os.path.join("../test_result/", f"FNO_JAC_{start_time}.txt")
    logging.basicConfig(filename=out_file, level=logging.INFO, format="%(message)s")
    logger = logging.getLogger()

    # call main
    main(logger, "MSE")
