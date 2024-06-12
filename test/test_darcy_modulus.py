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
import sys
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

sys.path.append('..')
from src.gen_darcy_data import *


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


def main(logger, loss_type):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device: ", device)

    ### Dataset ###
    resolution = 64
    batch_size = 4
    num_train = 1000
    num_test = 800
    # normalizer
    permeability_mean = 7.4836
    permeability_std_dev = 4.49996
    darcy_mean = 0.000574634
    darcy_std_dev = 0.000388433

    # data file
    path = f"../data/Darcy_train_{resolution}_{batch_size}_{num_train}.h5"
    test_path = f"../data/Darcy_test_{resolution}_{batch_size}_{num_test}.h5"

    if not (os.path.exists(path) or os.path.exists(test_path)):
        dataloader, val_dataloader = gen_darcy(permeability_mean, permeability_std_dev, darcy_mean, darcy_std_dev, resolution, batch_size, num_train, num_test)

    # To load the data back
    dataloader, val_dataloader = [], []
    with h5py.File(path, 'r') as f:
        for key in f.keys():
            group = f[key]
            loaded_dict = {subkey: torch.tensor(group[subkey]) for subkey in group.keys()}
            dataloader.append(loaded_dict)
    with h5py.File(test_path, 'r') as f:
        for key in f.keys():
            group = f[key]
            test_dict = {subkey: torch.tensor(group[subkey]) for subkey in group.keys()}
            val_dataloader.append(test_dict)
    print('Length of train: ', len(dataloader), 'Length of test: ', len(val_dataloader))

    # Check
    print("data", dataloader[0]['permeability'][0,0,:5,:5], "\n")
    print("val", val_dataloader[0]['permeability'][0,0,:5,:5], "\n")
    print("data", dataloader[1]['permeability'][0,0,:5,:5], "\n")
    print("val", val_dataloader[1]['permeability'][0,0,:5,:5], "\n")

    #### Model #####
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

        if epoch % 50 == 0:
            for test_data in val_dataloader:
                y_test_true = test_data['darcy'].to('cuda')
                x_test = test_data['permeability'].to('cuda')
                y_test_pred = model(x_test)
                test_loss = criterion(y_test_pred, y_test_true)
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
    phase_path = f"../plot/Phase_plot/FNO_Darcy_{loss_type}.png"
    phase_path_train = f"../plot/Phase_plot/FNO_Darcy_{loss_type}_train.png"
    plot_solution_darcy(val_dataloader, phase_path, darcy_mean, darcy_std_dev)
    plot_solution_darcy(dataloader, phase_path_train, darcy_mean, darcy_std_dev)


    logger.info("%s: %s", "Model Size", str(model_size))
    logger.info("%s: %s", "Loss Type", str(loss_type))
    logger.info("%s: %s", "Batch Size", str(batch_size))
    logger.info("%s: %s", "Training Loss", str(full_loss))

if __name__ == "__main__":

    start_time = datetime.datetime.now().strftime("%m_%d_%H_%M_%S")
    out_file = os.path.join("../test_result/", f"FNO_JAC_{start_time}.txt")
    logging.basicConfig(filename=out_file, level=logging.INFO, format="%(message)s")
    logger = logging.getLogger()

    # call main
    main(logger, "MSE")
