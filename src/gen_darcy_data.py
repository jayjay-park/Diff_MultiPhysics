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
import h5py
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

# Define a function to plot a single image
def plot_field(field, title, ax):
    ax.imshow(field, cmap='viridis')
    ax.set_title(title)
    ax.axis('off')

def gen_darcy(permeability_mean, permeability_std_dev, darcy_mean, darcy_std_dev, resolution, batch_size, num_train, num_test):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device: ", device)
    print("Creating Dataset")

    normaliser = {
        "permeability": (permeability_mean, permeability_std_dev),
        "darcy": (darcy_mean, darcy_std_dev),
    }
    darcy_dataloader = Darcy2D(
        resolution=resolution,
        batch_size=batch_size,
        normaliser=normaliser,
    )

    dataloader = []
    val_dataloader = []
    all_num = num_train + num_test
    for i in range(all_num):
        instance = next(iter(darcy_dataloader))
        p = instance["permeability"].cpu().detach() #.numpy()
        d = instance["darcy"].cpu().detach()
        if i < num_train:
            print("train", p[0,0])
            dataloader.append({"permeability": p, "darcy": d})
        else:
            print("test", p[0,0])
            val_dataloader.append({"permeability": p, "darcy": d})

        phase_path_train = f"../plot/Phase_plot/FNO_Darcy_data{i}.png"
        # take the first instance for simplicity
        true_instance = p[0, 0].cpu().numpy()
        pred_instance = d[0, 0].cpu().numpy()
        # Plotting the fields
        fig, axs = subplots(1, 2, figsize=(12, 6))
        plot_field(true_instance, 'Permeability', axs[0])
        plot_field(pred_instance, 'Pressure Field', axs[1])
        fig.savefig(phase_path_train, format='png', dpi=400, bbox_inches='tight', pad_inches=0.1)

    print("Mini-batch: ", len(dataloader))
    return dataloader, val_dataloader


if __name__ == '__main__':

    # normalizer
    permeability_mean = 7.4836
    permeability_std_dev = 4.49996
    darcy_mean = 0.000574634
    darcy_std_dev = 0.000388433
    resolution = 64
    batch_size = 4
    num_train = 5000
    num_test = 4000

    dataloader, val_dataloader = gen_darcy(permeability_mean, permeability_std_dev, darcy_mean, darcy_std_dev, resolution, batch_size, num_train, num_test)

    # test
    print("data", dataloader[0]['permeability'][0,0,:5,:5], "\n")
    print("val", val_dataloader[0]['permeability'][0,0,:5,:5], "\n")
    print("data", dataloader[1]['permeability'][0,0,:5,:5], "\n")
    print("val", val_dataloader[1]['permeability'][0,0,:5,:5], "\n")
 
    # Save the data
    path = f"../data/Darcy_train_{resolution}_{batch_size}_{num_train}.h5"
    with h5py.File(path, 'w') as f:
        for i, item in enumerate(dataloader):
            group = f.create_group(f'dict_{i}')
            for key, tensor in item.items():
                group.create_dataset(key, data=tensor.numpy())
    
    test_path = f"../data/Darcy_test_{resolution}_{batch_size}_{num_test}.h5"
    with h5py.File(test_path, 'w') as f:
        for i, item in enumerate(val_dataloader):
            group = f.create_group(f'dict_{i}')
            for key, tensor in item.items():
                group.create_dataset(key, data=tensor.numpy())

    # To load the data back
    loaded_data = []
    with h5py.File(path, 'r') as f:
        for key in f.keys():
            group = f[key]
            loaded_dict = {subkey: torch.tensor(group[subkey]) for subkey in group.keys()}
            loaded_data.append(loaded_dict)
    print(len(loaded_data))
