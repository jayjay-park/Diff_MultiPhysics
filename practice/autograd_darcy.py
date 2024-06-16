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
from modulus.datapipes.benchmarks.darcy import Darcy2D

'''
Check if Darcy2D generator is differentiable
'''

# Create an instance of the Darcy2D class
darcy = Darcy2D()
# Generate a batch of data (assuming __iter__ yields a batch)
batch_data = next(iter(darcy))
# print("batch_data", batch_data, batch_data.is_cuda)

# Assuming batch_data is a dictionary containing 'permeability' and 'darcy' tensors
# Get a tensor from the batch_data for which you want to compute gradients
tensor_to_differentiate = batch_data['permeability']
print("tensor", tensor_to_differentiate.is_cuda)

# Perform some operations on the tensor
# For example, compute a simple mean
output_tensor = torch.norm(tensor_to_differentiate)**2
print("output", output_tensor)

# Compute gradients
output_tensor.backward()

# Check if gradients have been computed
print(output_tensor.grad is not None)  # Should print True if gradients are computed
