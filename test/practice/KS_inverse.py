import torch
import torch.nn as nn
# import torch.autograd.functional as F
import torch.optim as optim
import torchdiffeq
import datetime
import numpy as np
import argparse
import json
import logging
import sys
import os
import csv
import math
from torch.func import vmap, vjp, jacrev
from matplotlib.pyplot import *
from mpl_toolkits.mplot3d import axes3d
import torch.distributions as dist

# mpirun -n 2 python test_....

from torch.utils.data import DataLoader, TensorDataset
from modulus.models.fno import FNO
from modulus.models.mlp.fully_connected import FullyConnected

sys.path.append('..')
from data.KS import *

import pyro
import pyro.distributions as dist
import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.sparse as sp

# Include all the functions you provided (rhs_KS_implicit, rhs_KS_explicit_nl, rhs_KS_explicit_linear, explicit_rk, implicit_rk, run_KS)

class KuramotoSivashinsky(torch.nn.Module):
    def __init__(self, c, alpha, beta, dx, dt, T, device='cuda'):
        super().__init__()
        self.c = torch.tensor(c, device=device)
        self.alpha = torch.tensor(alpha, device=device)
        self.beta = torch.tensor(beta, device=device)
        self.dx = dx
        self.dt = dt
        self.T = T
        self.device = device

    def forward(self, u0):
        u0 = u0.to(self.device)
        return run_KS(u0, self.c, self.dx, self.dt, self.T, self.alpha, self.beta, False, self.device)

def ks_probabilistic(data, verbose=False):
    device = data.device
    c = pyro.sample("c", dist.Uniform(0.5, 2.0).to(device))
    alpha = pyro.sample("alpha", dist.Uniform(0.5, 2.0).to(device))
    beta = pyro.sample("beta", dist.Uniform(0.5, 2.0).to(device))
    
    dx = 0.1  # Adjust as needed
    dt = 0.01  # Adjust as needed
    T = data.shape[0] * dt
    
    model = KuramotoSivashinsky(c, alpha, beta, dx, dt, T, device)
    u0 = data[0]
    
    predicted = model(u0)
    
    noise_scale = pyro.param("noise_scale", torch.tensor(0.1, device=device), constraint=dist.constraints.positive)
    pyro.sample("obs", dist.Normal(predicted, noise_scale).to_event(2), obs=data)
    
    return predicted

# Initialize
nx = 100  # Adjust as needed
dx = 0.1
dt = 0.01
T = 10
nt = int(T / dt)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device", device)

# Generate synthetic data (replace this with your actual data)
true_c, true_alpha, true_beta = 1.0, 1.0, 1.0
init_point = torch.randn(nx, device=device)
true_model = KuramotoSivashinsky(true_c, true_alpha, true_beta, dx, dt, T, device)
true_data = true_model(init_point.double())

# Define guide
guide = pyro.infer.autoguide.AutoDiagonalNormal(ks_probabilistic)

# Setup SVI
optim = pyro.optim.Adam({"lr": 0.01})
svi = pyro.infer.SVI(ks_probabilistic, guide, optim, loss=pyro.infer.Trace_ELBO())

# Training loop
num_iterations = 10000
for i in tqdm.tqdm(range(num_iterations)):
    loss = svi.step(true_data.cuda())
    if i % 100 == 0:
        print(f"Iteration {i}: loss = {loss}")

# Posterior sampling
num_samples = 10000
posterior_predictive = pyro.infer.Predictive(ks_probabilistic, guide=guide, num_samples=num_samples)
posterior_samples = posterior_predictive(true_data)

# Extract samples
c_samples = posterior_samples['c'].cpu().numpy()
alpha_samples = posterior_samples['alpha'].cpu().numpy()
beta_samples = posterior_samples['beta'].cpu().numpy()

# Compute means
c_mean = c_samples.mean()
alpha_mean = alpha_samples.mean()
beta_mean = beta_samples.mean()

print(f"Mean of c: {c_mean}")
print(f"Mean of alpha: {alpha_mean}")
print(f"Mean of beta: {beta_mean}")

# Plotting
plt.figure(figsize=(18, 12))

# Plot for c
plt.subplot(3, 1, 1)
sns.histplot(c_samples, kde=True)
plt.axvline(true_c, color='r', linestyle='--', label='True Value')
plt.title(f"Posterior Distribution of c, mean = {c_mean:.4f}")
plt.xlabel("c")
plt.legend()

# Plot for alpha
plt.subplot(3, 1, 2)
sns.histplot(alpha_samples, kde=True)
plt.axvline(true_alpha, color='r', linestyle='--', label='True Value')
plt.title(f"Posterior Distribution of alpha, mean = {alpha_mean:.4f}")
plt.xlabel("alpha")
plt.legend()

# Plot for beta
plt.subplot(3, 1, 3)
sns.histplot(beta_samples, kde=True)
plt.axvline(true_beta, color='r', linestyle='--', label='True Value')
plt.title(f"Posterior Distribution of beta, mean = {beta_mean:.4f}")
plt.xlabel("beta")
plt.legend()

plt.tight_layout()
plt.savefig("KS_parameter_distributions.png")
plt.close()

# Contour plots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

sns.kdeplot(x=c_samples, y=alpha_samples, ax=ax1, cmap="YlGnBu", shade=True, cbar=True)
ax1.scatter(true_c, true_alpha, color='r', s=100, label='True Value')
ax1.set_title("Joint Posterior Distribution of c and alpha")
ax1.set_xlabel("c")
ax1.set_ylabel("alpha")
ax1.legend()

sns.kdeplot(x=c_samples, y=beta_samples, ax=ax2, cmap="YlGnBu", shade=True, cbar=True)
ax2.scatter(true_c, true_beta, color='r', s=100, label='True Value')
ax2.set_title("Joint Posterior Distribution of c and beta")
ax2.set_xlabel("c")
ax2.set_ylabel("beta")
ax2.legend()

sns.kdeplot(x=alpha_samples, y=beta_samples, ax=ax3, cmap="YlGnBu", shade=True, cbar=True)
ax3.scatter(true_alpha, true_beta, color='r', s=100, label='True Value')
ax3.set_title("Joint Posterior Distribution of alpha and beta")
ax3.set_xlabel("alpha")
ax3.set_ylabel("beta")
ax3.legend()

plt.tight_layout()
plt.savefig("KS_joint_distributions.png")
plt.close()