import pyro
from pyro.distributions import LogNormal, Uniform
import tqdm
import arviz
import pandas as pd
import torch
from stochproc import timeseries as ts, distributions as dists
from torch.utils.data import DataLoader, TensorDataset
from modulus.models.fno import FNO
from modulus.models.mlp.fully_connected import FullyConnected
import matplotlib.pyplot as plt  # Import matplotlib
import seaborn as sns


# initialize
dim=3
dt = 0.01
T = 30
init_point = torch.randn(dim)
loss_type = "MSE"

def f(x, s, r, b):
    x1 = s * (x.value[..., 1] - x.value[..., 0])
    x2 = r * x.value[..., 0] - x.value[..., 1] - x.value[..., 0] * x.value[..., 2]
    x3 = - b * x.value[..., 2] + x.value[..., 0] * x.value[..., 1]

    return torch.stack((x1, x2, x3), dim=-1)

def lorenz_probabilistic(data, verbose=False):
    s = pyro.sample("sigma", Uniform(low=5.0, high=40.0))
    r = pyro.sample("rho", Uniform(low=10.0, high=50.0))
    b = pyro.sample("beta", Uniform(low=1.0, high=20.0))
    
    model = ts.RungeKutta(f, (s, r, b), init_point, dt=dt, event_dim=1, tuning_std=0.01)
    model.do_sample_pyro(pyro, data.shape[0], obs=data)
    return

# call model
model = FNO(
        in_channels=3,
        out_channels=3,
        num_fno_modes=3,
        padding=4,
        dimension=1,
        latent_channels=128).to('cuda')
FNO_path = "../test_result/best_model_FNO_Lorenz_"+str(loss_type)+".pth"
model.load_state_dict(torch.load(FNO_path))
model.eval()

# generate data
torch.cuda.empty_cache()
learned_traj = torch.zeros(T*int(1/dt), 3)
learned_traj[0] = init_point
print(learned_traj.shape)
for i in range(1, len(learned_traj)):
    out = model(learned_traj[i-1].reshape(1, dim, 1).cuda()).reshape(dim,-1)
    learned_traj[i] = out.squeeze().detach().cpu()

guide = pyro.infer.autoguide.AutoDiagonalNormal(lorenz_probabilistic)
optim = pyro.optim.Adam({"lr": 0.01})
svi = pyro.infer.SVI(lorenz_probabilistic, guide, optim, loss=pyro.infer.Trace_ELBO())

niter = 10000
pyro.clear_param_store()

running_average = 0.0
smoothing = 0.99

bar = tqdm.tqdm(range(niter))
for n in bar:
    loss = svi.step(learned_traj)
    
    running_average = smoothing * running_average + (1 - smoothing) * loss
    bar.set_description(f"Loss: {running_average:,.2f}")

num_samples = 1000
posterior_predictive = pyro.infer.Predictive(
    lorenz_probabilistic,
    guide=guide,
    num_samples=num_samples
)

posterior_draws = {k: v.unsqueeze(0) for k, v in posterior_predictive(learned_traj).items()}
print("posterior draws", posterior_draws['beta'].shape)

# compute mean

posteriors = arviz.from_dict(posterior_draws)
print(posteriors)

# Inspect the groups available in the InferenceData object
print(posteriors.groups()) # posterior
# Access a specific dataset, e.g., the posterior
posterior_data = posteriors.posterior
# Now you can inspect the shape of the specific dataset
print("posterior shape", posterior_data)

# Save the plot to a file
# fig = arviz.plot_posterior(posteriors, var_names=["sigma", "rho", "beta"])
fig = arviz.plot_pair(
    posteriors,
    var_names=["sigma", "rho", "beta"],
    kind=["scatter", "kde"],
    kde_kwargs={"fill_last": False},
    marginals=True,
    point_estimate="median",
    figsize=(11.5, 5)
)
plt.savefig("../plot/Post/Lorenz_posterior"+str(loss_type)+".png")  # Save the figure
plt.show()  # Show the plot if running interactively