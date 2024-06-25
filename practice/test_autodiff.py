import torch
import numpy as np
import torch.autograd.functional as F
import torchdiffeq
from matplotlib.pyplot import *
import sys
sys.path.append('..')

from scipy.integrate import odeint
from src import NODE_solve_Lorenz as sol 
from src import NODE_util as util
from examples import Lorenz as func
import test_node_lorenz as test_node



def test_jac_node(x, eps, optim_name, time_step):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = x.to(device)

    # load the saved model
    model = sol.create_NODE(device, n_nodes=3, T=time_step).double()
    path = "expt_lorenz/"+optim_name+"/"+str(time_step)+'/'+'model.pt'
    model.load_state_dict(torch.load(path))
    model.eval()

    # compute the jacobian of neural ode
    jacobian_node = F.jacobian(model, x)

    print("----- JAC_NODE_AD -----")
    jac_rk4_ad = torch.squeeze(jacobian_node)
    jac_rk4_ad = jac_rk4_ad.clone().detach()
    print(jac_rk4_ad)

    # initialize
    t_eval_point = torch.linspace(0, time_step, 2)
    jac_rk4_fd = torch.zeros(3,3).double()

    # Finite Differentiation using Central Difference Approximation
    for i in range(3):
        x_plus = x.clone()
        x_minus = x.clone()

        # create perturbed input
        x_plus[i] = x_plus[i] + eps
        x_minus[i] = x_minus[i] - eps

        # create model output
        m_plus = model(x_plus)
        #print("mp", m_plus)
        m_minus = model(x_minus)

        # compute central diff
        diff = m_plus.clone().detach() - m_minus.clone().detach()
        final = diff/2/eps
        jac_rk4_fd[:, i] = final

    print("jac_rk4_fd\n", jac_rk4_fd)

    print(torch.allclose(jac_rk4_ad.to("cpu"), jac_rk4_fd.to("cpu"), rtol=1e-05))

    diff = jac_rk4_ad.to("cpu") - jac_rk4_fd.to("cpu")
    diff = diff.reshape(3,3)
    #print("diff: ", diff)
    norm = torch.linalg.matrix_norm(diff)

    return norm



def test_autodiff(x, eps, time_step, method):
    deltat = time_step

    # initialize
    t_eval_point = torch.linspace(0, time_step, 2)
    jac_rk4_fd = torch.zeros(3,3).double()
    
    # Automatic Differentiation
    jac_rk4_ad = F.jacobian(lambda x: torchdiffeq.odeint(func.lorenz, x, t_eval_point, method=method), x)[1]
    #jac_rk4_ad = F.jacobian(lambda x: sol.simulate(0, time_step, x, 2), x)[1]
    jac_rk4_ad = torch.squeeze(jac_rk4_ad)
    
    # Finite Differentiation using Central Difference Approximation
    for i in range(3):
        x_plus = x.clone()
        x_minus = x.clone()

        # create perturbed input
        x_plus[i] = x_plus[i] + eps
        x_minus[i] = x_minus[i] # - eps

        # create model output
        m_plus = torchdiffeq.odeint(func.lorenz, x_plus, t_eval_point, method=method)[1]
        m_minus = torchdiffeq.odeint(func.lorenz, x_minus, t_eval_point, method=method)[1]

        # compute central diff
        diff = m_plus.clone().detach() - m_minus.clone().detach()
        final = diff/eps
        jac_rk4_fd[:, i] = final

    print("jac_rk4_fd\n", jac_rk4_fd)

    print(torch.allclose(jac_rk4_ad, jac_rk4_fd, rtol=1e-05))

    diff = jac_rk4_ad - jac_rk4_fd
    diff = diff.reshape(3,3)
    #print("diff: ", diff)
    norm = torch.linalg.matrix_norm(diff)

    return norm


def Jacobian_Matrix(v, sigma, r, b):
    '''Compute Jacobian Matrix'''

    x, y, z = [k for k in v]
    return np.array([[-sigma, sigma, 0], [r - z, -1, -x], [y, x, -b]])


# compute difference in ad_jacobian and fd_jacobian
'''torch.set_printoptions(sci_mode=True, precision=12)

# create random input
x = torch.rand(3).double()
x.requires_grad = True
print("random x", x)

# train the model
optim_name = 'AdamW'
time_step = 5e-4
method = "euler" # RK4, euler, NODE
eps_arr = torch.logspace(start=-7,end=-2,steps=10)
err = torch.zeros(10).double()

if method == "NODE":
    print("## --------------- NODE --------------- ##")
    for i, eps in enumerate(eps_arr):
        err[i] = test_jac_node(x, eps, optim_name, time_step)
else:
    print("## ---------------", method,"--------------- ##")
    for i, eps in enumerate(eps_arr):
        err[i] = test_autodiff(x, eps, time_step, method)

fig, ax = subplots()
ax.semilogx(eps_arr, err, ".", ms = 10.0)
ax.grid(True)
ax.set_xlabel("$\epsilon$", fontsize=20)
ax.set_ylabel("Diff in FD and AD jacobian", fontsize=20)
ax.xaxis.set_tick_params(labelsize=20)
ax.yaxis.set_tick_params(labelsize=20)
tight_layout()
fig.savefig("jac_test_" + method + ".png")
'''
