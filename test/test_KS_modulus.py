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

# def create_data(traj, n_train, n_test, n_nodes, n_trans):
#     ''' func: call simulate to create graph and train, test dataset
#         args: ti, tf, init_state = param for simulate()
#               n_train = num of training instance
#               n_test = num of test instance
#               n_nodes = num of nodes in graph
#               n_trans = num of transition phase '''

#     ##### create training dataset #####
#     X = np.zeros((n_train, n_nodes))
#     Y = np.zeros((n_train, n_nodes))

#     if torch.is_tensor(traj):
#         traj = traj.detach().cpu().numpy()
#     for i in torch.arange(0, n_train, 1):
#         i = int(i)
#         X[i] = traj[n_trans+i]
#         Y[i] = traj[n_trans+1+i]
#         # print("X", X[i])

#     X = torch.tensor(X).reshape(n_train,n_nodes)
#     Y = torch.tensor(Y).reshape(n_train,n_nodes)

#     ##### create test dataset #####
#     X_test = np.zeros((n_test, n_nodes))
#     Y_test = np.zeros((n_test, n_nodes))

#     for i in torch.arange(0, n_test, 1):
#         i = int(i)
#         X_test[i] = traj[n_trans+n_train+i]
#         Y_test[i] = traj[n_trans+1+n_train+i]

#     X_test = torch.tensor(X_test).reshape(n_test, n_nodes)
#     Y_test = torch.tensor(Y_test).reshape(n_test, n_nodes)

#     return [X, Y, X_test, Y_test]

def reg_jacobian_loss(time_step, True_J, cur_model_J, output_loss, reg_param):
    #reg_param: 1e-5 #5e-4 was working well #0.11

    diff_jac = True_J - cur_model_J
    norm_diff_jac = torch.norm(diff_jac)

    total_loss = reg_param * norm_diff_jac + (1/time_step/time_step)*output_loss

    return total_loss


def FIM(params, setting, init, delta = 0.001):

    listX = []
    dx, dt, c, n, T = setting
    params = init[1:-1]
    init_copy1 = init[1:-1].clone().detach()
    init_copy2 = init[1:-1].clone().detach()
    params_1 = params.clone().detach()
    params_2 = params.clone().detach()
    print(params.shape)

    for i in range(params.shape[0]):
        init_copy1[i] =  params[i] * (1+delta)
        init_copy2[i] = params[i] * (1-delta)
        res_1 = run_KS(init_copy1.to('cuda'), c, dx, dt, dt*2, eta, gamma, False, device)[-1]
        res_2 = run_KS(init_copy2.to('cuda'), c, dx, dt, dt*2, eta, gamma, False, device)[-1]
        # compute log prob (stein..)
        std = torch.std(params)
        mean = torch.mean(params)
        normal_dist = dist.Normal(mean, std)
        log_prob_res1 = normal_dist.log_prob(res_1)[i]
        log_prob_res2 = normal_dist.log_prob(res_2)[i]
        print("param", params[i])
        print("log", log_prob_res1)
        cur = (log_prob_res1 - log_prob_res2) / (2 * delta)
        print(cur.shape)
        listX.append(cur)

    partial_vec = torch.stack(listX)
    print(partial_vec.shape)
    partial_vec = partial_vec.view(-1, 1)
    FIM = torch.mm(partial_vec, partial_vec.T)
    return FIM

def FIM_noise(params, setting, init, eta, gamma, delta = 0.001):

    listX = []
    dx, dt, c, n, T = setting

    gaussian_noise = np.random.normal(0., 1.0, (2))
    eta_noise = eta + torch.tensor(gaussian_noise[0]).to('cuda')
    gamma_noise = eta + torch.tensor(gaussian_noise[1]).to('cuda')
    true_C = run_KS(init[1:-1], c, dx, dt, dt*2, eta, gamma, False, device)[-1]
    print("true", true_C, true_C.shape)
    simulated_C = run_KS(init[1:-1], c, dx, dt, dt*2, eta_noise, gamma_noise, False, device)[-1]
    log_likelihood = lambda simulated: 0.5*torch.norm(simulated - true_C)**2
    # This jacobian is not differentiated by parameter...
    jacobian = jacrev(log_likelihood, argnums=0)(simulated_C) # 127
    jacobian = jacobian.reshape(-1, 1)
    FIM = torch.mm(jacobian, jacobian.T) 

    return FIM

### Compute Metric ###
def one_step_rk4(f, y0, t):

    h = t[1] - t[0]
    k1 = f(t, y0)
    k2 = f(t + h/2, y0 + k1 * h / 2.)
    k3 = f(t + h/2, y0 + k2 * h / 2.)
    k4 = f(t + h, y0 + k3 * h)
    new_y = y0 + (h / 6.) * (k1 + 2*k2 + 2*k3 + k4)
    # print("new_shape", new_y.shape)
    return new_y

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

            # print("shape", traj)
            # jac = torch.func.jacrev(f)
            # x = traj[0].unsqueeze(dim=2).to('cuda')
            # batchsize = x.shape[0]
            # cur_model_J = jac(x)
            # squeezed_J = cur_model_J[:, :, 0, :, :, 0]
            # non_zero_indices = torch.nonzero(squeezed_J)
            # non_zero_values = squeezed_J[non_zero_indices[:, 0], non_zero_indices[:, 1], non_zero_indices[:, 2], non_zero_indices[:, 3]]
            # learned_J = non_zero_values.reshape(batchsize, 3, 3)
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


def main(logger, loss_type, dataset, data_config, setting, train_config):

    torch.autograd.set_detect_anomaly(True)

    X, Y, X_test, Y_test = dataset
    dx, dt, c, n, T = setting
    num_train, num_test, batch_size, dim, eta, gamma = data_config
    print(X.shape, Y.shape, X_test.shape, Y_test.shape)

    train_list = [X, Y]
    test_list = [X_test, Y_test]
    train_data = TensorDataset(*train_list)
    test_data = TensorDataset(*test_list)
    dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    print("Mini-batch: ", len(dataloader), dataloader.batch_size)

    model = FNO(
        in_channels=1,
        out_channels=1,
        num_fno_modes=25,
        padding=4,
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
    n_store, k, num_epochs, threshold, reg_param, cotangent = train_config
    timer = Timer()
    elapsed_time_train = []
    jac_diff, mse_diff = [], []

    # Jacobian Computation
    f = lambda x: run_KS(x, c, dx, dt, dt*2, eta, gamma, False, device)[-1]
    jac_diff_train, jac_diff_test = torch.empty(n_store+1), torch.empty(n_store+1)
    torch.cuda.empty_cache()

    if loss_type == "JAC":
        # make cotangent batch for the loss
        cotangent_copy = cotangent.clone().detach()
        cotangent_batch_slim = []
        for b in range(batch_size):
            cotangent_batch_slim.append(cotangent_copy)
        cotangent_batch_stacked = torch.stack(cotangent_batch_slim)
        cotangent_batch = cotangent_batch_stacked.unsqueeze(dim=1)
        print("size of cotangent batch: ", cotangent_batch.shape)

        print("Computing vjp")
        True_J = torch.zeros(len(dataloader), dataloader.batch_size, dim)
        index = 0
        for data in dataloader:
            x = data[0].to('cuda').double().requires_grad_(True)
            output, vjp_tru_func = vjp(f, x)
            res = vjp_tru_func(cotangent_batch_stacked)[0] # shape [127]
            True_J[index] = res.detach().cpu()
            index += 1
            print(index, "res", res.shape)
        # print("Sanity Check: \n", True_j[0], True_j[batch_size], True_j[2*batch_size], True_j[3*batch_size])
        # print("True: ", True_J[0:4, 0])
    
    print("Beginning training")
    for epoch in range(num_epochs):
        start_time = time.time()
        full_loss, full_test_loss = 0.0, 0.0
        idx, mse, jac = 0, 0., 0.
        test_true, test_pred = [], []

        for data in dataloader:
            optimizer.zero_grad()
            y_true = data[1].float().to('cuda')
            x = data[0].unsqueeze(dim=1).float().to('cuda')
            y_pred = model(x) #[batch, 1, 127]

            # MSE Loss
            loss_mse = criterion(y_pred.view(batch_size, -1), y_true.view(batch_size, -1))
            loss = loss_mse / torch.norm(y_true, p=2)
            mse += loss.detach().cpu().numpy()
            
            if loss_type == "JAC":
                with timer:
                    output, vjp_func = vjp(model, x)
                    vjp_out = vjp_func(cotangent_batch)[0].squeeze() 
                    # Is cotangent vector creation is problem? No
                    # Does same thing happen in MSE?
                    CurTrue_J = True_J[idx].to('cuda')
                    jac_norm_diff = criterion(CurTrue_J, vjp_out)
                    jac += jac_norm_diff.detach().cpu().numpy()
                    loss += (jac_norm_diff / torch.norm(CurTrue_J))*reg_param
    
            full_loss += loss
            idx += 1
            end_time = time.time()  
            elapsed_time_train.append(end_time - start_time)
            # test_true.append(y_true.detach().cpu().numpy())
            # test_pred.append(y_pred.squeeze().detach().cpu().numpy())
            rel_err = torch.norm(y_pred - y_true) / torch.norm(y_true)
            
        mse_diff.append(mse)
        jac_diff.append(jac)
        full_loss.backward(retain_graph=True) #retain_graph=True
        optimizer.step()
        
        for test_data in test_dataloader:
            y_test_true = test_data[1].float().to('cuda')
            y_test_pred = model(test_data[0].unsqueeze(dim=1).float().to('cuda'))
            test_loss = criterion(y_test_pred.view(batch_size, -1), y_test_true.view(batch_size, -1))
            full_test_loss += test_loss/len(test_dataloader)
            test_true.append(y_test_true.detach().cpu().numpy())
            test_pred.append(y_test_pred.squeeze().detach().cpu().numpy())
        
        print("epoch: ", epoch, "loss: ", full_loss.item()/len(dataloader), "test loss: ", full_test_loss.item())

        if full_loss < threshold:
            print("Stopping early as the loss is below the threshold.")
            break
        
    print("Finished Computing")
    modelsize = model_size(model)
    # Save the model
    torch.save(model.state_dict(), f"../test_result/best_model_FNO_KS_{loss_type}.pth")

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
    print("len", len(test_true), len(test_true[0]))
    true_traj = np.array(test_true).reshape(num_test, dim)
    learned_traj = np.array(test_pred).reshape(num_test, dim)

    plot_KS(true_traj, dx, n, c, 0, 0, (num_test)*dt, dt, False, False, True, loss_type)
    plot_KS(learned_traj, dx, n, c, 0, 0, (num_test)*dt, dt, False, False, False, loss_type)

    print("Create loss plot")
    jac_diff = np.asarray(jac_diff)
    print(jac_diff.shape)
    mse_diff = np.asarray(mse_diff)
    path = f"../plot/Loss/FNO_KS_{loss_type}.png"

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
    savefig(path, bbox_inches ='tight', pad_inches = 0.1)

    # compute LE
    # torch.cuda.empty_cache()
    # dim = 3
    # init = torch.randn(dim)
    # true_traj = torchdiffeq.odeint(lorenz, torch.randn(dim), torch.arange(0, 50, 0.01), method='rk4', rtol=1e-8)

    # init_point = torch.randn(dim)
    # learned_traj = torch.empty_like(true_traj).cuda()
    # learned_traj[0] = init_point
    # print(learned_traj.shape)
    # for i in range(1, len(learned_traj)):
    #     out = model(learned_traj[i-1].reshape(1, dim, 1).cuda()).reshape(dim,-1)
    #     print(out)
    #     learned_traj[i] = out.squeeze()


    print("Computing rest of metrics...")
    true_traj = torch.tensor(true_traj)
    learned_traj = torch.tensor(learned_traj)
    True_mean = torch.mean(true_traj, dim = 0)
    Learned_mean = torch.mean(learned_traj, dim = 0)
    True_var = torch.var(true_traj, dim = 0)
    Learned_var = torch.var(learned_traj, dim=0)

    logger.info("%s: %s", "Model Size", str(modelsize))
    logger.info("%s: %s", "Loss Type", str(loss_type))
    logger.info("%s: %s", "Cotangent", str(cotangent))
    logger.info("%s: %s", "Batch Size", str(batch_size))
    logger.info("%s: %s", "Training Loss", str(full_loss/len(dataloader)))
    logger.info("%s: %s", "Test Loss", str(full_test_loss))
    logger.info("%s: %s", "Relative Error", str(rel_err))
    # logger.info("%s: %s", "Learned LE", str(learned_LE))
    # logger.info("%s: %s", "True LE", str(True_LE))
    logger.info("%s: %s", "Mean Diff", str(torch.norm(True_mean-Learned_mean)))
    logger.info("%s: %s", "Var Diff", str(torch.norm(Learned_var-True_var)))
    logger.info("%s: %s", "MSE diff", str(mse_diff[-1]))
    if loss_type == "JAC":
        logger.info("%s: %s", "JAC diff", str(jac_diff[-1]))

    return test_true, test_pred


if __name__ == "__main__":

    # Set device
    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Enable cuDNN benchmark to allow PyTorch to find the best algorithms
    torch.backends.cudnn.benchmark = True
    print("device: ", device)

    # Set arguments (hyperparameters)
    DYNSYS_MAP = {'KS': [run_KS, 127]}

    parser = argparse.ArgumentParser()
    parser.add_argument("--time_step", type=float, default=0.25) #0.25
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--num_epoch", type=int, default=200)
    # parser.add_argument("--integration_time", type=int, default=0) #100
    # parser.add_argument("--num_train", type=int, default=400) #3000
    # parser.add_argument("--num_test", type=int, default=400)#3000
    parser.add_argument("--num_trans", type=int, default=0) #10000
    parser.add_argument("--iters", type=int, default=6000)
    parser.add_argument("--threshold", type=float, default=0.)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--loss_type", default="JAC", choices=["JAC", "MSE"])
    parser.add_argument("--reg_param", type=float, default=0.9) #1e-6
    parser.add_argument("--c", type=float, default=0.)
    parser.add_argument("--dim", type=int, default=127, choices = [127, 200, 1024]) 
    parser.add_argument("--T", type=int, default=301, choices = [5, 11, 51, 101, 201, 301, 501, 1001, 1501])
    parser.add_argument("--optim_name", default="AdamW", choices=["AdamW", "Adam", "RMSprop", "SGD"])
    parser.add_argument("--cotangent", default="FIM", choices=["ones", "rand", "QR", "FIM", "score"])
    parser.add_argument("--dyn_sys", default="KS", choices=DYNSYS_MAP.keys())

    args = parser.parse_args()
    dyn_sys_func = run_KS
    dim = args.dim
    dyn_sys_info = [dyn_sys_func, args.dyn_sys, dim]

    start_time = datetime.datetime.now().strftime("%m_%d_%H_%M_%S")
    out_file = os.path.join("../test_result/", f"FNO_KS_{start_time}.txt")
    logging.basicConfig(filename=out_file, level=logging.INFO, format="%(message)s")
    logger = logging.getLogger()
    for arg, value in vars(args).items():
        logger.info("%s: %s", arg, value)

    # Assign Initial Point of Orbit
    L = args.dim+1 #128 # n = [128, 256, 512, 700]
    n = args.dim # num of internal node
    T = args.T#1501 #1000 #100
    c = args.c
    eta = 1.
    gamma = 1. # viscosity term!

    dx = L/(n+1)
    dt = args.time_step
    x = torch.arange(0, L+dx, dx) # [0, 0+dx, ... 128] shape: L + 1
    u0 = 2.71828**(-(x-64)**2/512).to(device).double().requires_grad_(True)
    # u0 = (6.2831853/L)*x.to(device).double().requires_grad_(True)

    # boundary condition
    u0[0], u0[-1] = 0, 0 
    u0 = u0.requires_grad_(True)
    dyn_setting = [dx, dt, c, n, T]

    # Generate Training/Test/Multi-Step Prediction Data
    torch.cuda.empty_cache()
    num_samples_train = 3
    num_samples_test = 1
    length_traj = int((T-1) * int(1/dt))
    u_list_all = []
    # create pairs of eta and gamma
    eta_mean = 1.
    eta_std = 0.2
    gamma_mean = 1.
    gamma_std = 0.2
    eta_samples = torch.normal(mean=eta_mean, std=eta_std, size=(num_samples_train + num_samples_test,))
    gamma_samples = torch.normal(mean=gamma_mean, std=gamma_std, size=(num_samples_train + num_samples_test,))
    logger.info("%s: %s", "eta_samples", str(eta_samples))
    logger.info("%s: %s", "gamma_samples", str(gamma_samples))
    print("eta", eta_samples)
    print("gamma", gamma_samples)

    num_train = num_samples_train * (T-1) *int(1/dt)
    num_test = num_samples_test * (T-1) * 3 *int(1/dt)
    print("num of train", num_train)
    print("num of test", num_test)

    X = torch.zeros(num_train, dim)
    Y = torch.zeros(num_train, dim)
    X_test = torch.zeros(num_test, dim)
    Y_test = torch.zeros(num_test, dim)

    for s in range(num_samples_train):
        print("s", s)
        u_list = run_KS(u0, c, dx, dt, T+1, eta_samples[s], gamma_samples[s], False, device)
        u_short = u_list[:, 1:-1].detach().cpu() # remove the last boundary node and keep the first boundary node as it is initial condition
        print("list length", length_traj)
        plot_KS(u_short, dx, n, c, eta_samples[s], gamma_samples[s], u_short.shape[0]*dt, dt, True, False, True, args.loss_type)
        # Data split
        for i in torch.arange(0, length_traj, 1):
            X[s*length_traj + i] = u_short[i]
            # X[s*length_traj + i] = [eta_mean, eta_std, ]
            # print("1", s*length_traj + i, X[s*length_traj + i])
            Y[s*length_traj + i] = u_short[1+i]
            # print("2", s*length_traj+i, Y[s*length_traj + i] )
        # print("X", X[(s*length_traj)+i], "Y", Y[(s*length_traj)+i])


    for s in range(num_samples_test):
        u_list = run_KS(u0, c, dx, dt, T*3+1, eta_samples[s], gamma_samples[s], False, device)
        u_short = u_list[:, 1:-1].detach().cpu() # remove the last boundary node and keep the first boundary node as it is initial condition
        plot_KS(u_short, dx, n, c, eta_samples[s], gamma_samples[s], u_short.shape[0]*dt, dt, False, True, True, args.loss_type)
        # Data split
        for i in torch.arange(0, length_traj*3, 1):
            X_test[s*length_traj*3 + i] = u_short[i]
            Y_test[s*length_traj*3 + i] = u_short[1+i]
    
    dataset = [X, Y, X_test, Y_test]
    data_config = [num_train, num_test, args.batch_size, dim, eta, gamma]

    # train setting
    n_store, k = 100, 0
    if args.loss_type == "JAC":
        if args.cotangent == "ones":
            cotangent = torch.ones_like(dim).to('cuda')
        elif args.cotangent == "rand":
            cotangent = torch.rand(dim).to('cuda')
        elif args.cotangent == "FIM":
            # params, setting, init, eta, gamma, delta = 0.001
            covariance_mat = FIM_noise(args.c, dyn_setting, u0, eta_samples[0], gamma_samples[0], delta = 0.001)
            eigval, eigvecs = torch.linalg.eig(covariance_mat)
            # Find the index of the largest eigenvalue
            largest_eigval_index = torch.argmax(eigval.double())
            # Get the corresponding eigenvector
            cotangent = eigvecs[:, largest_eigval_index].float().to('cuda') # choose the first eigenvector (each column is eigen vector)
            print(eigval, largest_eigval_index, cotangent)
            logger.info("%s: %s", "max eigval", str(largest_eigval_index))
        
        train_config = [n_store, k, args.num_epoch, args.threshold, args.reg_param, cotangent]
    else:
        train_config = [n_store, k, args.num_epoch, args.threshold, None, None]

    # call main
    main(logger, args.loss_type, dataset, data_config, dyn_setting, train_config)