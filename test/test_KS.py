import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import logging
import os
import csv
import pandas as pd
import time
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, TensorDataset, Subset
from modulus.models.fno import FNO
from baseline import *

def kuramoto_sivashinsky_step(u_t, nu=1, gamma=1, L=100, nx=1024, dt=0.05):
    '''Solve one time step of the Kuramoto-Sivashinsky equation with an additional gamma parameter for the 4th-order term.'''
    
    # Wave number mesh
    k = torch.arange(-nx/2, nx/2, 1, dtype=torch.float32, device=u_t.device)
    
    # Fourier Transform of the linear operator
    FL = (((2 * np.pi) / L) * k) ** 2 - gamma * nu * (((2 * np.pi) / L) * k) ** 4
    
    # Fourier Transform of the non-linear operator
    FN = - (1 / 2) * ((1j) * ((2 * np.pi) / L) * k)
    
    # Fourier Transform of current state
    u_hat = (1 / nx) * torch.fft.fftshift(torch.fft.fft(u_t))
    u_hat2 = (1 / nx) * torch.fft.fftshift(torch.fft.fft(u_t**2))
    
    # Crank-Nicholson + Adam scheme
    u_hat_next = (1 / (1 - (dt / 2) * FL)) * (
        (1 + (dt / 2) * FL) * u_hat + 
        (((3 / 2) * FN) * u_hat2 - ((1 / 2) * FN) * u_hat2) * dt
    )
    
    # Go back to real space
    u_t_next = torch.real(nx * torch.fft.ifft(torch.fft.ifftshift(u_hat_next)))

    return u_t_next


def log_likelihood(data, model_output, noise_std):
    # return -0.5 * torch.sum((data - model_output)**2) / (noise_std**2) - \
    #     data.numel() * torch.log(torch.tensor(noise_std))
    return (1/(2*noise_std**2))*torch.sum((data - model_output)**2)

def compute_fim_KS(simulator, input, T_data, noise_std, nx, time_step):
    # Ensure k is a tensor with gradient tracking
    q = input.requires_grad_().cuda()
    fim = torch.zeros((nx, nx))
    
    # # Add noise
    mean = 0.0
    std_dev = 0.1

    # Generate Gaussian noise
    noise = torch.randn(q.size()) * std_dev + mean
    # Solve heat equation
    # w_current = ns_solver(w_current, f=forcing, T=time_step, Re=Re)
    # T_pred = simulator(q, f=forcing, T=time_step, Re=Re)
    T_pred = simulator(input, nu=1, gamma=1, L=100, nx=nx, dt=time_step)
    T_pred = T_pred + noise.cuda()
    ll = log_likelihood(T_data.cuda(), T_pred, noise_std)
    flat_Jacobian = torch.autograd.grad(inputs=q, outputs=ll, create_graph=True)[0].flatten() # 50 by 50 -> [2500]
    print("flatten", flat_Jacobian.shape)
    flat_Jacobian = flat_Jacobian.reshape(1, -1)
    fim = torch.matmul(flat_Jacobian.T, flat_Jacobian)

    return fim

class KSDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def generate_dataset(num_samples, num_init, time_step, nx=1024, L=100, nu=1):
    input, output = [], []

    for s in range(num_init):
        print(f"Generating data for initialization {s}")
        
        # Generate initial condition
        eta_mean = 1.
        eta_std = 0.2
        gamma_mean = 1.
        gamma_std = 0.2
        eta_samples = torch.normal(mean=eta_mean, std=eta_std, size=(1,)).cuda()
        gamma_samples = torch.normal(mean=gamma_mean, std=gamma_std, size=(1,)).cuda()

        x = torch.linspace(0, L, nx, device="cuda")
        u0 = torch.cos((2 * np.pi * x) / L) + 0.1 * torch.cos((4 * np.pi * x) / L)
        u_current = u0.cuda()
        
        u_data = [u_current.cpu().numpy()]

        # Solve the KS equation
        for i in range(num_samples // num_init):
            u_next = kuramoto_sivashinsky_step(u_current, nu=eta_samples, gamma=gamma_samples, L=L, nx=nx, dt=time_step)
            u_data.append(u_next.cpu().numpy())
            u_current = u_next
        
        input.append(u_data[:-1])
        output.append(u_data[1:])

    return input, output

def save_dataset_to_csv(dataset, prefix):
    df = pd.DataFrame(dataset)
    df.to_csv(f'{prefix}', index=False)
    print(f"Saved {prefix} dataset to CSV file")

def load_dataset_from_csv(prefix, nx):
    df = pd.read_csv(f'{prefix}')
    data = [torch.tensor(row.values).reshape(nx) for _, row in df.iterrows()]
    return data

def plot_results(true, pred, path):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(true.cpu().numpy(), aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.title('True Solution')

    plt.subplot(1, 3, 2)
    plt.imshow(pred.cpu().numpy(), aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.title('Predicted Solution')

    plt.subplot(1, 3, 3)
    plt.imshow((true - pred).cpu().numpy(), aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.title('Error')

    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()

def main(logger, args, loss_type, dataloader, test_dataloader, vec, simulator):
    # Initialization
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device: ", device)

    model = FNO(
        in_channels=1,  # Adjusted for vx and vy inputs
        out_channels=1, # Adjusted for wz output
        decoder_layer_size=128,
        num_fno_layers=6,
        num_fno_modes=20,
        padding=3,
        dimension=2,
        latent_channels=64
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-5
    )

    criterion = torch.nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1e-3)
    nx, ny = args.nx, args.ny

    # Gradient-matching and training logic
    if args.loss_type == "Sobolev":
        Sobolev_Loss = HsLoss_2d()
    elif args.loss_type == "Dissipative":
        Sobolev_Loss = HsLoss_2d()
        # DISSIPATIVE REGULARIZATION PARAMETERS
        # below, the number before multiplication by S is the radius in the L2 norm of the function space
        S=args.nx
        radius = 156.25 * S # radius of inner ball
        scale_down = 0.5 # rate at which to linearly scale down inputs
        loss_weight = 0.01 * (S**2) # normalized by L2 norm in function space
        radii = (radius, (525 * S) + radius) # inner and outer radii, in L2 norm of function space
        sampling_fn = sample_uniform_spherical_shell #numsampled is batch size
        target_fn = linear_scale_dissipative_target
        dissloss = nn.MSELoss(reduction='mean')

        modes = 20
        width = 64

        in_dim = 1
        out_dim = 1
    elif args.loss_type == "JAC":
        csv_filename = f'../data/true_j_NS_{nx}_{args.num_train}.csv'
        if os.path.exists(csv_filename):
            # Load True_j
            True_j_flat = pd.read_csv(csv_filename).values
            print("len", True_j_flat.shape, len(dataloader)*dataloader.batch_size*nx*ny)
            True_j = torch.tensor(True_j_flat)[:len(dataloader)*dataloader.batch_size, :].reshape(len(dataloader), dataloader.batch_size, nx, ny)
            print(f"Data loaded from {csv_filename}")
        else:
            True_j = torch.zeros(len(dataloader), dataloader.batch_size, nx, ny)
            f = lambda x: simulator(x, f=forcing, T=args.time_step, Re=Re) 
            # Iterate over the DataLoader
            for batch_idx, (batch_data, batch_labels) in enumerate(dataloader):
                for i in range(batch_data.shape[0]):  # Iterate over each sample in the batch
                    # single sample [nx, ny]
                    x = batch_data[i]
                    output, vjp_tru_func = torch.func.vjp(f, x.cuda())
                    print(batch_idx, i)
                    True_j[batch_idx, i] = vjp_tru_func(vec)[0].detach().cpu()

            # Save True_j to a CSV file
            True_j_flat = True_j.reshape(-1, nx * ny)  # Flatten the last two dimensions
            pd.DataFrame(True_j_flat.numpy()).to_csv(csv_filename, index=False)
            print(f"Data saved to {csv_filename}")
        # Create vec_batch
        True_j = True_j.float()
        vec_batch = vec.unsqueeze(0).repeat(dataloader.batch_size, 1, 1)
        vec_batch = vec_batch.cuda().float()

        # Create vec_batch
        True_j = True_j.float()
        vec_batch = vec.unsqueeze(0).repeat(dataloader.batch_size, 1, 1, 1)
        print("vec", vec_batch.shape)
        vec_batch = vec_batch.cuda().float()


    ### Training Loop ###
    elapsed_time_train, mse_diff, jac_diff_list, test_diff = [], [], [], []
    lowest_loss = float('inf')

    print("Beginning training")
    for epoch in range(args.num_epoch):
        start_time = time.time()
        full_loss, full_test_loss, jac_misfit = 0.0, 0.0, 0.0
        idx = 0
        
        for X, Y in dataloader:
            X, Y = X.cuda().float(), Y.cuda().float()
            
            # MSE 
            optimizer.zero_grad()
            if args.loss_type == "MSE":
                output = model(X.unsqueeze(dim=1))
                loss = criterion(output.squeeze(), Y.squeeze()) / torch.norm(Y)
            elif args.loss_type == "Sobolev":
                output = model(X.unsqueeze(dim=1))
                loss = criterion(output.squeeze(), Y.squeeze()) / torch.norm(Y)
                sob_loss = Sobolev_Loss(output.squeeze(), Y.squeeze())
                loss += sob_loss
            elif args.loss_type == "Dissipative":
                output = model(X.unsqueeze(dim=1))
                loss = Sobolev_Loss(output.squeeze(), Y.squeeze())
                x_diss = torch.tensor(sampling_fn(X.shape[0], radii, (S, S, 2)), dtype=torch.float).to(device)
                y_diss = torch.tensor(target_fn(x_diss, scale_down), dtype=torch.float).to(device)
                out_diss = model(x_diss.reshape(-1, 2, S, S)).reshape(-1, out_dim)
                diss_loss = (1/(S**2)) * loss_weight * dissloss(out_diss, y_diss.reshape(-1, out_dim)) # weighted by 1 / (S**2)
                loss += diss_loss
            else:
            # GM
                target = True_j[idx].cuda()
                output, vjp_func = torch.func.vjp(model, X.unsqueeze(dim=1))
                loss = criterion(output.squeeze(), Y.squeeze()) / torch.norm(Y)
                vjp_out = vjp_func(vec_batch)[0].squeeze()
                jac_diff = criterion(target, vjp_out)
                jac_misfit += jac_diff.detach().cpu().numpy()
                loss += jac_diff * args.reg_param

            loss.backward(retain_graph=True)
            optimizer.step()
            full_loss += loss.item()
            idx += 1
        
        # Save loss
        mse_diff.append(full_loss)
        if args.loss_type == "JAC":
            jac_diff_list.append(jac_misfit)
        # Save time
        end_time = time.time()  
        elapsed_time_train.append(end_time - start_time)
        
        # Validation
        model.eval()
        with torch.no_grad():
            for X_test, Y_test in test_dataloader:
                X_test, Y_test = X_test.cuda().float(), Y_test.cuda().float()
                output = model(X_test.unsqueeze(dim=1))
                test_loss = criterion(output.squeeze(), Y_test) / torch.norm(Y_test)
                full_test_loss += test_loss.item()
            test_diff.append(full_test_loss)
        model.train()
        
        print(f"Epoch: {epoch}, Train Loss: {full_loss:.6f}, JAC misfit: {jac_misfit}, Test Loss: {full_test_loss:.6f}")
        if epoch % 50 == 0:
            torch.save(model.state_dict(), f"../test_result/Checkpoint/FNO_NS_vort_{loss_type}_{args.nx}_{args.num_train}_{epoch}.pth")
        if full_test_loss < lowest_loss:
            print("saved lowest loss model")
            lowest_loss = full_test_loss
            torch.save(model.state_dict(), f"../test_result/best_model_FNO_NS_vort_{loss_type}.pth")
            # Save plot
            X_test, Y_test = next(iter(test_dataloader))
            X_test, Y_test = X_test.cuda().float(), Y_test.cuda().float()
            print("shape", X_test.shape, Y_test.shape)
            with torch.no_grad():
                Y_pred = model(X_test.unsqueeze(dim=1))
            plot_path = f"../plot/NS_plot/FNO_NS_vort_lowest_{loss_type}.png"
            plot_results(Y_test[0].squeeze().cpu(), Y_test[1].squeeze().cpu(), Y_pred[0].squeeze(), Y_pred[1].squeeze(), plot_path)
                
        if full_loss < args.threshold:
            print("Stopping early as the loss is below the threshold.")
            break

    print("Finished Computing")

    # Save the model
    torch.save(model.state_dict(), f"../test_result/best_model_FNO_NS_vort_full epoch_{loss_type}.pth")
    # Save the elapsed times
    with open(f'../test_result/Time/FNO_NS_vort_{args.loss_type}_{args.nx}_{args.num_train}.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Epoch', 'Elapsed Time (seconds)'])
        for epoch, elapsed_time in enumerate(elapsed_time_train, 1):
            writer.writerow([epoch, elapsed_time])
    # Save the losses
    loss_data = [
        (mse_diff, 'mse_loss'),
        (jac_diff_list, 'jac_loss') if args.loss_type == "JAC" else (None, None),
        (test_diff, 'test_loss')
    ]
    for data, name in loss_data:
        if data:
            with open(f'../test_result/Losses/NS_vort_{name}_{args.loss_type}_{args.nx}_{args.num_train}.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Epoch', 'Loss'])
                writer.writerows(enumerate(data, 1))
    print("Losses saved to CSV files.")

    # Create loss plot
    print("Create loss plot")
    mse_diff = np.asarray(mse_diff)
    jac_diff_list = np.asarray(jac_diff_list)
    test_diff = np.asarray(test_diff)
    path = f"../plot/Loss/FNO_NS_vort_{loss_type}.png"

    fig, ax = plt.subplots()
    ax.plot(mse_diff, "P-", lw=2.0, ms=6.0, color="coral", label="MSE (Train)")
    ax.plot(test_diff, "P-", lw=2.0, ms=6.0, color="indianred", label="MSE (Test)")
    if args.loss_type == "JAC":
        ax.plot(jac_diff_list, "P-", lw=2.0, color="slateblue", ms=6.0, label=r"$\|J^Tv - \hat{J}^Tv\|$")
    ax.set_xlabel("Epochs",fontsize=24)
    ax.set_ylabel("Loss", fontsize=24)
    ax.legend()
    plt.savefig(path, dpi=150, bbox_inches="tight")

    print("Plot saved.")


    logger.info("%s: %s", "Loss Type", str(loss_type))
    logger.info("%s: %s", "Batch Size", str(args.batch_size))
    logger.info("%s: %s", "Training Loss", str(full_loss/len(dataloader)))
    logger.info("%s: %s", "Test Loss", str(full_test_loss/len(test_dataloader)))
    logger.info("%s: %s", "MSE diff", str(mse_diff[-1]))

    return model


if __name__ == "__main__":
    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.backends.cudnn.benchmark = True
    print("device: ", device)

    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--num_epoch", type=int, default=1000)
    parser.add_argument("--num_train", type=int, default=1000)
    parser.add_argument("--num_test", type=int, default=200)
    parser.add_argument("--num_init", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=1e-8)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--loss_type", default="MSE", choices=["MSE"])
    parser.add_argument("--nx", type=int, default=1024)
    parser.add_argument("--noise", type=float, default=0.01)
    parser.add_argument("--nu", type=float, default=1) # Viscosity
    parser.add_argument("--time_step", type=float, default=0.05) # time step

    args = parser.parse_args()

    # Save initial settings
    start_time = time.strftime("%m_%d_%H_%M_%S")
    out_file = os.path.join("../test_result/", f"FNO_KS_{start_time}.txt")
    logging.basicConfig(filename=out_file, filemode="w", level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(__name__)
    for arg, value in vars(args).items():
        logger.info("%s: %s", arg, value)

    # Generate Training/Test Data
    trainx_file = f'../data/KS/train_x_{args.nx}_{args.num_train}.csv'
    trainy_file = f'../data/KS/train_y_{args.nx}_{args.num_train}.csv'
    testx_file = f'../data/KS/test_x_{args.nx}_{args.num_test}.csv'
    testy_file = f'../data/KS/test_y_{args.nx}_{args.num_test}.csv'
    if not os.path.exists(trainx_file):
        print("Creating Dataset")
        input, output = generate_dataset(args.num_train + args.num_test, args.num_init, args.time_step, args.nx)
        input = torch.tensor(input).reshape(-1, args.nx)
        output = torch.tensor(output).reshape(-1, args.nx)
        print("data size", len(input), len(output))

        train_x = input[:args.num_train].detach().cpu().numpy()
        train_y = output[:args.num_train].detach().cpu().numpy()
        test_x = input[args.num_train:].detach().cpu().numpy()
        test_y = output[args.num_train:].detach().cpu().numpy()
        # Save datasets to CSV files
        save_dataset_to_csv(train_x, trainx_file)
        save_dataset_to_csv(train_y, trainy_file)
        save_dataset_to_csv(test_x, testx_file)
        save_dataset_to_csv(test_y, testy_file)

    print("Loading Dataset")
    train_x_raw = load_dataset_from_csv(trainx_file, args.nx)
    train_y_raw = load_dataset_from_csv(trainy_file, args.nx)
    test_x_raw = load_dataset_from_csv(testx_file, args.nx)
    test_y_raw = load_dataset_from_csv(testy_file, args.nx)

    # Randomly sample indices for train and test sets
    train_indices = np.random.choice(len(train_x_raw), args.num_train, replace=False)
    test_indices = np.random.choice(len(test_y_raw), args.num_test, replace=False)
    # Create subsets of the datasets
    train_dataset = TensorDataset(torch.stack(train_x_raw), torch.stack(train_y_raw))
    test_dataset = TensorDataset(torch.stack(test_x_raw), torch.stack(test_y_raw))
    train_dataset = Subset(train_dataset, train_indices)
    test_dataset = Subset(test_dataset, test_indices)
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    print("Mini-batch: ", len(train_loader), train_loader.batch_size)

    # compute FIM eigenvector
    if args.loss_type == "JAC":
        nx, ny = args.nx, args.ny
        noise_std = 0.01
        print("Reloaded train: ", train_x_raw[0].shape)
        # fim = compute_fim_NS(ns_solver, train_x_raw[0].cuda(), train_y_raw[0].cuda(), noise_std, nx, ny, forcing, args.time_step, Re).detach().cpu()
        fim =  compute_fim_KS(kuramoto_sivashinsky_step, train_x_raw[0].cuda(), train_y_raw[0].cuda(), noise_std, args.nx, args.time_step)
        # Compute FIM
        for s in range(args.num_sample - 1):
            print(s)
            # k = torch.exp(torch.randn(nx, ny)).cuda()  # Log-normal distribution for k
            fim += compute_fim_KS(kuramoto_sivashinsky_step, train_x_raw[0].cuda(), train_y_raw[0].cuda(), noise_std, args.nx, args.time_step).detach().cpu()
        fim /= args.num_sample

        # Analyze the FIM
        eigenvalues, eigenvec = torch.linalg.eigh(fim.cuda())

        # Sort eigenvalues in descending order and get their indices
        sorted_indices = torch.argsort(eigenvalues, descending=True)

        # Iterate over sorted eigenvalues to find the largest one with a non-zero eigenvector
        largest_eigenvector = None
        for idx in sorted_indices:
            print("candidate idx", idx)
            candidate_eigenvector = eigenvec[:, idx]
            
            # Check if the eigenvector is non-zero
            if torch.any(candidate_eigenvector != 0):
                largest_eigenvector = candidate_eigenvector
                break
        # Handle the case where no non-zero eigenvector is found
        if largest_eigenvector is None:
            raise ValueError("No non-zero eigenvector found.")

        # idx = torch.argmax(eigenvalues)
        # largest_eigenvector = eigenvec[:, idx]
        largest_eigenvector = largest_eigenvector.reshape(args.nx)

        print("Largest Eigenvalue and index:", eigenvalues[idx], idx)
        print("Corresponding Eigenvector:", largest_eigenvector)
        print("Eigenvector shape", largest_eigenvector.shape)
        print("eigenvalue: ", eigenvalues)
        print("eigenvector: ", eigenvec)
    else:
        largest_eigenvector = None

    # train
    main(logger, args, args.loss_type, train_loader, test_loader, largest_eigenvector, kuramoto_sivashinsky_step)