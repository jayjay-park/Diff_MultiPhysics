import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import nflows
from nflows import flows, distributions, transforms
from modulus.models.fno import FNO

# Assuming you have these functions from your original code
from test_heat import solve_heat_equation, create_q_function, load_dataset_from_csv

# Load your trained FNO model
def load_trained_fno(path, device):
    model = FNO(
        in_channels=1,
        out_channels=1,
        decoder_layer_size=128,
        num_fno_layers=6,
        num_fno_modes=24,
        padding=3,
        dimension=2,
        latent_channels=64
    ).to(device)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

# Define the Normalizing Flow model
class NormalizingFlow(nn.Module):
    def __init__(self, dim):
        super().__init__()
        num_layers = 5
        self.flows = flows.Sequential(
            *[flows.MaskedAffineAutoregressiveTransform(
                features=dim,
                hidden_features=50,
                num_blocks=2,
                use_residual_blocks=True
            ) for _ in range(num_layers)]
        )
        self.base_dist = distributions.StandardNormal(shape=[dim])

    def forward(self, x):
        return self.flows.log_prob(x)

    def sample(self, num_samples):
        z = self.base_dist.sample(num_samples)
        return self.flows.inverse(z)

# Function to compute log likelihood
def log_likelihood(y_true, y_pred, noise_std):
    return -0.5 * torch.sum((y_true - y_pred)**2) / (noise_std**2) - \
        y_true.numel() * torch.log(torch.tensor(noise_std))

# Main function for posterior inference
def posterior_inference(args, device):
    # Load data
    test_x = load_dataset_from_csv(f'../data/test_x_{args.nx}_{args.ny}_{args.num_test}.csv', args.nx, args.ny)
    test_y = load_dataset_from_csv(f'../data/test_y_{args.nx}_{args.ny}_{args.num_test}.csv', args.nx, args.ny)
    test_dataset = TensorDataset(torch.stack(test_x), torch.stack(test_y))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Load trained FNO model
    fno_model = load_trained_fno(f"../test_result/best_model_FNO_Heat_{args.loss_type}.pth", device)

    # Initialize Normalizing Flow model
    flow_model = NormalizingFlow(dim=args.nx * args.ny).to(device)
    optimizer = optim.Adam(flow_model.parameters(), lr=1e-3)

    # Training loop for Normalizing Flow
    for epoch in range(args.num_epoch):
        total_loss = 0
        for q, T_true in test_loader:
            q, T_true = q.to(device), T_true.to(device)
            
            # Forward pass through FNO
            T_pred = fno_model(q.unsqueeze(1)).squeeze()
            
            # Compute log-likelihood
            log_prob = log_likelihood(T_true, T_pred, args.noise_std)
            
            # Forward pass through Normalizing Flow
            flow_log_prob = flow_model(q.view(q.shape[0], -1))
            
            # Compute loss
            loss = -torch.mean(log_prob + flow_log_prob)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{args.num_epoch}], Loss: {total_loss/len(test_loader):.4f}")

    # Generate samples from the posterior
    num_samples = 1000
    posterior_samples = flow_model.sample(num_samples).view(num_samples, args.nx, args.ny)

    # You can now use these samples for further analysis or visualization
    return posterior_samples

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--nx", type=int, default=100)
    parser.add_argument("--ny", type=int, default=100)
    parser.add_argument("--num_test", type=int, default=600)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epoch", type=int, default=100)
    parser.add_argument("--noise_std", type=float, default=0.01)
    parser.add_argument("--loss_type", default="MSE")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    posterior_samples = posterior_inference(args, device)

    # You can add code here to analyze or visualize the posterior samples