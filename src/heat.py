import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Simulation function
def solve_heat_equation(k, q, nx=50, ny=50, num_iterations=1000):
    dx = dy = 1.0 / (nx - 1)
    T = torch.zeros((nx, ny), device=device)
    
    for _ in range(num_iterations):
        T_old = T.clone()
        T[1:-1, 1:-1] = (
            k[1:-1, 1:-1] * (T_old[2:, 1:-1] / k[2:, 1:-1] + T_old[:-2, 1:-1] / k[:-2, 1:-1] + 
                             T_old[1:-1, 2:] / k[1:-1, 2:] + T_old[1:-1, :-2] / k[1:-1, :-2])
            - dx * dy * q[1:-1, 1:-1]
        ) / (k[1:-1, 1:-1] * (1/k[2:, 1:-1] + 1/k[:-2, 1:-1] + 1/k[1:-1, 2:] + 1/k[1:-1, :-2]))
        
        # Boundary conditions (Dirichlet)
        T[0, :] = T[-1, :] = T[:, 0] = T[:, -1] = 0
    
    return T

# Generate dataset
def generate_dataset(num_samples, nx=50, ny=50):
    dataset = []
    for _ in range(num_samples):
        # Log-normal distribution for k (common in heat transfer problems)
        k = torch.exp(torch.randn(nx, ny, device=device))
        q = torch.ones((nx, ny), device=device) * 100  # Constant heat source term
        T = solve_heat_equation(k, q)
        dataset.append((k, T))
    return dataset

# Custom Dataset
class HeatDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Neural Network Model
class HeatNN(nn.Module):
    def __init__(self, nx, ny):
        super(HeatNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(32, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.conv4(x)
        return x.squeeze(1)

# Generate dataset
nx, ny = 50, 50
num_samples = 1000
dataset = generate_dataset(num_samples, nx, ny)
train_loader = DataLoader(HeatDataset(dataset), batch_size=32, shuffle=True)

# Initialize model, loss, and optimizer
model = HeatNN(nx, ny).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    total_loss = 0
    for k, T in train_loader:
        k, T = k.unsqueeze(1).to(device), T.to(device)
        
        optimizer.zero_grad()
        output = model(k)
        loss = criterion(output, T)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader)}")

# Test the model
with torch.no_grad():
    test_k, test_T = dataset[0]
    test_k = test_k.unsqueeze(0).unsqueeze(0).to(device)
    prediction = model(test_k).squeeze().cpu()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    im0 = axes[0].imshow(test_k.squeeze().cpu(), cmap='viridis')
    axes[0].set_title("Input: Log-Thermal Conductivity (k)")
    fig.colorbar(im0, ax=axes[0])
    
    im1 = axes[1].imshow(test_T.cpu(), cmap='viridis')
    axes[1].set_title("True Temperature (T)")
    fig.colorbar(im1, ax=axes[1])
    
    im2 = axes[2].imshow(prediction, cmap='viridis')
    axes[2].set_title("Predicted Temperature (T)")
    fig.colorbar(im2, ax=axes[2])
    
    plt.tight_layout()
    plt.show()