import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def solve_darcy(K, f, nx=50, ny=50, num_iterations=1000):
    dx = dy = 1.0 / (nx - 1)
    h = torch.zeros((nx, ny), device=device)
    
    for num in range(num_iterations):
        h_old = h.clone()
        # Update hydraulic head (using a more standard finite difference approach)
        h[1:-1, 1:-1] = 0.25 * (
            h_old[2:, 1:-1] + h_old[:-2, 1:-1] + h_old[1:-1, 2:] + h_old[1:-1, :-2]
            - dx * dy * f[1:-1, 1:-1] / K[1:-1, 1:-1]
        )
        
        # Boundary conditions (Dirichlet)
        h[0, :] = h[-1, :] = h[:, 0] = h[:, -1] = 0
    
    return h


# Generate dataset
def generate_dataset(num_samples, nx=50, ny=50):
    dataset = []
    for _ in range(num_samples):
        # Log-normal distribution for K (common in hydrology)
        K = torch.exp(torch.randn(nx, ny, device=device))
        f = torch.ones((nx, ny), device=device) * 100  # Constant source term
        h = solve_darcy(K, f)
        dataset.append((K, h))
    return dataset

# Custom Dataset
class DarcyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Neural Network Model
class DarcyNN(nn.Module):
    def __init__(self, nx, ny):
        super(DarcyNN, self).__init__()
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
train_loader = DataLoader(DarcyDataset(dataset), batch_size=32, shuffle=True)

# Initialize model, loss, and optimizer
model = DarcyNN(nx, ny).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    total_loss = 0
    for K, h in train_loader:
        K, h = K.unsqueeze(1).to(device), h.to(device)
        
        optimizer.zero_grad()
        output = model(K)
        loss = criterion(output, h)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader)}")

# Test the model
with torch.no_grad():
    test_K, test_h = dataset[0]
    print("K", test_K.shape)
    print("h", test_h.shape)
    test_K = test_K.unsqueeze(0).unsqueeze(0).to(device)
    prediction = model(test_K).squeeze().cpu()
    print("prediction", prediction.shape)
    
    print("debug1")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    im0 = axes[0].imshow(test_K.squeeze().cpu(), cmap='viridis')
    axes[0].set_title("Input: Log-Permeability (K)")
    fig.colorbar(im0, ax=axes[0])
    
    im1 = axes[1].imshow(test_h.cpu(), cmap='viridis')
    axes[1].set_title("True Hydraulic Head (h)")
    fig.colorbar(im1, ax=axes[1])
    print("debug2")
    im2 = axes[2].imshow(prediction, cmap='viridis')
    axes[2].set_title("Predicted Hydraulic Head (h)")
    fig.colorbar(im2, ax=axes[2])
    print("debug3")
    plt.tight_layout()
    plt.show()
    plt.savefig('darcy_flow_result.png', dpi=300, bbox_inches='tight')