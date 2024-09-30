import torch
import matplotlib.pyplot as plt

# Parameters
mean = torch.tensor([0.0, 0.0])  # Mean of the distribution
variance = 1.0  # Variance (same for all dimensions)
sigma = torch.sqrt(torch.tensor(variance))  # Standard deviation

# Create an isotropic Gaussian distribution
# The covariance matrix is sigma^2 * I
covariance_matrix = variance * torch.eye(2)  # 2D covariance matrix

# Number of samples
num_samples = 1000

# Generate samples from the isotropic Gaussian
samples = torch.normal(mean.unsqueeze(0).expand(num_samples, -1), sigma * torch.ones(num_samples, 2))

# Print the covariance matrix
print("Covariance Matrix:")
print(covariance_matrix)


# Plotting the samples
plt.figure(figsize=(8, 8))
plt.scatter(samples[:, 0].numpy(), samples[:, 1].numpy(), alpha=0.5, s=10)
plt.title('Samples from Isotropic Gaussian Distribution')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.axis('equal')
plt.grid()
plt.show()
