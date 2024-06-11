import torch
import torch.nn as nn

# Define the input tensor
input_tensor = torch.randn(1, 5)  # Example input tensor with shape (1, 5)

# Define the linear layer
linear_layer = nn.Linear(5, 3)  # Input size is 5, output size is 3

# Compute the linear transformation
output_tensor = linear_layer(input_tensor)

# Compute the 2-norm of the output tensor
output_norm = torch.norm(output_tensor, p=2)

# Compute gradients with respect to input and parameters
output_norm.backward()

# Print the gradients
print("Gradient with respect to input:")
print(input_tensor.grad)

print("\nGradients with respect to parameters:")
print(linear_layer.weight.grad)
print(linear_layer.bias.grad)
