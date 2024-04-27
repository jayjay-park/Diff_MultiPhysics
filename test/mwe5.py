import torch
import torch.nn as nn
from torch.func import jacrev

'''
Example code to show that computed jacobian is correct.
'''

# Define input size and output size for the linear layer
input_size = 3
output_size = 3

# Create a random input tensor
input_tensor = torch.randn(1, input_size, requires_grad=True)
linear_layer = nn.Linear(input_size, output_size)
output_tensor = linear_layer(input_tensor)

# Manually compute the Jacobian matrix for the linear layer
manual_jacobian = linear_layer.weight

# Compute the Jacobian matrix using torch.jacrev
torch_jacobian = jacrev(linear_layer)(input_tensor)

# Print
print("Manually computed Jacobian:")
print(manual_jacobian)

print("\nJacobian computed using torch.jacrev:")
print(torch_jacobian)


