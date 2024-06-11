import torch
import torch.nn as nn
import torch.autograd.functional as F

# Define input size and output size for the linear layer
input_size = 5
output_size = 3
input_tensor = torch.randn(1, input_size, requires_grad=True)
linear_layer = nn.Linear(input_size, output_size)
# Compute the forward pass
output_tensor = linear_layer(input_tensor)

# Compute the first derivative with respect to input
grad_input = torch.autograd.grad(output_tensor.sum(), input_tensor, create_graph=True)[0]
twonorm_grad = 0.5 * torch.norm(grad_input, p=2) **2

print("grad", grad_input)

# Compute the second derivative (Hessian) with respect to parameters
parameters = list(linear_layer.parameters())
hessian_params = torch.autograd.grad(twonorm_grad, parameters, retain_graph=True)

# Manually compute the Hessian with respect to parameters (which should be zeros for a linear layer)
manual_hessian_params = [torch.zeros_like(param) for param in parameters]

# Print the manually computed Hessian with respect to parameters and the computed Hessian
print("Manually computed Hessian with respect to parameters:")
print(manual_hessian_params)

print("\nHessian with respect to parameters computed using autograd:")
print(hessian_params)

# Compare the results by checking if all elements of the Hessians are close to zero
print("\nAre the two Hessians (with respect to parameters) close to zero?")
print(all(torch.allclose(hessian_param, manual_hessian_param) for hessian_param, manual_hessian_param in zip(hessian_params, manual_hessian_params)))
