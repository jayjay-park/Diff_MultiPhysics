import torch
import torch.nn.functional as F
from functools import partial
from torch.func import vmap, vjp
from torch.func import jacrev


_ = torch.manual_seed(0)

# 1. A simple linear function
def predict(weight, bias, x):
    # y= x*A.T + b
    return 0.5 * torch.norm(F.linear(x, weight, bias), p=2)**2

def linear(weight, bias, x):
    # y= x*A.T + b
    return F.linear(x, weight, bias)

# 2. Dummy data
D = 2
weight = torch.randn(D, D, requires_grad=True)
bias = torch.randn(1, D, requires_grad=True)
x = torch.randn(1, D)
print("weight: ", weight, weight.shape)
print("input: ", x, x.shape, "\n")
print("bias: ", bias, bias.shape, "\n")


'''
torch.func.jacrev: performs vmap-vjp composition to compute Jacobians. 
'''

'''
Step 1. Compute df/dx
'''

# First order derivative is correct
ft_jacobian = jacrev(predict, argnums=2)(weight, bias, x)
print("dh/dx", ft_jacobian)
print("dh/dx", torch.mm((torch.mm(x, weight.T) + bias), weight), "\n")

# 2 Norm of First order derivative is correct
jacobian_input = lambda w: 0.5 * torch.norm(jacrev(predict, argnums=2)(w, bias, x), p=2)**2
analytical = torch.mm((torch.mm(x, weight.T) + bias), weight)
jacobian_analytical = lambda w: 0.5 * torch.norm(analytical, p=2)**2

print("Step 2: ", jacobian_input(weight))
print("Step 2: ", jacobian_analytical(weight), "\n")

# 3. Why does jacrev not work here when .backward() works?
jacobian_param = jacrev(jacobian_input)(weight)
jacobian_compute = jacrev(jacobian_analytical)(weight)
print("Step 3: ", jacobian_param)
print("Step 3: ", jacobian_compute)
print("Step 3: ", weight.grad)

# Perform a forward pass
loss = jacobian_analytical(weight)
loss.backward()
w_gradients = weight.grad
print(w_gradients)

# print("Step 3: ", torch.mm(torch.mm((torch.mm(x, weight.T) + bias), weight), torch.mm(2*weight, x.T)+bias.T))
