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
cotangents = torch.ones(1, D)
cotangent_gradient = torch.tensor(1.)
print("weight: ", weight, weight.shape)
print("input: ", x, x.shape, "\n")
print("bias: ", bias, bias.shape, "\n")
print("cotangents", cotangents.shape)


'''
Step 1. Compute df/dx
'''

# First order derivative is correct
output, vjp_func = vjp(predict, weight, bias, x)

print("output", output, "\n")
print("dh/dx", vjp_func(cotangent_gradient)[2], "\n")
print("dh/dx", torch.mm((torch.mm(x, weight.T) + bias), weight), "\n")

# 2 Norm of First order derivative is correct
vjp_lambda = lambda w, c: vjp(predict, w, bias, x)[1](c) # get only the function
jacobian_input = lambda w: 0.5 * torch.norm(vjp_lambda(w, cotangent_gradient)[2], p=2)**2
print("Step 2: ", jacobian_input(weight), "\n")

analytical = lambda w: torch.mm((torch.mm(x, w.T) + bias), w)
jacobian_analytical = lambda w: 0.5 * torch.norm(analytical(w), p=2)**2
print("Step 2: ", jacobian_analytical(weight), "\n")

# 3. Differentiate w.r.t. network parameter
w2 = weight.clone().detach().requires_grad_(True)

vjp_norm = jacobian_input(weight)
vjp_norm.backward()
print("Step 3: ", weight.grad)

vjp_true_norm = jacobian_analytical(w2)
vjp_true_norm.backward()
print("Step 3: ", w2.grad)
