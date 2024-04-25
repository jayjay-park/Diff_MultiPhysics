import torch
import torch.nn.functional as F
from functools import partial
from torch.func import vmap, vjp
from torch.func import jacrev


_ = torch.manual_seed(0)

# 1. A simple linear function
def predict(weight, bias, x):
    return 0.5 * torch.linalg.norm(F.linear(x, weight, bias), 2)**2 #.tanh()

# 2. Dummy data
D = 3
weight = torch.randn(D, D)
bias = torch.randn(D)
x = torch.randn(D)
print("weight: ", weight)
print("input: ", x, "\n")


'''
torch.func.jacrev: performs vmap-vjp composition to compute Jacobians. 
'''

'''
Step 1. Compute df/dx
'''

jacobian_input = lambda x: jacrev(predict, argnums=2)(weight, bias, x)
print("Step 1: ", jacobian_input, "\n")

'''
Step 2. Compute h(f) = 1/2||f(x)||^2_2
'''

# norm = lambda v: 0.5 * torch.linalg.norm(v, 2)**2
# print("Step 2: ", norm(jacobian_input), "\n")

'''
Step 3. Compute dh/df
'''

jacobian_param = jacrev(jacobian_input, argnums=0)(weight)
print("Step 3: ", jacobian_param)