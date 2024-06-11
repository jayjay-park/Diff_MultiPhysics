import torch
import torch.nn.functional as F
from functools import partial
from torch.func import vmap, vjp
from torch.func import jacrev


_ = torch.manual_seed(0)

# 1. A simple linear function we want to compute the jacobian of 
def predict(weight, bias, x):
    return F.linear(x, weight, bias).tanh()

# 2. Dummy data
D = 16
weight = torch.randn(D, D)
bias = torch.randn(D)
x = torch.randn(D)  # feature vector

'''
PyTorch Autograd computes vecto-Jacobian products. To compute the full Jacobian of this function, we have to compute it row by row using a different unit vector each time.
'''

# 3. Compute Jacobian Matrix using Autograd
def compute_jac(xp):
    jacobian_rows = [torch.autograd.grad(predict(weight, bias, xp), xp, vec)[0]
                     for vec in unit_vectors]
    return torch.stack(jacobian_rows)

xp = x.clone().requires_grad_()
unit_vectors = torch.eye(D)
jacobian = compute_jac(xp)

print(jacobian.shape)
print(jacobian[0])  # show first row

'''
torch.func.vjp comes with vmap
torch.func.vjp computes the reverse-mode Jacobian of func with respect to primals times cotangents
'''

# 4. use vjp
_, vjp_fn = vjp(partial(predict, weight, bias), x)

ft_jacobian, = vmap(vjp_fn)(unit_vectors)

# let's confirm both methods compute the same result
assert torch.allclose(ft_jacobian, jacobian)

'''
torch.func.jacrev: performs vmap-vjp composition to compute Jacobians. 
'''
ft_jacobian = jacrev(predict, argnums=2)(weight, bias, x)

# Confirm by running the following:
assert torch.allclose(ft_jacobian, jacobian)