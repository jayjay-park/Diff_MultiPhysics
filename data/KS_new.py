import torch as th
from torch.fft import fft, ifft
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import cm
from mpl_toolkits.mplot3d import Axes3D

# Spatial grid and initial condition
N = 1024
x = 32*th.pi*th.arange(1, N + 1) / N
u = th.cos(x / 16) * (1 + th.sin(x / 16))
v = fft(u) # fourier transform of initial u

# Precompute ETDRK4 scalar quantities
h = 1 / 4
k = th.cat([th.arange(0, N / 2), th.tensor([0.]), th.arange(-N / 2 + 1, 0)], 0) / 16
# L = k**2 - k**4
L = k**2 - k**4
E = (h * L).exp()
E2 = (h * L / 2).exp()
M = 16
r = (1j * th.pi * (th.arange(1, M + 1) - .5) / M)
LR = h * L[:, None].repeat_interleave(M, 1) + r[None, :].repeat_interleave(N, 0)
Q = h * (((LR / 2).exp() - 1) / LR).mean(dim=1).real
f1 = h * ((-4 - LR + LR.exp() * (4 - 3 * LR + LR**2)) / LR**3).mean(dim=1).real
f2 = h * ((2 + LR + LR.exp() * (-2 + LR)) / LR**3).mean(dim=1).real
f3 = h * ((-4 - 3 * LR - LR**2 + LR.exp() * (4 - LR)) / LR**3).mean(dim=1).real

# Timestepping
uu = [u]
tt = [0]
tmax = 200
nmax = int(tmax / h)
nplt = int((tmax / 100) / h)
g = -.5j * k

for n in range(1, nmax + 1):
    t = n * h
    Nv = g * fft(ifft(v).real**2)
    a = E2 * v + Q * Nv
    Na = g * fft(ifft(a).real**2)
    b = E2 * v + Q * Na
    Nb = g * fft(ifft(b).real**2)
    c = E2 * a + Q * (2 * Nb - Nv)
    Nc = g * fft(ifft(c).real**2)
    v = E * v + Nv * f1 + 2 * (Na + Nb) * f2 + Nc * f3
    if n % nplt == 0:
        u = ifft(v).real
        uu.append(u)
        tt.append(t)

uu = th.stack(uu)
tt = th.tensor(tt)

print(uu.shape, tt.shape)
print(tt[0], tt[-1])

# Save the image
# plt.imshow(uu)
# plt.colorbar()  # Adding a colorbar to give more information about the values
# plt.title('Evolution of u over time')
# plt.xlabel('Spatial Dimension')
# plt.ylabel('Time Step')
# plt.savefig("../plot/KS_new.png")  # Save the image as 'evolution_of_u.png'
# plt.show()


fig, ax = plt.subplots(figsize=(12, 18))
x = np.linspace(0, N, uu.shape[1])
t = np.linspace(0, tt[-1], tt.shape[0])
print("t", t)
xx, tt = np.meshgrid(x, t)
levels = np.arange(-4, 4, 0.01)
cs = ax.contourf(xx, tt, uu, cmap=cm.jet)
cbar = fig.colorbar(cs)
cbar.ax.tick_params(labelsize=33)
ax.set_xlabel("X", fontsize=35)
ax.set_ylabel("T", fontsize=35)
ax.xaxis.set_tick_params(labelsize=34)
ax.yaxis.set_tick_params(labelsize=34)
plt.tight_layout()
plt.savefig("../plot/KS_new.png")


# ax.set_xlabel("x")
# ax.set_ylabel("t")
# ax.set_title(f"Kuramoto-Sivashinsky: L = {L}")
# fig.savefig("../plot/KS_new.png")