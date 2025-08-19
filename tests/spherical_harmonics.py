import torch
torch.serialization.add_safe_globals([slice])
import cuda_mace.ops
from e3nn import o3
from time import time

sph = torch.classes.spherical_harmonics.SphericalHarmonics()

max_ell = 3
sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
sph_e3nn = o3.SphericalHarmonics(sh_irreps, normalize=True, normalization="component")

nedges = 1000

xyz = torch.rand(nedges, 3, device="cuda", dtype=torch.float32, requires_grad=True)
xyz_cuda = xyz.detach().clone().requires_grad_(True)

out_cuda = sph.forward(xyz_cuda)
out_e3nn = sph_e3nn(xyz)

print (torch.mean(torch.abs(out_cuda.transpose(-1, -2) - out_e3nn)))

print (out_cuda.shape)
print (out_e3nn.shape)

(out_e3nn*2.0).sum().backward()
(out_cuda*2.0).sum().backward()

print(xyz.grad)
print(xyz_cuda.grad)

niter = 50

for i in range(niter):
    out_e3nn = sph_e3nn(xyz)
torch.cuda.synchronize()

start = time()
for i in range(niter):
    out_e3nn = sph_e3nn(xyz)
    (out_e3nn**2.0).sum().backward()
torch.cuda.synchronize()
end = time()

print(f"--E3NN time (ms)")
print(f"{((end - start) * 1000 / niter):.2e} ms")

start = time()
for i in range(niter):
    out_cuda = sph.forward(xyz)
    (out_cuda**2.0).sum().backward()
torch.cuda.synchronize()
end = time()

print(f"--CUDA time (ms)")
print(f"{((end - start) * 1000 / niter):.2e} ms")
