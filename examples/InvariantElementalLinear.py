# Implementation of the linear layer
from time import time
from typing import List
from math import prod
import torch
from e3nn import o3
from mace_ops.ops.linear import ElementalLinear

torch.backends.cuda.matmul.allow_tf32 = False

# INPUTS#
n_channels = 96
n_out_channels = n_channels
max_l = 3
nnodes = 5000
nelements = 3

x = torch.randn(nnodes, (max_l+1)**2, n_channels,
                device='cuda', dtype=torch.float32, requires_grad=True)
weights = torch.randn(nelements, 4, n_channels, n_channels, device='cuda', dtype=torch.float)
weights_transposed = weights.transpose(-1, -2).contiguous()

one_hot_embedding = torch.randn(nnodes, nelements, device='cuda', dtype=torch.float)

print (one_hot_embedding.argmax(dim=-1))
one_hot_embedding[:] = 0.0
one_hot_embedding[0:10, 0] = 1
one_hot_embedding[10:90,1] = 1
one_hot_embedding[90:,2] = 1

one_hot_embedding = one_hot_embedding.int()

print (one_hot_embedding)
## E3NN LINEAR##
irreps_in = o3.Irreps(
    (n_channels * o3.Irreps.spherical_harmonics(max_l))
    .sort()
    .irreps.simplify()
)
irreps_out = o3.Irreps(
    f"{n_out_channels}x0e + {n_out_channels}x1o + {n_out_channels}x2e + {n_out_channels}x3o")


#need o3 linear to pull weights asnd instructions
linear = o3.Linear(irreps_in=irreps_in, irreps_out=irreps_out).to('cuda')

instructions = linear.instructions
ws = linear.weight



cuda_out = torch.ops.linear_wmma.elemental_linear(x, weights, weights_transposed, one_hot_embedding)
print (cuda_out)


torch.cuda.cudart().cudaProfilerStart()
torch.cuda.synchronize()
start = time()
for i in range(1000):
    cuda_out = torch.ops.linear_wmma.elemental_linear(x, weights, weights_transposed, one_hot_embedding)
    t = cuda_out.sum() * 2.0
    t.backward()
torch.cuda.synchronize()
end = time()
torch.cuda.cudart().cudaProfilerStop()
print("fwd CUDA linear:", end - start)
