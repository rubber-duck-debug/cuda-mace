# Implementation of the linear layer
from time import time
import torch
from mace_ops import cuda

torch.backends.cuda.matmul.allow_tf32 = False

# INPUTS#
n_channels = 128
n_out_channels = 128

max_l = 3

nnodes = 5000

x = torch.randn(nnodes, (max_l+1)**2, n_channels,
                device='cuda', dtype=torch.float32, requires_grad=True)

W = torch.randn(n_channels, n_out_channels, device='cuda',
                dtype=torch.float32, requires_grad=False)

x_t = x.clone().detach().transpose(-1, -2).contiguous()

torch_out = torch.matmul(x, W)
torch.cuda.synchronize()

start = time()
for i in range (1000):
    output = torch.ops.matmul.do_matmul(x_t, W)
    torch.cuda.synchronize()
end = time()

print ("do_matmul: ", end-start)

print (output[0], output[1])

start = time()
for i in range (1000):
    output = torch.ops.matmul.matmul_test(x_t, W)
    torch.cuda.synchronize()
end = time()

print ("test matmul: ", end-start)

start = time()
for i in range (1000):
    torch_out = torch.matmul(x, W)
    torch.cuda.synchronize()
end = time()

print ("torch: ", end-start)

print (torch_out[0], torch_out[1])


W = torch.randn(4, n_channels, n_out_channels, device='cuda',
                dtype=torch.float32, requires_grad=False)

path_weights = torch.ones(4, dtype=torch.float32)

torch.cuda.synchronize()
start = time()
for i in range (1000):
    torch_out = torch.ops.matmul.linear_test(x, W, path_weights)
    torch.cuda.synchronize()
end = time()

print ("linear fwd:", end - start)

print (torch_out[0])
print (torch_out[150])
