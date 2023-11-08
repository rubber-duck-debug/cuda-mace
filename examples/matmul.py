# Implementation of the linear layer
from time import time
import torch
from mace_ops import cuda

torch.backends.cuda.matmul.allow_tf32 = True


def get_gflops(time_in_ms):
    nops = 2 * n_channels*n_out_channels*nnodes * ((max_l+1)**2)
    return 1.0e-9 * nops / (time_in_ms / 1000.0)


class MatMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weights):

        ctx.save_for_backward(x, weights)
        return torch.matmul(x, weights)

    @staticmethod
    def backward(ctx, grad_output):

        print(grad_output.shape)

        x, weights = ctx.saved_tensors

        grad_w = torch.zeros(n_channels, n_out_channels, device='cuda')

        for i in range(x.shape[0]):
            grad_w += torch.matmul(x[i].transpose(-1, -2), grad_output[i])

        return torch.matmul(grad_output, weights.transpose(-1, -2).contiguous()), grad_w


# INPUTS#
n_channels = 128
n_out_channels = 128

max_l = 3

nnodes = 1000

x = torch.randn(nnodes, (max_l+1)**2, n_channels,
                device='cuda', dtype=torch.float32, requires_grad=True)

W = torch.randn(n_channels, n_out_channels, device='cuda',
                dtype=torch.float32, requires_grad=False)

W_T = W.clone().detach().transpose(-1, -2).cuda().contiguous()

print("torch double", torch.matmul(x.double(), W.double())[0])

wmma_out = torch.ops.linear_wmma.matmul_fwd(x, W, W_T, True)

print("cuda_out", wmma_out[0])

torch_out = torch.matmul(x, W)
torch.cuda.synchronize()

start = time()
for i in range(1000):
    torch_out = torch.matmul(x, W)
    torch.cuda.synchronize()
end = time()

print("torch matmul", torch_out[0])
print("torch matmul:", end - start, get_gflops(end-start))

start = time()
for i in range(1000):
    wmma_out = torch.ops.linear_wmma.matmul_fwd(x, W, W_T, False)
    torch.cuda.synchronize()
end = time()

print(wmma_out[0])
print("wmma without correction:", end - start, get_gflops(end-start))
torch.cuda.synchronize()

start = time()
for i in range(1000):
    wmma_out = torch.ops.linear_wmma.matmul_base(x, W, False)
    torch.cuda.synchronize()
end = time()

print(wmma_out[0])
print("wmma without correction (base):", end - start, get_gflops(end-start))

torch.cuda.synchronize()

start = time()
for i in range(1000):
    wmma_out = torch.ops.linear_wmma.matmul_fwd(x, W, W_T, True)

    torch.cuda.synchronize()
end = time()

print(wmma_out[0])
print("wmma with correction:", end - start, get_gflops(end-start))


start = time()
for i in range(1000):
    matmul_out = torch.ops.linear_wmma.matmul(x, W)
    torch.cuda.synchronize()
end = time()

print(matmul_out[0])
print("simple matmul:", end - start, get_gflops(end-start))
