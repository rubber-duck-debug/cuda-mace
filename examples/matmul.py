# Implementation of the linear layer
from time import time
import torch
from mace_ops import cuda


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
n_channels = 96
n_out_channels = 64

max_l = 3

nnodes = 4

x = torch.randn(nnodes, (max_l+1)**2, n_channels,
                device='cuda', dtype=torch.float32, requires_grad=True)

W = torch.randn(n_channels, n_out_channels, device='cuda',
                dtype=torch.float32, requires_grad=False)

W_T = W.clone().detach().transpose(-1, -2).cuda().contiguous()
x_wmma = x.clone().detach().requires_grad_(True).cuda().contiguous()
x_py = x.clone().detach().requires_grad_(True).cuda().contiguous().requires_grad_(True)

print(W)
print(W_T)

print(W.shape)
print(W_T.shape)

wmma_out = torch.ops.linear_wmma.matmul(x_wmma, W, W_T)

print(wmma_out[0])

wmma_loss = wmma_out.sum() ** 0.5
wmma_loss.backward()
print("x WMMA grad:")
print(x_wmma.grad[0])


torch_out = torch.matmul(x, W)

torch_loss = torch_out.sum() ** 0.5
torch_loss.backward()
print("x grad:")
print(x.grad[0])


output_fn = MatMul.apply(x_py, W)
t_fn = output_fn.sum() ** 0.5
t_fn.backward()
print(x_py.grad[0])
