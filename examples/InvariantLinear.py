# Implementation of the linear layer
from time import time
from typing import List
from math import prod
import torch
from e3nn import o3
from mace_ops.ops.linear import Linear

torch.backends.cuda.matmul.allow_tf32 = False


class shape_irreps(torch.nn.Module):
    # code the reverse of reshape_irreps
    def __init__(self, irreps: o3.Irreps) -> None:
        super().__init__()
        self.irreps = irreps

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        # the reverse of reshape_irreps
        ix = 0
        out = []
        batch, _, _ = tensor.shape
        for mul, ir in self.irreps:
            d = ir.dim
            field = tensor[:, :, ix: ix + d]
            field = field.reshape(batch, mul * d)
            ix = ix + d
            out.append(field)
        return torch.cat(out, dim=-1)


class reshape_irreps(torch.nn.Module):
    def __init__(self, irreps: o3.Irreps) -> None:
        super().__init__()
        self.irreps = irreps

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        ix = 0
        out = []
        print(tensor.shape)
        batch, _ = tensor.shape
        for mul, ir in self.irreps:
            d = ir.dim
            field = tensor[:, ix: ix + mul * d]  # [batch, sample, mul * repr]
            ix += mul * d
            field = field.reshape(batch, mul, d)
            out.append(field)
        return torch.cat(out, dim=-1)


def _sum_tensors(xs: List[torch.Tensor], shape: torch.Size, like: torch.Tensor):
    if len(xs) > 0:
        out = xs[0]
        for x in xs[1:]:
            out = out + x
        return out
    return like.new_zeros(shape)


class LinearRef(torch.nn.Module):

    def __init__(self, irreps_in, irreps_out, e3nn_instructions, e3nn_weights):

        super().__init__()

        self.irreps_in = irreps_in
        self.irreps_out = irreps_out

        self.e3nn_instructions = e3nn_instructions
        self.e3nn_weights = e3nn_weights

        self.out_lmax = int(irreps_out.lmax)
        self.out_dim = int(irreps_out.dim / (self.out_lmax + 1) ** 2)

        self.instructions = []

        flat_weight_index = 0
        for ins in e3nn_instructions:
            path_nweight = prod(ins.path_shape)
            mul_ir_out = irreps_out[ins.i_out]
            # extract the weights for the current path
            w = e3nn_weights.narrow(-1, flat_weight_index, path_nweight)
            w = w.reshape(ins.path_shape)
            # 0 | 1 2 3 | 4 5 6
            start = ins.i_in ** 2
            end = start + (2 * ins.i_in + 1)

            self.instructions.append((start, end, w, ins.path_weight))

            flat_weight_index += path_nweight

    def forward(self, x):

        output = torch.zeros(x.shape[0], (self.out_lmax + 1) ** 2, self.out_dim,
                             device='cuda', dtype=torch.float32)

        for i, instruction in enumerate(self.instructions):

            start_l_idx, end_l_idx, weights, path_weight = instruction

            output[:, start_l_idx:end_l_idx, :] = path_weight * \
                torch.matmul(x[:, start_l_idx:end_l_idx, :], weights)

        return output

# INPUTS#
n_channels = 96
n_out_channels = 96
max_l = 3
nnodes = 5000

x = torch.randn(nnodes, n_channels*(max_l+1)**2,
                device='cuda', dtype=torch.float32, requires_grad=True)
## E3NN LINEAR##
irreps_in = o3.Irreps(
    (n_channels * o3.Irreps.spherical_harmonics(max_l))
    .sort()
    .irreps.simplify()
)
irreps_out = o3.Irreps(
    f"{n_out_channels}x0e + {n_out_channels}x1o + {n_out_channels}x2e + {n_out_channels}x3o")

print("IRREPS_IN, IRREPS_OUT")
print(irreps_in, irreps_out)

linear = o3.Linear(irreps_in=irreps_in, irreps_out=irreps_out).to('cuda')
### UTILS###

instructions = linear.instructions
ws = linear.weight

#need to reshape x so it has the right shape for the CUDA code
x_reshape = []
ix = 0
for mul, ir in irreps_in:
    field = x[:, ix: ix + mul * ir.dim]  # [batch, sample, mul * repr]
    field = field.reshape(
        x.shape[0], mul, ir.dim).transpose(-1, -2).contiguous()
    ix += mul * ir.dim
    x_reshape.append(field)
x_r = torch.cat(x_reshape, dim=1).contiguous()

grad_check_x_c = x_r.clone().detach().cuda().contiguous().requires_grad_(True)

linear_ref = LinearRef(irreps_in, irreps_out, instructions, ws)

##CUDA LINEAR##
linear_cuda = Linear(irreps_in, irreps_out, instructions, ws)


flat_weight_index = 0
for ins in instructions:
    path_nweight = prod(ins.path_shape)
    mul_ir_out = irreps_out[ins.i_out]
    # extract the weights for the current path
    w = ws.narrow(-1, flat_weight_index, path_nweight)
    w = w.reshape(ins.path_shape)
    # 0 | 1 2 3 | 4 5 6 
    start = ins.i_in ** 2
    end = start + (2 * ins.i_in + 1)

    out = ins.path_weight * torch.matmul(x_r[:, start:end, :], w)

    flat_weight_index += path_nweight

shared_weights = linear.shared_weights

irreps_in = linear.irreps_in
irreps_out = linear.irreps_out
batch_out = x.shape[0]
z = "" if shared_weights else "z"
out_list = []
size = x.shape[:-1]
outsize = size + (irreps_out.dim,)
# extract the different irreps in the input into a list aka: if x has l=0 with 2 channels and l=1 with 2 channels, x_list = [x0,x1]
# with x0.shape = (batch, 2) and x1.shape = (batch, 6)
x_list = [
    x.narrow(-1, i.start, mul_ir.dim).reshape(batch_out,
                                              *(()), mul_ir.mul, mul_ir.ir.dim)
    for i, mul_ir in zip(irreps_in.slices(), irreps_in)
]

flat_weight_index = 0
for ins in instructions:
    #print(ins)
    path_nweight = prod(ins.path_shape)
    mul_ir_out = irreps_out[ins.i_out]
    # extract the weights for the current path
    w = ws.narrow(-1, flat_weight_index, path_nweight)
    # print("w.shape before reshape: ", w.shape)
    flat_weight_index += path_nweight
    w = w.reshape((() if shared_weights else (-1,)) + () + ins.path_shape)
    # print("w.shape after reshape: ", w.shape)
    # print("x input shape: ", x_list[ins.i_in].shape)
    ein_out = torch.einsum(f"{z}uw,zui->zwi", w,
                           x_list[ins.i_in])  # do the contraction
    ein_out = ins.path_weight * ein_out  # apply the path weight
    # print("ein_out.shape: ", ein_out.shape)
    out_list += [ein_out.reshape(batch_out, *(), mul_ir_out.dim)]
out = [
    _sum_tensors(
        [out for ins, out in zip(instructions, out_list)
         if ins.i_out == i_out],
        shape=(batch_out, *(), mul_ir_out.dim),
        like=x,
    )
    for i_out, mul_ir_out in enumerate(irreps_out)
    if mul_ir_out.mul > 0
]
if len(out) > 1:
    out = torch.cat(out, dim=-1)
else:
    out = out[0]

out = out.reshape(outsize)

print(torch.allclose(linear(x), out, atol=1e-5))

out_reshape = []
ix = 0
for mul, ir in irreps_in:
    field = out[:, ix: ix + mul * ir.dim]  # [batch, sample, mul * repr]
    field = field.reshape(
        out.shape[0], mul, ir.dim).transpose(-1, -2).contiguous()
    ix += mul * ir.dim
    out_reshape.append(field)

out_t = torch.cat(out_reshape, dim=1).contiguous()


torch.cuda.synchronize()
start = time()
for i in range(1000):
    _ = linear(x)
    t = _.sum()
    t.backward()
    torch.cuda.synchronize()
end = time()

print (x.grad, x.grad.shape)

print("fwd e3nn linear:", end - start)

torch.cuda.cudart().cudaProfilerStart()
torch.cuda.synchronize()
start = time()
for i in range(1000):
    cuda_lin_out = linear_cuda(grad_check_x_c)
    t = cuda_lin_out.sum()
    t.backward()
    # torch.cuda.synchronize() -> not needed because of cudaStreamDestroy
end = time()

torch.cuda.cudart().cudaProfilerStop()

print (grad_check_x_c.grad, grad_check_x_c.shape)

print("fwd CUDA linear:", end - start)


linear_ref = linear_ref(grad_check_x_c)

print (linear_ref[-1])
print (cuda_lin_out[-1])

#print (linear_ref[-1])
#print (cuda_lin_out[-1])

#assert torch.allclose(linear_ref, cuda_lin_out, )

idx = torch.where(linear_ref - cuda_lin_out > 1e-4)

print (idx)

print (linear_ref[idx])
print (cuda_lin_out[idx])