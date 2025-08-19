from math import prod
import torch
from e3nn import o3
from typing import List


class Linear(torch.nn.Module):

    def __init__(self, linear: o3.Linear):

        super().__init__()

        self.cuda_obj = torch.classes.linear_wmma.Linear()

        self.irreps_out = linear.irreps_out
        self.irreps_in = linear.irreps_in.simplify()
        self.e3nn_instructions = linear.instructions  # e3nn_instructions
        self.e3nn_weights = linear.weight.clone().detach()  # e3nn_weights
        
        self.out_lmax = int(self.irreps_out.lmax)
        self.out_dim = int(self.irreps_out.dim / (self.out_lmax + 1) ** 2)

        l_start = []
        l_end = []
        path_weights = []
        weights = []

        flat_weight_index = 0

        for ins in self.e3nn_instructions:
            path_nweight = prod(ins.path_shape)
            mul_ir_out = self.irreps_out[ins.i_out]
            # extract the weights for the current path
            w = self.e3nn_weights.narrow(-1, flat_weight_index, path_nweight)
            w = w.reshape(ins.path_shape)
            # 0 | 1 2 3 | 4 5 6
            start = ins.i_in ** 2
            end = start + (2 * ins.i_in + 1)

            l_start.append(start)
            l_end.append(end)
            path_weights.append(ins.path_weight)
            weights.append(w.clone().detach())

            flat_weight_index += path_nweight

        self.register_buffer("weights", torch.stack(
            weights).contiguous().cuda().float())

        self.register_buffer("weights_transposed",  self.weights.clone(
        ).detach().transpose(-1, -2).contiguous().cuda())

    def forward(self, x: torch.Tensor):
        return self.cuda_obj.forward(x, self.weights, self.weights_transposed)


class ElementalLinear(torch.nn.Module):

    def __init__(self, irreps_in, irreps_out, e3nn_instructions, e3nn_weights, num_elements):

        super().__init__()

        self.cuda_obj = torch.classes.linear_wmma.ElementalLinear()

        self.num_elements = num_elements
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
            w = w.reshape([-1] + list(ins.path_shape))
            # 0 | 1 2 3 | 4 5 6
            start = ins.i_in ** 2
            end = start + (2 * ins.i_in + 1)

            # print (ins, w.shape)
            self.instructions.append((start, end, w, ins.path_weight))

            flat_weight_index += path_nweight

        weights = torch.zeros(
            self.num_elements, 4, w.shape[-2], w.shape[-1], dtype=torch.float32, device='cuda')

        for i, ins in enumerate(self.instructions):
            start_l_idx, end_l_idx, w, path_weight = ins
            weights[:, i, ...] = w

        self.register_buffer("weights", weights)
        self.register_buffer("weights_transposed", self.weights.clone(
        ).detach().transpose(-1, -2).contiguous().cuda())

    def forward(self, x, y):
        # x : [batch,  num_l, num_channels]
        # y : [batch, num_elements]
        return self.cuda_obj.forward(x, self.weights, self.weights_transposed, y)
