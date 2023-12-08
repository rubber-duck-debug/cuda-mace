from math import prod
import torch
from mace_ops import cuda
from e3nn import o3
from typing import List

class Linear(torch.nn.Module):

    def __init__(self, 
                 irreps_in: o3.Irreps, 
                 irreps_out: o3.Irreps,
                 e3nn_instructions : List,
                 e3nn_weights: torch.Tensor):

        super().__init__()

        self.irreps_in = irreps_in
        self.irreps_out = irreps_out

        self.e3nn_instructions = e3nn_instructions
        self.e3nn_weights = e3nn_weights

        self.out_lmax = int(irreps_out.lmax)
        self.out_dim = int(irreps_out.dim / (self.out_lmax + 1) ** 2)

        self.l_start = []
        self.l_end = []
        self.path_weights = []
        self.weights = []

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

            self.l_start.append(start)
            self.l_end.append(end)
            self.path_weights.append(ins.path_weight)
            self.weights.append(w.clone().detach())

            flat_weight_index += path_nweight

        self.l_start = torch.tensor(self.l_start).int().cuda()
        self.l_end = torch.tensor(self.l_end).int().cuda()
        self.weights = torch.stack(self.weights).contiguous().cuda().float()
        self.weights_transposed = self.weights.clone().detach().transpose(-1, -2).contiguous().cuda()
        self.path_weights = torch.tensor(self.path_weights).float()

    def forward(self, x: torch.Tensor):
        return torch.ops.linear_wmma.linear(x, self.weights, self.weights_transposed)
    

class ElementalLinear(torch.nn.Module):

    def __init__(self, irreps_in, irreps_out, e3nn_instructions, e3nn_weights, num_elements):

        super().__init__()
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

            #print (ins, w.shape)
            self.instructions.append((start, end, w, ins.path_weight))

            flat_weight_index += path_nweight
        
        self.weights = torch.zeros(self.num_elements, 4, w.shape[-2], w.shape[-1], dtype=torch.float32, device='cuda')
        
        for i, ins in enumerate(self.instructions):
            start_l_idx, end_l_idx, w, path_weight = ins
            self.weights[:, i, ... ] = w
            
        self.weights_transposed = self.weights.clone().detach().transpose(-1, -2).contiguous().cuda()
            
    def forward(self, x, y):
        # x : [batch,  num_l, num_channels]
        # y : [batch, num_elements]
        return torch.ops.linear_wmma.elemental_linear(x, self.weights, self.weights_transposed, y)
        
        