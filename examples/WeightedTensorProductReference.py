import torch
import numpy as np

from torch.profiler import profile, record_function, ProfilerActivity
from typing import Tuple, List
from e3nn.util import prod
from e3nn import o3
from e3nn import o3
from e3nn_jax import Instruction, Irreps
from e3nn_jax._src.core_tensor_product import _normalize_instruction_path_weights

class TensorProductReference(torch.nn.Module):
    def __init__(
        self,
        irreps_in1,
        irreps_in2,
        target_irreps,
        nchannels,
        device="cpu",
    ):
        super().__init__()

        self.irreps_in1 = o3.Irreps(irreps_in1)
        self.irreps_in2 = o3.Irreps(irreps_in2)
        self.target_irreps = o3.Irreps(target_irreps)

        self.dim_out = 0
        self.device = device

        self.mu_list = []
        self.cg_sparse_list = []
        self.nchannels = nchannels

        instructions = []
        irreps_out = []

        l_channel = 0

        self.weight_indices = {}

        # Collect possible irreps and their instructions
        irreps_out_list: List[Tuple[int, o3.Irreps]] = []
        instructions = []
        for i, (mul, ir_in) in enumerate(irreps_in1):
            for j, (_, ir_edge) in enumerate(irreps_in2):
                for ir_out in ir_in * ir_edge:  # | l1 - l2 | <= l <= l1 + l2
                    if ir_out in target_irreps:
                        k = len(irreps_out_list)  # instruction index
                        irreps_out_list.append((self.nchannels, ir_out))
                        instructions.append((i, j, k, "uvu", True, self.nchannels))

        # We sort the output irreps of the tensor product so that we can simplify them
        # when they are provided to the second o3.Linear
        irreps_out = o3.Irreps(irreps_out_list)
        irreps_out, permut, _ = irreps_out.sort()
        self.irreps_out = irreps_out
        # Permute the output indexes of the instructions to match the sorted irreps:
        instructions = [
            Instruction(
                i_in1=i_in1,
                i_in2=i_in2,
                i_out=permut[i_out],
                connection_mode=mode,
                has_weight=train,
                path_weight=1.0,
                weight_std=None,
                first_input_multiplicity=mul,
                second_input_multiplicity=1,
                output_multiplicity=mul,
            )
            for i_in1, i_in2, i_out, mode, train, mul in instructions
        ]
        self.instructions = _normalize_instruction_path_weights(
            instructions,
            irreps_in1,
            irreps_in2,
            self.irreps_out,
            [1.0 for _ in self.irreps_in1],
            [1.0 for _ in self.irreps_in2],
            [1.0 for _ in self.irreps_out],
            irrep_normalization="component",
            path_normalization_exponent=1.0,  # path
            gradient_normalization_exponent=1.0,  # path
        )
        for ins in self.instructions:
            l1 = self.irreps_in1[ins.i_in1].ir.l
            l2 = self.irreps_in2[ins.i_in2].ir.l
            l3 = self.irreps_out[ins.i_out].ir.l

            cg = o3.wigner_3j(l1, l2, l3).to(self.device)

            # normalisation and weighting:
            cg = cg * ins.path_weight

            mu1, mu2, mu3 = cg.nonzero(as_tuple=True)
            cg_sparse = cg[(mu1, mu2, mu3)]

            sorted_indices = mu3.argsort()

            mu1 = mu1[sorted_indices]
            mu2 = mu2[sorted_indices]
            mu3 = mu3[sorted_indices]
            cg_sparse = cg_sparse[sorted_indices].cuda()

            self.mu_list += [(mu1, mu2, mu3)]
            self.cg_sparse_list += [cg_sparse]

        self.n_l_channels = l_channel
        self.weight_numel = sum(
            prod(ins.path_shape) for ins in instructions if ins.has_weight
        )

    def forward_unweighted_tp(self, x, y):
        all_outputs = []

        for i in range(x.shape[0]):  # loop over edges
            outputs = []
            for ins, (mu1, mu2, mu3), cg_sparse in zip(
                self.instructions, self.mu_list, self.cg_sparse_list
            ):
                ir_in1 = self.irreps_in1[ins.i_in1].ir
                ir_in2 = self.irreps_in2[ins.i_in2].ir
                ir_out = self.irreps_out[ins.i_out].ir
                offset1 = self.irreps_in1[: ins.i_in1].dim
                offset2 = self.irreps_in2[: ins.i_in2].dim
                cg_iteration = (
                    x[i, :, offset1 + mu1] * cg_sparse * y[i, :, offset2 + mu2]
                )
                output = torch.zeros(x.shape[1], ir_out.dim, device=self.device)
                output.index_add_(1, mu3, cg_iteration)

                # assert len(outputs) == ins.i_out, (len(outputs), ins.i_out)
                outputs.insert(ins.i_out, output)

            output_i = torch.cat(outputs, dim=-1)

            all_outputs.append(output_i)

        return torch.stack(all_outputs)

    def forward_weighted_tp(self, x, y, weights):
        all_outputs = []

        for i in range(x.shape[0]):  # loop over edges
            outputs = []
            for ins, (mu1, mu2, mu3), cg_sparse in zip(
                self.instructions, self.mu_list, self.cg_sparse_list
            ):
                ir_out = self.irreps_out[ins.i_out].ir
                offset1 = self.irreps_in1[: ins.i_in1].dim
                offset2 = self.irreps_in2[: ins.i_in2].dim
                cg_iteration = (
                    weights[i, ins.i_out * self.nchannels : (ins.i_out + 1) * self.nchannels, :]
                    * x[i, :, offset1 + mu1]
                    * cg_sparse
                    * y[i, :, offset2 + mu2]
                )
                output = torch.zeros(x.shape[1], ir_out.dim, device=self.device)
                output.index_add_(1, mu3, cg_iteration)

                # assert len(outputs) == ins.i_out, (len(outputs), ins.i_out)
                outputs.insert(ins.i_out, output)

            output_i = torch.cat(outputs, dim=-1)

            all_outputs.append(output_i)

        return torch.stack(all_outputs)

    def forward(self, x, y, weights=None):
        if weights is not None:
            return self.forward_weighted_tp(x, y, weights)
        else:
            return self.forward_unweighted_tp(x, y)

    def grad_dX2(self, x, y):

        output_grad = torch.zeros_like(y, device=self.device)

        for i in range(x.shape[0]):  # loop over edges

            for ins, (mu1, mu2, mu3), cg_sparse in zip(
                self.instructions, self.mu_list, self.cg_sparse_list
            ):
                ir_in1 = self.irreps_in1[ins.i_in1].ir
                ir_in2 = self.irreps_in2[ins.i_in2].ir
                ir_out = self.irreps_out[ins.i_out].ir

                offset1 = self.irreps_in1[: ins.i_in1].dim
                offset2 = self.irreps_in2[: ins.i_in2].dim

                l1 = self.irreps_in1[ins.i_in1].ir.l
                l2 = self.irreps_in2[ins.i_in2].ir.l
                l3 = self.irreps_out[ins.i_out].ir.l

                cg_iteration = x[i, :, offset1 + mu1] * cg_sparse

                output = torch.zeros(x.shape[1], ir_out.dim, device=self.device)

                output.index_add_(1, mu3, cg_iteration)

                test = torch.sum(
                    self.weights[:, self.weight_indices[l1, l2, l3]][:, None] * output,
                    dim=0,
                )

                output_grad[i, :, offset2 : offset2 + test.shape[0]] += test

                # output[i] += self.weights[:, self.weight_indices[l1, l2, l3]] * torch.sum(cg_iteration, dim=-1)

        return output_grad


n_edges = 2

nchannels = 3

irreps1, irreps2, target_irreps = (
    o3.Irreps(f"1x0e + 1x1o"),
    o3.Irreps("0e + 1o"),
    o3.Irreps(f"1x0e + 1x1o"),
)

tp_reference = TensorProductReference(
    irreps1, irreps2, target_irreps, nchannels, device="cuda"
)

n_edges = 1
X1 = torch.randn(n_edges, nchannels, (irreps1.lmax + 1) ** 2).cuda()
X2 = torch.randn(n_edges, 1, irreps2.dim).cuda()
out = tp_reference.forward(X1, X2)

weights = torch.randn(n_edges, tp_reference.weight_numel, 1).cuda()
out_weighted = tp_reference.forward(X1, X2, weights)



# Based on mir-group/nequip
from typing import Tuple, List
def tp_out_irreps_with_instructions(
    irreps1: o3.Irreps, irreps2: o3.Irreps, target_irreps: o3.Irreps
) -> Tuple[o3.Irreps, List]:
    trainable = True

    # Collect possible irreps and their instructions
    irreps_out_list: List[Tuple[int, o3.Irreps]] = []
    instructions = []
    for i, (mul, ir_in) in enumerate(irreps1):
        for j, (_, ir_edge) in enumerate(irreps2):
            for ir_out in ir_in * ir_edge:  # | l1 - l2 | <= l <= l1 + l2
                if ir_out in target_irreps:
                    k = len(irreps_out_list)  # instruction index
                    irreps_out_list.append((mul, ir_out))
                    instructions.append((i, j, k, "uvu", trainable))

    # We sort the output irreps of the tensor product so that we can simplify them
    # when they are provided to the second o3.Linear
    irreps_out = o3.Irreps(irreps_out_list)
    irreps_out, permut, _ = irreps_out.sort()

    # Permute the output indexes of the instructions to match the sorted irreps:
    instructions = [
        (i_in1, i_in2, permut[i_out], mode, train)
        for i_in1, i_in2, i_out, mode, train in instructions
    ]

    instructions = sorted(instructions, key=lambda x: x[2])

    return irreps_out, instructions

irreps1, target_irreps = o3.Irreps("3x0e + 3x1o"), o3.Irreps(f"3x0e + 3x1o")
irreps_out, instructions = tp_out_irreps_with_instructions(irreps1, irreps2, target_irreps)

tp_torch = o3.TensorProduct(irreps1, irreps2,irreps_out, instructions, shared_weights=False, internal_weights=False,).to("cuda")
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
        batch, _ = tensor.shape
        for mul, ir in self.irreps:
            d = ir.dim
            field = tensor[:, ix : ix + mul * d]  # [batch, sample, mul * repr]
            ix += mul * d
            field = field.reshape(batch, mul, d)
            out.append(field)
        return torch.cat(out, dim=-1)
X1_torch = shape_irreps(irreps1)(X1)
X2_torch = shape_irreps(irreps2)(X2)
out_2 = tp_torch(X1_torch, X2_torch, torch.ones((1, tp_torch.weight_numel), device="cuda"))
out_2_weighted = tp_torch(X1_torch, X2_torch, weights.squeeze(-1))
reshape_irreps(irreps_out)(out_2) - out
reshape_irreps(irreps_out)(out_2_weighted) - out_weighted