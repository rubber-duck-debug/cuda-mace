
import torch
from e3nn import o3
from mace_ops.cuda.instruction import Instruction, _normalize_instruction_path_weights
from mace_ops.cuda.irreps import Irreps
import numpy as np
from time import time
from e3nn.util import prod
from mace.modules.irreps_tools import (
    tp_out_irreps_with_instructions
)

from mace.tools.scatter import scatter_sum


class TensorProduct(torch.nn.Module):

    def __init__(self, irreps_in1, irreps_in2, target_irreps, nchannels, device="cuda", dtype=torch.float64):
        super().__init__()

        self.irreps_in1 = o3.Irreps(irreps_in1)
        self.irreps_in2 = o3.Irreps(irreps_in2)
        self.target_irreps = o3.Irreps(target_irreps)
        self.nchannels = nchannels

        self.device = device
        self.dtype = dtype

        instructions = []
        irreps_out = []

        for i, (mul, ir_in) in enumerate(self.irreps_in1):
            for j, (_, ir_edge) in enumerate(self.irreps_in2):
                for ir_out in ir_in * ir_edge:  # | l1 - l2 | <= l <= l1 + l2
                    if ir_out in target_irreps:

                        l1 = ir_in.l
                        l2 = ir_edge.l
                        l3 = ir_out.l

                        instructions.append(
                            Instruction(
                                i_in1=i,
                                i_in2=j,
                                i_out=len(instructions),
                                connection_mode="uvu",
                                has_weight=False,
                                path_weight=1.0,
                                weight_std=None,
                                first_input_multiplicity=mul,
                                second_input_multiplicity=1,
                                output_multiplicity=mul,
                            )
                        )
                        irreps_out.append((mul, ir_out))

        self.irreps_out = Irreps(irreps_out)

        print(self.irreps_out)

        self.instructions = _normalize_instruction_path_weights(
            instructions,
            self.irreps_in1,
            self.irreps_in2,
            self.irreps_out,
            [1.0 for _ in self.irreps_in1],
            [1.0 for _ in self.irreps_in2],
            [1.0 for _ in self.irreps_out],
            irrep_normalization="component",
            path_normalization_exponent=1.0,  # path
            gradient_normalization_exponent=1.0,  # path
        )

        mu_1 = []
        mu_2 = []
        mu_3 = []
        cg_coeffs = []

        self.mu_list = []
        self.cg_sparse_list = []
        self.weight_index_list = []

        for i, ins in enumerate(self.instructions):

            l1 = self.irreps_in1[ins.i_in1].ir.l
            l2 = self.irreps_in2[ins.i_in2].ir.l
            l3 = self.irreps_out[ins.i_out].ir.l

            offset1 = self.irreps_in1[: ins.i_in1].dim
            offset2 = self.irreps_in2[: ins.i_in2].dim
            offset3 = self.irreps_out[: ins.i_out].dim

            print(i, ins)
            print(l1, l2, l3)
            print(ins.i_in1, ins.i_in2, ins.i_out)

            cg = o3.wigner_3j(l1, l2, l3).to(self.device).type(dtype)

            # normalisation and weighting:
            cg = cg * ins.path_weight

            mu1, mu2, mu3 = cg.nonzero(as_tuple=True)

            cg_sparse = cg[(mu1, mu2, mu3)]

            self.mu_list += [(mu1, mu2, mu3)]
            self.cg_sparse_list += [cg_sparse]

            for j in range(mu1.shape[0]):
                self.weight_index_list.append(ins.i_out)

            mu1 = mu1 + offset1
            mu2 = mu2 + offset2
            mu3 = mu3 + offset3

            mu_1.append(mu1)
            mu_2.append(mu2)
            mu_3.append(mu3)

            cg_coeffs.append(cg_sparse)

        self.mu1 = torch.cat(mu_1).cuda().type(torch.int32)
        self.mu2 = torch.cat(mu_2).cuda().type(torch.int32)
        self.mu3 = torch.cat(mu_3).cuda().type(torch.int32)
        self.weight_index_list = torch.tensor(
            self.weight_index_list).cuda().type(torch.int32)

        # count the size of the output
        nmax1 = self.mu1.max().item() + 1
        nmax2 = self.mu2.max().item() + 1
        nmax3 = self.mu3.max().item() + 1

        self.nmax3 = nmax3

        print("nmax:", nmax1, nmax2, nmax3)

        self.cg_coeffs = torch.cat(cg_coeffs).type(self.dtype).cuda()

        self.mu_1_sort = torch.argsort(
            self.mu1).type(torch.int32).cuda().long()
        self.mu_2_sort = torch.argsort(
            self.mu2).type(torch.int32).cuda().long()
        self.mu_3_sort = torch.argsort(
            self.mu3).type(torch.int32).cuda().long()

        self.weight_numel = sum(
            prod((self.nchannels, ins.path_shape[-1])) for ins in self.instructions
        )

        print(self.weight_numel)

    def forward(self, x, y):
        all_outputs = []

        for i in range(x.shape[0]):  # loop over edges
            outputs = []

            for ins, (mu1, mu2, mu3), cg_sparse in zip(self.instructions, self.mu_list, self.cg_sparse_list):

                mu1 = mu1.cuda()
                mu2 = mu2.cuda()
                mu3 = mu3.cuda()
                cg_sparse = cg_sparse.cuda()

                ir_in1 = self.irreps_in1[ins.i_in1].ir
                ir_in2 = self.irreps_in2[ins.i_in2].ir
                ir_out = self.irreps_out[ins.i_out].ir

                offset1 = self.irreps_in1[: ins.i_in1].dim
                offset2 = self.irreps_in2[: ins.i_in2].dim
                offset3 = self.irreps_out[: ins.i_out].dim

                cg_iteration = x[i, offset1 + mu1, :] * \
                    cg_sparse[:, None] * y[i, offset2 + mu2, :]

                output = torch.zeros(
                    ir_out.dim, x.shape[2], device=self.device, dtype=self.dtype)
                output.index_add_(0, mu3, cg_iteration)

                assert len(outputs) == ins.i_out, (len(outputs), ins.i_out)

                outputs.append(output)

            output_i = torch.cat(outputs, dim=0)

            all_outputs.append(output_i)

        return torch.stack(all_outputs)


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


if __name__ == "__main__":

    torch.set_printoptions(edgeitems=3)

    dtype = torch.float32

    benchmark = False

    nedges = 3000
    nnodes = 100
    nfeatures = 32

    irreps1, irreps2, target_irreps = (
        o3.Irreps(f"0e + 1o"),
        o3.Irreps("0e + 1o + 2e + 3o"),
        o3.Irreps(f"0e + 1o + 2e + 3o"),
    )

    print("nnodes: ", nnodes)
    print("nfeatures: ", nfeatures)
    print("irreps1: ", irreps1, irreps1.dim)
    print("irreps2: ", irreps2, irreps2.dim)
    print("target_irreps: ", target_irreps, target_irreps.dim)
    print("dtype: ", dtype)

    indices = torch.sort(torch.randint(nnodes, (nedges,), device='cuda'))[0]

    indices_cuda = indices.cuda().int()

    neighbour_cuda = torch.ops.mace_ops_equivariant_tp.calculate_neighbours(
        indices_cuda, nnodes, 64)

    tp_cuda = TensorProduct(
        irreps1, irreps2, target_irreps, nfeatures, device="cuda", dtype=dtype)

    X1 = torch.randn(nedges, nfeatures, (irreps1.lmax + 1) ** 2).cuda()
    X2 = torch.randn(nedges, 1, irreps2.dim).cuda()
    weights = torch.randn(nedges, tp_cuda.weight_numel).cuda()

    node_feats_irreps, edge_attrs_irreps, target_irreps = (
        o3.Irreps(f"{nfeatures}x0e + {nfeatures}x1o"),
        o3.Irreps("0e + 1o + 2e + 3o"),
        o3.Irreps(
            f"{nfeatures}x0e + {nfeatures}x1o + {nfeatures}x2e + {nfeatures}x3o"),
    )

    irreps_mid, instructions = tp_out_irreps_with_instructions(
        node_feats_irreps,
        edge_attrs_irreps,
        target_irreps,
    )

    mu1 = tp_cuda.mu1#[tp_cuda.mu_3_sort]
    mu2 = tp_cuda.mu2#[tp_cuda.mu_3_sort]
    mu3 = tp_cuda.mu3#[tp_cuda.mu_3_sort]

    cg_coeffs = tp_cuda.cg_coeffs#[tp_cuda.mu_3_sort]

    weight_indices = tp_cuda.weight_index_list#[tp_cuda.mu_3_sort]

    indices_start = torch.tensor([0, 22, 44, 65]).int().cuda()
    nwork = torch.tensor([22, 22, 21, 21]).int().cuda()

    print(weight_indices)
    print(weights.shape)

    out_unweighted = torch.ops.mace_ops_equivariant_tp.equivariant_outer_product_forward(
        X1.transpose(-1, -2),
        X2.squeeze(1),
        indices_cuda,
        neighbour_cuda,
        mu1,
        mu2,
        mu3,
        cg_coeffs,
        indices_start,
        nwork,
        tp_cuda.nmax3,
        nnodes,
        32, 4, 1)

    out_weighted = torch.ops.mace_ops_equivariant_tp.weighted_equivariant_outer_product_forward(
        X1.transpose(-1, -2),
        X2.squeeze(1),
        indices_cuda,
        neighbour_cuda,
        mu1,
        mu2,
        mu3,
        cg_coeffs,
        indices_start,
        nwork,
        weight_indices,
        weights,
        tp_cuda.nmax3,
        nnodes,
        32, 4, 1)

    torch.cuda.synchronize()

    print(out_weighted[0])

    conv_tp = o3.TensorProduct(
        node_feats_irreps,
        edge_attrs_irreps,
        irreps_mid,
        instructions=instructions,
        shared_weights=False,
        internal_weights=False,
    ).to('cuda')

    X1_torch = shape_irreps(node_feats_irreps)(X1)
    X2_torch = shape_irreps(edge_attrs_irreps)(X2)

    mji = conv_tp(
        X1_torch, X2_torch, weights
    )

    message = scatter_sum(
        src=mji, index=indices_cuda.long(), dim=0, dim_size=nnodes
    )  # [n_nodes, irreps]

    message = reshape_irreps(irreps_mid)(message)

    print(out_weighted[0][:, -1], out_weighted[0].shape)
    print(message[0].transpose(-1, -2)[:, -1], message[0].shape)
    
    



    mji = conv_tp(
        X1_torch, X2_torch, torch.ones_like(weights)
    )

    message = scatter_sum(
        src=mji, index=indices_cuda.long(), dim=0, dim_size=nnodes
    )  # [n_nodes, irreps]

    message = reshape_irreps(irreps_mid)(message)

    print(out_unweighted[0][:, -1], out_unweighted[0].shape)
    print(message[0].transpose(-1, -2)[:, -1], message[0].shape)
