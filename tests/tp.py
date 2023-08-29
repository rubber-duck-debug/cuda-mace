
import torch
from e3nn import o3
from mace_ops.cuda.instruction import Instruction, _normalize_instruction_path_weights
from mace_ops.cuda.irreps import Irreps
import numpy as np
from time import time

from mace.modules.irreps_tools import (
    tp_out_irreps_with_instructions
)

from mace.tools.scatter import scatter_sum


class TensorProduct(torch.nn.Module):

    def __init__(self, irreps_in1, irreps_in2, target_irreps, device="cuda", dtype=torch.float64):
        super().__init__()

        self.irreps_in1 = o3.Irreps(irreps_in1)
        self.irreps_in2 = o3.Irreps(irreps_in2)
        self.target_irreps = o3.Irreps(target_irreps)

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


if __name__ == "__main__":

    torch.set_printoptions(edgeitems=3)

    dtype = torch.float32

    benchmark = False

    nedges = 30000
    nnodes = 1000
    nfeatures = 96

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

    X = torch.randn(nedges, irreps1.dim, nfeatures,
                    requires_grad=True, device='cuda', dtype=dtype)
    Y = torch.randn(nedges, irreps2.dim, requires_grad=True,
                    device='cuda', dtype=dtype)

    tp_cuda = TensorProduct(
        irreps1, irreps2, target_irreps, device="cuda", dtype=dtype)

    out_ref = tp_cuda.forward(X, Y.unsqueeze(-1))

    output = torch.zeros(
        nnodes, out_ref.shape[1], out_ref.shape[2], device="cuda", dtype=dtype)

    output.index_add_(0, indices_cuda, out_ref)

    # print (output.shape)
    print(output[-1])

    mu1 = tp_cuda.mu1[tp_cuda.mu_3_sort]
    mu2 = tp_cuda.mu2[tp_cuda.mu_3_sort]
    mu3 = tp_cuda.mu3[tp_cuda.mu_3_sort]
    cg_coeffs = tp_cuda.cg_coeffs[tp_cuda.mu_3_sort]

    # print (mu1)
    # print (mu2)
    # print (mu3)

    last_val = 0
    last_idx = 0

    nelements = []
    for i in range(len(mu3)):
        if (mu3[i] != last_val):
            nelements.append(i - last_idx)
            last_idx = i
    nelements.append(len(mu3)-last_idx)

    csum = np.cumsum(nelements)
    # print (csum, len(csum))

    lsum = 0
    nwork_per_thread_y = len(mu3) / 4
    csum = np.insert(csum, 0, 0)

    ends = []
    for i in range(len(nelements)):
        lsum += nelements[i]
        if (lsum > nwork_per_thread_y):
            ends.append(i)
            lsum = 0
    ends.append(len(mu3))
    # print ("ends:", ends)

    for v in ends:
        if (v+1 < len(mu3)):
            print(mu3[v-1], mu3[v], mu3[v+1])

    indices_start = torch.tensor([0, 22, 44, 65]).int().cuda()
    nwork = torch.tensor([22, 22, 21, 21]).int().cuda()

    print(mu3)

    # tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
    #    18, 19, 19, 19, 20, 20, 21, 21, 22, 22, 22, 23, 23, 24, 24, 25, 25, 25,
    #    25, 26, 26, 26, 27, 27, 27, 27, 28, 28, 29, 29, 29, 30, 30, 30, 30, 31,
    #    31, 31, 32, 32, 32, 32, 33, 33, 33, 34, 34, 35, 35, 35, 35, 35, 36, 36,
    #    36, 36, 37, 37, 37, 38, 38, 38, 38, 39, 39, 39, 39, 39],

    print(indices_start)
    print(nwork)

    for i in range(len(indices_start)):

        for j in range(nwork[i]):
            idx = indices_start[i] + j
            print("thread: ", i, "index:", idx, "mu3: ", mu3[idx])

    # 192x0e+288x1o+288x2e+192x3o
    # 192x0e : 2 x 1 : 2
    # 288x1o : 3 x 3 : 9
    # 288x2e : 3 x 5 : 15
    # 192x3o : 2 x 7 : 14

    start = time()
    for i in range(1000):
        out = torch.ops.mace_ops_equivariant_tp.equivariant_outer_product_forward_v2(
            X,
            Y,
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
    torch.cuda.synchronize()
    end = time()
    print(end - start)

    out = torch.ops.mace_ops_equivariant_tp.equivariant_outer_product_forward_v2(
        X,
        Y,
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
        32, 2, 1)

    print(out[-1])
    idx = torch.where(out - output > 1e-4)

    print(idx)
    print(out[idx])
    print(output[idx])

    node_feats_irreps, edge_attrs_irreps, target_irreps = (
        o3.Irreps(f"{nfeatures}x0e + {nfeatures}x1o"),
        o3.Irreps("1x0e + 1x1o + 1x2e + 1x3o"),
        o3.Irreps(
            f"{nfeatures}x0e + {nfeatures}x1o + {nfeatures}x2e + {nfeatures}x3o"),
    )

    irreps_mid, instructions = tp_out_irreps_with_instructions(
        node_feats_irreps,
        edge_attrs_irreps,
        target_irreps,
    )

    print(instructions)

    # print ("RealAgnosticInteractionBlock: edge_attrs_irreps = ", self.edge_attrs_irreps)
    # print ("RealAgnosticInteractionBlock: irreps_mid = ", irreps_mid)

    print("irreps simplify")
    print(irreps_mid.simplify())

    conv_tp = o3.TensorProduct(
        node_feats_irreps,
        edge_attrs_irreps,
        irreps_mid,
        instructions=instructions,
        shared_weights=True,
        internal_weights=True,
    ).to("cuda")

    X_copy = X.clone().detach().cuda().requires_grad_(
        True).transpose(-1, -2).flatten(start_dim=1).float().contiguous()
    Y_copy = Y.clone().detach().cuda().requires_grad_(True).float()
    indices_cuda = indices_cuda.long()
    print(X.dtype, Y.dtype)
    print(node_feats_irreps.dim)
    print(edge_attrs_irreps.dim)

    start = time()
    for i in range(1000):
        mji = conv_tp(
            X_copy, Y_copy
        )
        message = scatter_sum(
            src=mji, index=indices_cuda, dim=0, dim_size=nnodes
        )  # [n_nodes, irreps]
        torch.cuda.synchronize()
    end = time()

    print(end - start)

    print(mji.shape)
    print(message.shape)
