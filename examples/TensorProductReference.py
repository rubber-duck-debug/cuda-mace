import torch
import numpy as np

from typing import Tuple, List

from e3nn import o3
from mace_ops.cuda.instruction import Instruction, _normalize_instruction_path_weights
from mace_ops.cuda.irreps import Irreps

class TensorProductReference(torch.nn.Module):

    def __init__(self, irreps_in1, irreps_in2, target_irreps, nchannels, device="cuda", dtype=torch.float32):
        super().__init__()
        
        self.irreps_in1 = o3.Irreps(irreps_in1)
        self.irreps_in2 = o3.Irreps(irreps_in2)
        self.target_irreps = o3.Irreps(target_irreps)

        self.dim_out = 0
        self.device = device
        self.dtype = dtype

        self.mu_list = []
        self.cg_sparse_list = []
        
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
            cg_sparse = cg_sparse[sorted_indices].type(self.dtype).cuda()

            self.mu_list += [(mu1, mu2, mu3)]
            self.cg_sparse_list += [cg_sparse]
    
        self.nchannels = nchannels


    def forward_unweighted_tp(self, x, y):
        all_outputs = []
        
        for i in range(x.shape[0]):  # loop over edges
            outputs = []

            for ins, (mu1, mu2, mu3), cg_sparse in zip(self.instructions, self.mu_list, self.cg_sparse_list):
                ir_in1 = self.irreps_in1[ins.i_in1].ir
                ir_in2 = self.irreps_in2[ins.i_in2].ir
                ir_out = self.irreps_out[ins.i_out].ir

                offset1 = self.irreps_in1[: ins.i_in1].dim
                offset2 = self.irreps_in2[: ins.i_in2].dim
                offset3 = self.irreps_out[: ins.i_out].dim

                cg_iteration = x[i,offset1 + mu1, :] * cg_sparse[:, None] * y[i, offset2 + mu2, :]

                output = torch.zeros(ir_out.dim, x.shape[2] , device=self.device, dtype=self.dtype)
                output.index_add_(0, mu3, cg_iteration)

                assert len(outputs) == ins.i_out, (len(outputs), ins.i_out)
             
                outputs.append(output)

            output_i = torch.cat(outputs, dim=0)

            all_outputs.append(output_i)

        return torch.stack(all_outputs)
    
    
    def forward(self, x, y):
        return self.forward_unweighted_tp(x, y)

if __name__ == "__main__":

    n_edges = 500

    nchannels = 128

    irreps1, irreps2, target_irreps = (
        o3.Irreps(f"0e + 1o"),
        o3.Irreps("0e + 1o + 2e + 3o"),
        o3.Irreps(f"0e + 1o + 2e + 3o"),
    )

    tp_reference = TensorProductReference(irreps1, irreps2, target_irreps, nchannels, device="cuda")

    X1 = torch.randn(n_edges, irreps1.dim, nchannels).cuda()
    X2 = torch.randn(n_edges, irreps2.dim, 1).cuda()

    out = tp_reference.forward(X1, X2)

    #print (out)