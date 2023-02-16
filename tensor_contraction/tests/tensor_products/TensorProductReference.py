import torch
import numpy as np

from torch.profiler import profile, record_function, ProfilerActivity
from typing import Tuple, List

from e3nn import o3
from e3nn_jax import Instruction, Irreps
from e3nn_jax._src.core_tensor_product import _normalize_instruction_path_weights

class TensorProductReference(torch.nn.Module):

    def __init__(self, irreps_in1, irreps_in2, target_irreps, nchannels, weights=None, weighted_tp=True, device="cpu"):
        super().__init__()
        
        self.irreps_in1 = o3.Irreps(irreps_in1)
        self.irreps_in2 = o3.Irreps(irreps_in2)
        self.target_irreps = o3.Irreps(target_irreps)

        self.dim_out = 0
        self.device = device
        
        self.mu_list = []
        self.cg_sparse_list = []
        
        instructions = []
        irreps_out = []
        
        l_channel = 0
        
        self.weight_indices = {}
        
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
                        
                        self.weight_indices[l1, l2, l3] = l_channel
                                    
                        l_channel +=1
                        
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
            cg_sparse = cg_sparse[sorted_indices].cuda()

            self.mu_list += [(mu1, mu2, mu3)]
            self.cg_sparse_list += [cg_sparse]
    
        self.n_l_channels = l_channel
        self.nchannels = nchannels
        
        if (weights == None):
            self.weights = torch.randn(nchannels, l_channel).cuda()
        else: 
            self.weights = weights
            
        self.weighted_tp = weighted_tp

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

                cg_iteration = x[i, :, offset1 + mu1] * cg_sparse * y[i, :, offset2 + mu2]

                output = torch.zeros(x.shape[1], ir_out.dim, device=self.device)

                output.index_add_(1, mu3, cg_iteration)

                assert len(outputs) == ins.i_out, (len(outputs), ins.i_out)
                
                outputs.append(output)

            output_i = torch.cat(outputs, dim=-1)

            all_outputs.append(output_i)

        return torch.stack(all_outputs)
    
    def forward_weighted_tp(self, x, y):
    
        output = torch.zeros(x.shape[0], x.shape[1], device=self.device)
        
        for i in range(x.shape[0]): # loop over edges

            for ins, (mu1, mu2, mu3), cg_sparse in zip(self.instructions, self.mu_list, self.cg_sparse_list):
                ir_in1 = self.irreps_in1[ins.i_in1].ir
                ir_in2 = self.irreps_in2[ins.i_in2].ir
                ir_out = self.irreps_out[ins.i_out].ir

                offset1 = self.irreps_in1[: ins.i_in1].dim
                offset2 = self.irreps_in2[: ins.i_in2].dim
                
                l1 = self.irreps_in1[ins.i_in1].ir.l
                l2 = self.irreps_in2[ins.i_in2].ir.l
                l3 = self.irreps_out[ins.i_out].ir.l
                
                cg_iteration = x[i, :, offset1 + mu1] * cg_sparse * y[i, :, offset2 + mu2]

                output[i] += self.weights[:, self.weight_indices[l1, l2, l3]] * torch.sum(cg_iteration, dim=-1)

        return output
    
    def forward(self, x, y):
        if (self.weighted_tp):
            return self.forward_weighted_tp(x, y)
        else:
            return self.forward_unweighted_tp(x, y)
        
    def grad_dX2(self, x, y):
        
        output_grad = torch.zeros_like(y, device=self.device)
        
        for i in range(x.shape[0]): # loop over edges

            for ins, (mu1, mu2, mu3), cg_sparse in zip(self.instructions, self.mu_list, self.cg_sparse_list):
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
            
                
                test = torch.sum(self.weights[:, self.weight_indices[l1, l2, l3]][:, None] * output, dim=0)
                
                output_grad[i, :, offset2: offset2+test.shape[0]] += test
                
                #output[i] += self.weights[:, self.weight_indices[l1, l2, l3]] * torch.sum(cg_iteration, dim=-1)

        return output_grad
        
    

if __name__ == "__main__":

    n_edges = 2

    nchannels = 256

    irreps1, irreps2, target_irreps = (
        o3.Irreps(f"0e + 1o"),
        o3.Irreps("0e + 1o + 2e + 3o"),
        o3.Irreps(f"0e + 1o + 2e + 3o"),
    )

    tp_reference = TensorProductReference(irreps1, irreps2, target_irreps, nchannels, device="cuda")

    n_edges = 2
    X1 = torch.randn(n_edges, nchannels, irreps1.dim).cuda()
    X2 = torch.randn(n_edges, 1, irreps2.dim).cuda()

    out = tp_reference.forward(X1, X2)

    print(out)
    print(out.shape)

    tp_reference.use_weights=True
    out_weighted = tp_reference.forward(X1, X2)

    print(out_weighted)
    print(out_weighted.shape)