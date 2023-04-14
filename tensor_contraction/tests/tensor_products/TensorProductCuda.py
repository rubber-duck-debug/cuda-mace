import torch
import numpy as np
from torch.utils import cpp_extension

from torch.profiler import profile, record_function, ProfilerActivity
from typing import Tuple, List

import torch.utils.benchmark as benchmark
from torch.utils.cpp_extension import load
from TensorProductReference import TensorProductReference

from e3nn import o3
from e3nn_jax import Instruction, Irreps
from e3nn_jax._src.core_tensor_product import _normalize_instruction_path_weights

tensor_product_cuda = load(
    'tensor_product_cuda', ['../../cuda/tensor_product_kernel.cu'], verbose=True, extra_cflags=['-O3'], extra_cuda_cflags=['-O3'])

class WeightedTPFunction(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, x, mu_1, y, mu_2, cg_coeffs, weights, weight_indices):

        ctx.save_for_backward(x, y, mu_1, mu_2, cg_coeffs, weights, weight_indices)
        
        res = tensor_product_cuda.weighted_forward(x, mu_1, y, mu_2, cg_coeffs, weights, weight_indices)
        
        return res[0]
        
    @staticmethod
    def backward (ctx, grad_input):
        
        print ("--grad_input--")
        print (grad_input)
        x, y, mu_1, mu_2, cg_coeffs, weights, weight_indices = ctx.saved_tensors
        
        grad_X1, = tensor_product_cuda.weighted_backward_dX1(x, mu_1, y, mu_2, cg_coeffs, weights, weight_indices, grad_input.contiguous())
        
        grad_X2, = tensor_product_cuda.weighted_backward_dX2(x, mu_1, y, mu_2, cg_coeffs, weights, weight_indices, grad_input.contiguous())
        
        
        return grad_X1, grad_X2, None, None, None, None, None
    
class TensorProductCuda(torch.nn.Module):

  def __init__(self, irreps_in1, irreps_in2, target_irreps, nchannels, weights=None,weighted_tp=True, device="cpu"):
    super().__init__()
    
    self.irreps_in1 = o3.Irreps(irreps_in1)
    self.irreps_in2 = o3.Irreps(irreps_in2)
    self.target_irreps = o3.Irreps(target_irreps)
    
    self.device = device
    self.weighted_tp = weighted_tp
    
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
    
    mu_1 = []
    mu_2 = []
    mu_3 = []
    cg_coeffs = []
    weight_indices = []
    
    l_channel = 0
    
    for ins in self.instructions:
        l1 = self.irreps_in1[ins.i_in1].ir.l
        l2 = self.irreps_in2[ins.i_in2].ir.l
        l3 = self.irreps_out[ins.i_out].ir.l

        offset1 = self.irreps_in1[: ins.i_in1].dim
        offset2 = self.irreps_in2[: ins.i_in2].dim
        offset3 = self.irreps_in2[: ins.i_out].dim
        
        print (l1, l2, l3, offset1, offset2, offset3)
        
        cg = o3.wigner_3j(l1, l2, l3).to(self.device)

                  
        # normalisation and weighting:
        cg = cg * ins.path_weight

        mu1, mu2, mu3 = cg.nonzero(as_tuple=True)
        
        cg_sparse = cg[(mu1, mu2, mu3)]
        
        mu1 = mu1 + offset1
        mu2 = mu2 + offset2
        mu3 = mu3 + offset3

        sorted_indices = mu3.argsort()

        mu1 = mu1[sorted_indices]
        mu2 = mu2[sorted_indices]
        mu3 = mu3[sorted_indices]
        cg_sparse = cg_sparse[sorted_indices]

        
        mu_1.append(mu1)
        mu_2.append(mu2)
        mu_3.append(mu3)

        cg_coeffs.append(cg_sparse)

        weight_idxs = torch.ones_like(mu1)
        
        weight_indices.append(weight_idxs * l_channel)
        
        l_channel += 1

    
    if (weights == None):
        self.weights = torch.randn(nchannels, l_channel).cuda()
    else:
        self.weights = weights
        
    self.weight_indices = torch.cat(weight_indices).int().cuda()

    self.mu_1 = torch.cat(mu_1).int()
    self.mu_2 = torch.cat(mu_2).int()
    self.mu_3 = torch.cat(mu_3).int()

    print ("--- mu ---")

    print (self.mu_1)
    print (self.mu_2)
    print (self.mu_3)

    self.cg_coeffs = torch.cat(cg_coeffs)

    print ("---cg coeffs---")
    print ('coeffs')
    print (self.cg_coeffs)


  def forward(self,x,y):
    
    if (self.weighted_tp):
        return WeightedTPFunction.apply(x, self.mu_1, y, self.mu_2, self.cg_coeffs, self.weights, self.weight_indices)
    else:
        raise Exception ("unweighted TP backwards is not supported yet")
    return None
  


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

def compute_total_l_channels(lmax):
    sum_l = 1

    for l in range (1, lmax+ 1):
        sum_l += (2 * l) + 1

    return sum_l

if __name__ == "__main__":

    torch.set_printoptions(edgeitems=6)
    
    nchannels=1

    l1 = 1
    l2 = 3
    n_l_channels = 10
    n_edges = 2
    
    X1 = torch.randn(n_edges, nchannels, (l1 + 1)**2, requires_grad=True, device='cuda')
    X2 = torch.randn(n_edges, 1, (l2 + 1)**2, requires_grad=True, device='cuda')
    weights = torch.rand(nchannels, n_l_channels, device='cuda')
    
    irreps1, irreps2, target_irreps = (
        o3.Irreps(f"0e + 1o"),
        o3.Irreps("0e + 1o + 2e + 3o"),
        o3.Irreps(f"0e + 1o + 2e + 3o"),
    )

    tp_cuda = TensorProductCuda(irreps1,irreps2,target_irreps,nchannels,weights, device="cuda")
    tp_reference = TensorProductReference(irreps1,irreps2,target_irreps,nchannels, weights, device="cuda")

    tp_reference.weighted_tp=True
    tp_cuda.weighed_tp = True

    X1_r = X1
    X2_r = X2

    cuda_weighted = tp_cuda.forward(X1, X2)
    
    s_cuda = cuda_weighted.sum()
    s_cuda.backward()
    
    reference_weighted = tp_reference.forward(X1_r, X2_r)
    
    s_reference = reference_weighted.sum()
    s_reference.backward()

    #print ("X1 grad diff")
    #print (X1.grad - X1_r.grad)
    
    

    #print ("X2 grad diff")
    #print (X2.grad - X2_r.grad)
    
    # print ("CUDA vs Reference weighted TP difference")
    # print (output_weighted - reference_weighted)
    # print (output_weighted)

    t0 = benchmark.Timer(
        stmt='tp(X1, X2)',
        globals={'X1': X1, 'X2': X2, "tp": tp_cuda.forward})

    print("CUDA TP (weights)", t0.timeit(1000))
    
    grad_input = torch.rand(cuda_weighted.shape).cuda()
    
    grad_out_X1 = tensor_product_cuda.weighted_backward_dX1(X1, tp_cuda.mu_1, X2, tp_cuda.mu_2, tp_cuda.cg_coeffs, tp_cuda.weights, tp_cuda.weight_indices, grad_input)
    grad_out_X2 = tensor_product_cuda.weighted_backward_dX2(X1, tp_cuda.mu_1, X2, tp_cuda.mu_2, tp_cuda.cg_coeffs, tp_cuda.weights, tp_cuda.weight_indices, grad_input)
    print (grad_out_X2[0].shape)
    
    print (grad_out_X2[0])
    
    t0_backward = benchmark.Timer(
        stmt='tp(X1, mu1, X2, mu2, coeffs, weights, weight_indices, grad_input)',
        globals={'X1': X1, 'X2': X2, 'mu1': tp_cuda.mu_1, 'mu2': tp_cuda.mu_2, 'coeffs': tp_cuda.cg_coeffs, 'weights': tp_cuda.weights, 
                 'weight_indices': tp_cuda.weight_indices, 'grad_input': grad_input, "tp": tensor_product_cuda.weighted_backward_dX1})


    print("CUDA TP backward X1 (weights)", t0_backward.timeit(1000))
    
    t0_backward = benchmark.Timer(
        stmt='tp(X1, mu1, X2, mu2, coeffs, weights, weight_indices, grad_input)',
        globals={'X1': X1, 'X2': X2, 'mu1': tp_cuda.mu_1, 'mu2': tp_cuda.mu_2, 'coeffs': tp_cuda.cg_coeffs, 'weights': tp_cuda.weights, 
                 'weight_indices': tp_cuda.weight_indices, 'grad_input': grad_input, "tp": tensor_product_cuda.weighted_backward_dX2})


    print("CUDA TP backward X2 (weights)", t0_backward.timeit(1000))

    import e3nn_jax as e3nn

    outj = e3nn.tensor_product(
        e3nn.IrrepsArray(irreps1, X1.cpu().detach().numpy()),
        e3nn.IrrepsArray(irreps2, X2.cpu().detach().numpy()),
        filter_ir_out=e3nn.Irreps(target_irreps),
    )
    
    out = tp_reference.grad_dX2(X1, X2)
    
    print (X2_r.grad)
    print (X2_r.grad.shape)
    
    print (out)
    