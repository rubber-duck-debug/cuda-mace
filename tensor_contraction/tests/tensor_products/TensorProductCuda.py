import torch
import numpy as np
from torch.utils import cpp_extension
from e3nn import o3
from torch.profiler import profile, record_function, ProfilerActivity
from typing import Tuple, List

from torch.utils.cpp_extension import load
from TensorProductReference import TensorProductReference

tensor_product_cuda = load(
    'tensor_product_cuda', ['../../cuda/tensor_product_kernel.cu'], verbose=True, extra_cflags=['-O2'], extra_cuda_cflags=['-O2'])

class TensorProductCuda(torch.nn.Module):

  def __init__(self, irreps1, irreps2, target_irreps, nchannels, weights=None, device="cpu"):
    super().__init__()
    self.irreps1 = irreps1
    self.irreps2 = irreps2
    self.target_irreps = target_irreps
    self.cg_dict = {}
    self.mus_dict = {}
    self.dim_out = 0
    self.device = device

    mu_1 = []
    mu_2 = []
    mu_3 = []

    cg_coeffs = []

    offset_1 = 0
    offset_2 = 0
    offset_3 = 0

    offsets = {}
    # l0 = 0
    # l1 = 1,2,3
    # l2 = 4,5,6,7,8
    
    offset_sph_harm = 0

    for l in range (9):
      offsets[l] = offset_sph_harm
      offset_sph_harm += (2 * l) + 1

    n_l_channels = 0 
    weight_indices = []
    
    for i, (mul, ir_in) in enumerate(irreps1):
        for j, (_, ir_edge) in enumerate(irreps2):
            for ir_out in ir_in * ir_edge:  # | l1 - l2 | <= l <= l1 + l2
                if ir_out in target_irreps:
                  l1 = ir_in.l
                  l2 = ir_edge.l
                  l3 = ir_out.l

                  cg = o3.wigner_3j(l1,l2,l3).to(device)

                  mu1, mu2, mu3 = cg.nonzero(as_tuple=True)
                  
                  x,y,z = cg.shape[0],  cg.shape[1],  cg.shape[2]

                  cg_sparse = cg[(mu1, mu2, mu3)]
                  
                  sorted_indices = mu3.argsort()
                  
                  mu1 = mu1[sorted_indices]
                  mu2 = mu2[sorted_indices]
                  mu3 = mu3[sorted_indices]

                  mu_1.append(mu1 + offsets[l1])
                  mu_2.append(mu2 + offsets[l2])
                  mu_3.append(mu3 + offset_3)

                  cg_sparse = cg_sparse[sorted_indices]

                  cg_coeffs.append(cg_sparse)

                  weight_idxs = torch.ones_like(mu1)
                  
                  weight_indices.append(weight_idxs * n_l_channels)
                  
                  self.dim_out += z
                  offset_3 += z
                  
                  
                  n_l_channels+=1

    if (weights == None):
        self.weights = torch.randn(nchannels, n_l_channels).cuda()
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

    res = tensor_product_cuda.forward(x, self.mu_1, y, self.mu_2, self.cg_coeffs, self.mu_3, self.dim_out)

    return res[0]
  
  def weighted_forward(self, x, y):
    res = tensor_product_cuda.weighted_forward(x, self.mu_1, y, self.mu_2, self.cg_coeffs, self.weights, self.weight_indices)

    return res[0]


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
    
    nchannels=256

    l1 = 1
    l2 = 3
    n_l_channels = 10
    n_edges = 1000
    
    X1 = torch.randn(n_edges, nchannels, (l1 + 1)**2)
    X2 = torch.randn(n_edges, 1, (l2 + 1)**2)
    weights = torch.rand(nchannels, n_l_channels)
    
    X1 = X1.to("cuda")
    X2 = X2.to("cuda")
    weights = weights.to("cuda")
    
    irreps1, irreps2, target_irreps = o3.Irreps(str(nchannels) +"x0e + " + str(nchannels)+"x1o"), o3.Irreps("1x0e + 1x1o + 1x2e + 1x3o"), \
                                      o3.Irreps(str(nchannels)+"x0e + " + str(nchannels)+"x1o +" + str(nchannels)+"x2e + " + str(nchannels)+"x3o")

    tp_cuda = TensorProductCuda(irreps1,irreps2,target_irreps,nchannels,weights, device="cuda")
    tp_reference = TensorProductReference(irreps1,irreps2,target_irreps,nchannels, n_l_channels, weights, device="cuda")

    import torch.utils.benchmark as benchmark
    
    output = tp_cuda.forward(X1, X2)
    reference = tp_reference.forward(X1, X2)
    
    reference_weighted = tp_reference.weighted_forward(X1, X2)
    output_weighted = tp_cuda.weighted_forward(X1, X2)
    
    print ("CUDA vs Reference TP difference")
    print (output - reference)
    print ("CUDA vs Reference weighted TP difference")
    print (output_weighted - reference_weighted)

    t0 = benchmark.Timer(
        stmt='tp(X1, X2)',
        globals={'X1': X1, 'X2': X2, "tp": tp_cuda.forward})

    print("CUDA TP (no weights)", t0.timeit(1000))
    
    t0 = benchmark.Timer(
        stmt='tp(X1, X2)',
        globals={'X1': X1, 'X2': X2, "tp": tp_cuda.weighted_forward})

    print("CUDA TP (weights)", t0.timeit(1000))

    irreps_out, instructions = tp_out_irreps_with_instructions(irreps1, irreps2, target_irreps)

    tp_torch = o3.TensorProduct(irreps1, irreps2,irreps_out, instructions).to("cuda")

    t0 = benchmark.Timer(
    stmt='tp(X1, X2)',
    globals={'X1': X1.reshape(n_edges, nchannels * compute_total_l_channels(l1)), 'X2': X2.reshape(n_edges,compute_total_l_channels(l2)), "tp": tp_torch})

    print("Pyotrch TP (no weights)", t0.timeit(1000))