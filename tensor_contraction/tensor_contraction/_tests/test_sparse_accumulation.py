import torch
import numpy as np
from sparse_accumulation import accumulate
from torch.utils import cpp_extension
from e3nn import o3
from torch.profiler import profile, record_function, ProfilerActivity
from typing import Tuple, List

cpp_extension.load(
    name="sparse_accumulation_cuda",
    sources=["/users/browning/CSCS/COSMO/sparse_accumulation/sparse_accumulation/cuda_extension/sparse_accumulation_cuda_kernel2D.cu"],
    is_python_module=False,
    extra_cuda_cflags=None,
    verbose=True,
)


from torch.utils.cpp_extension import load

tensor_product_cuda = load(
    'tensor_product_cuda', ['/users/browning/CSCS/COSMO/sparse_contraction/tensor_contraction/tensor_contraction/cuda/tensor_product_kernel.cu'], verbose=True, extra_cflags=['-O2'], extra_cuda_cflags=['-O2'])


# cpp_extension.load(
#     name="tensor_product_cuda",
#     sources=["/users/browning/CSCS/COSMO/sparse_contraction/tensor_contraction/tensor_contraction/cuda/tensor_product_kernel.cu"],
#     is_python_module=False,
#     extra_cuda_cflags=None,
#     verbose=True,
# )
#import tensor_product_cuda
import sparse_accumulation_cuda



class Tensor_product(torch.nn.Module):

  def __init__(self, irreps1, irreps2, target_irreps, device="cpu"):
    super().__init__()
    self.irreps1 = irreps1
    self.irreps2 = irreps2
    self.target_irreps = target_irreps
    self.cg_dict = {}
    self.mus_dict = {}
    self.dim_out = 0
    n_cg_coeffs = 0

    for i, (mul, ir_in) in enumerate(irreps1):
        for j, (_, ir_edge) in enumerate(irreps2):
            for ir_out in ir_in * ir_edge:  # | l1 - l2 | <= l <= l1 + l2
                if ir_out in target_irreps:
                  l1 = ir_in.l
                  l2 = ir_edge.l
                  l3 = ir_out.l
                  cg = o3.wigner_3j(l1,l2,l3).to(device)

                  #print (cg.shape)
                  
                  mu1, mu2, mu3 = cg.nonzero(as_tuple=True)
                  
                  #print (mu1, mu2, mu3)

                  cg_sparse = cg[(mu1, mu2, mu3)]
                  
                  #print (len(cg_sparse))

                  n_cg_coeffs += len(cg_sparse)

                  sorted_indices = mu3.argsort()
                  
                  mu1 = mu1[sorted_indices]
                  mu2 = mu2[sorted_indices]
                  mu3 = mu3[sorted_indices]

                  #print (mu1.shape, mu2.shape, mu3.shape)
                  cg_sparse = cg_sparse[sorted_indices]

                  self.cg_dict[l1, l2, l3] = cg_sparse
                  self.mus_dict[l1, l2, l3] = (mu1, mu2, mu3)
                  self.dim_out += len(mu3)

    #print (n_cg_coeffs)

  def forward(self,x,y):
    out = []
    for i, (mul, ir_in) in enumerate(self.irreps1):

        for j, (_, ir_edge) in enumerate(self.irreps2):
            for ir_out in ir_in * ir_edge:  # | l1 - l2 | <= l <= l1 + l2
                if ir_out in self.target_irreps:
                  l1 = ir_in.l
                  l2 = ir_edge.l
                  l3 = ir_out.l
                  cg_sparse = self.cg_dict[l1,l2,l3]
                  mu1, mu2, mu3 = self.mus_dict[l1,l2,l3]

   
                  res = sparse_accumulation_cuda.forward(x, y, mu3, 2 * l3 + 1, mu1, mu2, cg_sparse)


                  out.append(res[0])
    return torch.cat(out, dim=-1)


class Tensor_product_custom(torch.nn.Module):

  def __init__(self, irreps1, irreps2, target_irreps, device="cpu"):
    super().__init__()
    self.irreps1 = irreps1
    self.irreps2 = irreps2
    self.target_irreps = target_irreps
    self.cg_dict = {}
    self.mus_dict = {}
    self.dim_out = 0

    n_cg_coeffs = 0

    mu_1_offsets = []
    mu_1 = []
    mu_2_offsets = []
    mu_2 = []
    mu_3_offsets = []
    mu_3 = []

    cg_coeffs = []
    n_cg_coeffs = []
    cg_offsets = []

    offset_1 = 0
    offset_2 = 0
    offset_3 = 0

    offsets = {}
    # l0 = 0
    # l1 = 1,2,3
    # l2 = 4,5,6,7,8

    num_sph_harm = 0

    for l in range (9):
      offsets[l] = num_sph_harm
      num_sph_harm += (2 * l) + 1

    cg_offset = 0

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

                  mu_1.append(mu1)
                  mu_2.append(mu2)
                  mu_3.append(mu3)

                  mu_1_offsets.append(offsets[l1])
                  mu_2_offsets.append(offsets[l2])
                  mu_3_offsets.append(offset_3)

                  cg_sparse = cg_sparse[sorted_indices]

                  cg_coeffs.append(cg_sparse)
                  n_cg_coeffs.append(cg_sparse.shape[0])
                  cg_offsets.append(cg_offset)

                  cg_offset += cg_sparse.shape[0]

                  self.dim_out += z
                  offset_3 += z

   
    self.mu_1 = torch.cat(mu_1).int()
    self.mu_2 = torch.cat(mu_2).int()
    self.mu_3 = torch.cat(mu_3).int()

    self.mu_1_offsets = torch.Tensor(mu_1_offsets).int().cuda()
    self.mu_2_offsets = torch.Tensor(mu_2_offsets).int().cuda()
    self.mu_3_offsets = torch.Tensor(mu_3_offsets).int().cuda()

    print ("--- mu ---")

    print (self.mu_1)
    print (self.mu_2)
    print (self.mu_3)

    print ("---mu offsets---")
    print (self.mu_1_offsets)
    print (self.mu_2_offsets)
    print (self.mu_3_offsets)


    self.cg_coeffs = torch.cat(cg_coeffs)
    self.n_cg_coeffs = torch.Tensor(n_cg_coeffs).int().cuda()
    self.cg_offsets = torch.Tensor(cg_offsets).int().cuda()

    print ("---cg coeffs---")

    print (self.cg_coeffs)
    print (self.n_cg_coeffs)

  def forward(self,x,y):

    res = tensor_product_cuda.forward(x, self.mu_1, self.mu_1_offsets, y, self.mu_2, self.mu_2_offsets, self.cg_coeffs, self.n_cg_coeffs, self.cg_offsets, self.mu_3, self.mu_3_offsets, self.dim_out)

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

if __name__ == "__main__":

    irreps1, irreps2, target_irreps = o3.Irreps("256x0e + 256x1o"), o3.Irreps("1x0e + 1x1o + 1x2e + 1x3o"), o3.Irreps("256x0e + 256x1o + 256x2e + 256x3o")

    tp = Tensor_product(irreps1,irreps2,target_irreps, device="cuda")
    tp_custom = Tensor_product_custom(irreps1,irreps2,target_irreps, device="cuda")
    l1 = 1
    l2 = 3
    
    n_edges = 2000

    X1 = torch.randn(n_edges, 256, (l1 + 1)**2)
    X2 = torch.randn(n_edges, 1, (l2 + 1)**2)
    X1 = X1.to("cuda")
    X2 = X2.to("cuda")

    import torch.utils.benchmark as benchmark

  

    # print(t0.timeit(1000))

    # with profile(
    #     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    #     with_stack=True,
    # ) as prof:
    #     tp(X1, X2)

    # print(prof.key_averages(group_by_stack_n=5).table(sort_by="self_cuda_time_total", row_limit=5))

    print (irreps1)
    print (irreps2)
    print (target_irreps)

    print ('----')
    
    out = tp(X1, X2)


    print ('----')

    print (out.shape)

    print ('----')
    out_custom = tp_custom.forward_1(X1, X2)
    print ('----')

    
    #print (out)
    #print (out_custom)

    t0 = benchmark.Timer(
        stmt='tp(X1, X2)',
        globals={'X1': X1, 'X2': X2, "tp": tp})

    print(t0.timeit(1000))

    t0 = benchmark.Timer(
         stmt='tp(X1, X2)',
         globals={'X1': X1, 'X2': X2, "tp": tp_custom.forward_1})

    print(t0.timeit(1000))

    t0 = benchmark.Timer(
         stmt='tp(X1, X2)',
         globals={'X1': X1, 'X2': X2, "tp": tp_custom.forward_2})

    print(t0.timeit(1000))

    irreps_out, instructions = tp_out_irreps_with_instructions(irreps1, irreps2, target_irreps)

    tp_torch = o3.TensorProduct(irreps1, irreps2,irreps_out, instructions).to("cuda")

    t0 = benchmark.Timer(
    stmt='tp(X1, X2)',
    globals={'X1': X1.reshape(n_edges, 1024), 'X2': X2.reshape(n_edges,16), "tp": tp_torch})

    print(t0.timeit(1000))

    #out = tp_torch(X1.reshape(n_edges, 1024), X2.reshape(n_edges,16))

    #out =  out.reshape(n_edges, 256, 40)

    #print (out)