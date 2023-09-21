import torch
from time import time
from mace_ops import cuda
from mace_ops.cuda.instruction import Instruction, _normalize_instruction_path_weights
from mace_ops.cuda.irreps import Irreps

def benchmark(dtype, device):

    nedges = 300000
    nnodes = 10000
    nfeatures = 64
    nl = 16
    l_max = 3
    L_max = 2

    print(f"--DTYPE: {dtype}")
    print(f"Benchmarking dtype {dtype} and device {device}")
    print(f"nodes: {nnodes} and edges: {nedges}")
    print(f"nfeatures: {nfeatures} and nsphericalharmonics: {nl}")
    X = torch.rand((nedges,  (L_max + 1) ** 2, nfeatures), dtype=dtype,
                   device=device, requires_grad=True)
    Y = torch.rand((nedges, (l_max+1)**2), dtype=dtype,
                   device=device, requires_grad=True)
    radial = torch.randn((nedges,  (l_max + 1) ** 2 + (L_max + 1)**2, nfeatures), dtype=dtype,
                   device=device, requires_grad=True) 



    indices = torch.sort(torch.randint(nnodes, (nedges,), device=device))[0]

    indices_cuda = indices.cuda().int()

    start = time()
    for _ in range(1000):
        neighbour_cuda = torch.ops.mace_ops_equivariant_tp.calculate_neighbours(
            indices_cuda, nnodes, 64)
    finish = time()
    print(f"The indices for CUDA implementation took {finish-start:.3f} seconds")


    # torch::Tensor message_passing_tensor_product(torch::Tensor X,
    #                                          torch::Tensor Y,
    #                                          torch::Tensor radial,
    #                                          torch::Tensor receiver_list,
    #                                          torch::Tensor neighbour_indices,
    #                                          int64_t L_MAX,
    #                                          int64_t l_max,
    #                                          int64_t natoms,
    #                                          int64_t nthreadx,
    #                                          int64_t nthready,
    #                                          int64_t nthreadz)
    

    start = time()
    for i in range (1000):
        out = torch.ops.invariant_tp.message_passing_tensor_product(
            X,
            Y,
            radial,
            indices_cuda, 
            neighbour_cuda, 
            nnodes,  
            32, 8, 1)
        
    end = time()
    print (out[0])
    print (X.shape)
    print (Y.shape)
    print (radial.shape)
    print (out.shape)
    
    print (end - start)
    
    
if __name__ == "__main__":
    benchmark(torch.float32, "cuda")
