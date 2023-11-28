import torch
import math
from time import time
from mace_ops import cuda
from mace_ops.cuda.instruction import Instruction, _normalize_instruction_path_weights
from mace_ops.cuda.irreps import Irreps


def reference(X, Y,  radial, receiver_list, nnodes ):

    output = torch.zeros(nnodes, Y.shape[1], X.shape[1], device=X.device, dtype=X.dtype)

    for i in range (Y.shape[1]):
        
        
        out = X * Y[:, i][:, None] * radial[:, int(math.sqrt(i)), :]
        
        output[:, i, :].index_add_(0,receiver_list, out )

    return output


def benchmark(dtype, device):

    nedges = 30000 * 5
    nnodes = 1000   * 5
    nfeatures = 96
    L_MAX = 3
    nl = (L_MAX +1) ** 2


    print(f"--DTYPE: {dtype}")
    print(f"Benchmarking dtype {dtype} and device {device}")
    print(f"nodes: {nnodes} and edges: {nedges}")
    print(f"nfeatures: {nfeatures} and nsphericalharmonics: {nl}")

    X_nodes = torch.rand((nnodes, nfeatures), dtype=dtype,
                   device=device, requires_grad=True)
    
    Y = torch.rand((nedges, nl), dtype=dtype,
                   device=device, requires_grad=True)
    radial = torch.randn((nedges, L_MAX+1, nfeatures), dtype=dtype,
                   device=device, requires_grad=True) 

    indices = torch.sort(torch.randint(nnodes, (nedges,), device=device))[0]
    indices_cuda = indices.cuda().int()

    X_nodes_ref = X_nodes.clone().detach().requires_grad_(True)
    X_ref =  X_nodes_ref[indices]
    Y_ref = Y.clone().detach().requires_grad_(True)
    radial_ref = radial.clone().detach().requires_grad_(True)

    torch.matmul(torch.rand(1024, 1024, device='cuda'),torch.rand(1024, 1024, device='cuda'))
    torch.cuda.synchronize()
    
    for i in range (1):
        out_ref  = reference(X_ref, Y_ref, radial_ref, indices_cuda, nnodes)

        t = out_ref.sum()

        t.backward()
    torch.cuda.synchronize()

    neighbour_cuda = torch.ops.invariant_tp.calculate_neighbours(indices_cuda, nnodes, 64)
    
    torch.cuda.synchronize()
    
    torch.cuda.cudart().cudaProfilerStart()
    
    fwd_time = 0
    sum_time = 0
    bwd_time = 0
    nrepeats = 1
    for i in range (nrepeats):
        start = time()
        out =  torch.ops.invariant_tp.forward(
            X_nodes,
            Y,
            radial,
            indices_cuda, 
            neighbour_cuda)
        torch.cuda.synchronize()
        fwd_time += time() - start
        

        start = time()
        osum = out.sum()
        torch.cuda.synchronize()
        sum_time += time() - start
        
        start = time()
        osum.backward()
        torch.cuda.synchronize()
        bwd_time += time() - start
        
    torch.cuda.cudart().cudaProfilerStop()

    idx = torch.where(X_nodes.grad - X_nodes_ref.grad > 1e-5)
    
    print (X_nodes.grad[idx])
    print (X_nodes_ref.grad[idx])
    
    assert torch.allclose(out, out_ref, atol=1e-5), "output assert failed"
    assert torch.allclose(radial_ref.grad, radial.grad, atol=1e-5), "radial grad assert failed"
    assert torch.allclose(Y_ref.grad, Y.grad, atol=1e-5), "Y grad assert failed"
    assert torch.allclose(X_nodes.grad, X_nodes_ref.grad, atol=1e-4), "X grad assert failed"

    
    
if __name__ == "__main__":
    benchmark(torch.float32, "cuda")
