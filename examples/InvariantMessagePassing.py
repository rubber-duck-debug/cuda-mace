import torch
import math
from time import time
from mace_ops import cuda
from mace_ops.cuda.instruction import Instruction, _normalize_instruction_path_weights
from mace_ops.cuda.irreps import Irreps
from mace_ops.ops.invariant_message_passing import InvariantMessagePassingTP

def reference(X, Y,  radial, receiver_list, nnodes ):

    output = torch.zeros(nnodes, Y.shape[1], X.shape[1], device=X.device, dtype=X.dtype)

    for i in range (Y.shape[1]):
        
        
        out = X * Y[:, i][:, None] * radial[:, int(math.sqrt(i)), :]
        
        output[:, i, :].index_add_(0,receiver_list, out )

    return output

def check_correctness(node_feats, edge_attrs, tp_weights, receiver_list, nnodes):
    
    node_feats_cuda = node_feats.clone().detach().requires_grad_(True)
    edge_attrs_cuda = edge_attrs.clone().detach().requires_grad_(True)
    tp_weights_cuda = tp_weights.clone().detach().requires_grad_(True)
    
    node_feats_ref = node_feats.clone().detach().requires_grad_(True)
    node_feats_ref_sampled =  node_feats_ref[receiver_list]
    edge_attrs_ref = edge_attrs.clone().detach().requires_grad_(True)
    tp_weights_ref = tp_weights.clone().detach().requires_grad_(True)
    
    #run the reference
    torch.cuda.synchronize()
    out_ref  = reference(node_feats_ref_sampled, edge_attrs_ref, tp_weights_ref, receiver_list, nnodes)
    t = out_ref.sum()
    t.backward()
    torch.cuda.synchronize()
    
    tp = InvariantMessagePassingTP()
    
    first_occurences = tp.calculate_first_occurences(receiver_list, nnodes)
    
    torch.cuda.synchronize()
    out = tp.forward(
        node_feats_cuda,
        edge_attrs_cuda,
        tp_weights_cuda,
        receiver_list, 
        first_occurences)
    osum = out.sum()
    osum.backward()
    torch.cuda.synchronize()
    
    assert torch.allclose(out, out_ref, atol=1e-5), "output assert failed"
    assert torch.allclose(tp_weights_ref.grad, tp_weights_cuda.grad, atol=1e-5), "tp_weights grad assert failed"
    assert torch.allclose(edge_attrs_ref.grad, edge_attrs_cuda.grad, atol=1e-5), "edge_attrs grad assert failed"
    assert torch.allclose(node_feats_ref.grad, node_feats_cuda.grad, atol=1e-4), "node_feats grad assert failed"
    
    
    #run the CUDA code

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

    node_feats = torch.rand((nnodes, nfeatures), dtype=dtype,
                   device=device, requires_grad=True)
    
    edge_attrs = torch.rand((nedges, nl), dtype=dtype,
                   device=device, requires_grad=True)
    tp_weights = torch.randn((nedges, L_MAX+1, nfeatures), dtype=dtype,
                   device=device, requires_grad=True) 

    receiver_list = torch.sort(torch.randint(nnodes, (nedges,), device=device, dtype=torch.int))[0]

    #warmup
    torch.matmul(torch.rand(1024, 1024, device='cuda'),torch.rand(1024, 1024, device='cuda'))
    torch.cuda.synchronize()
    
    check_correctness(node_feats, edge_attrs, tp_weights, receiver_list, nnodes)
    
    tp = InvariantMessagePassingTP()
    
    first_occurences = tp.calculate_first_occurences(receiver_list, nnodes)
    
    torch.cuda.synchronize()
    
    torch.cuda.cudart().cudaProfilerStart()
    
    #do some runs so we can gather timings via `nsys nvprof`
    nrepeats = 1000
    for i in range (nrepeats):
        out = tp.forward(
            node_feats,
            edge_attrs,
            tp_weights,
            receiver_list, 
            first_occurences)
        torch.cuda.synchronize()
        
        osum = out.sum()
        osum.backward()
        torch.cuda.synchronize()
        
    torch.cuda.cudart().cudaProfilerStop()

    
if __name__ == "__main__":
    benchmark(torch.float32, "cuda")
