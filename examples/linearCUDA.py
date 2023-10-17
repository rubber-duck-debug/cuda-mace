
import torch
from e3nn import o3
from mace_ops.cuda.instruction import Instruction, _normalize_instruction_path_weights
from mace_ops.cuda.irreps import Irreps
import numpy as np
from time import time

from mace.modules.irreps_tools import (
    tp_out_irreps_with_instructions
)

def reference(X, weights, weight_indices, output_indices, noutputs):

    output = torch.zeros(X.shape[0], noutputs, X.shape[2], device='cuda')

    for i in range(len(output_indices)):
        
        output[:, output_indices[i], :] += X[:, i, :] * weights[weight_indices[i], :]

    return output



if __name__ == "__main__":

    torch.set_printoptions(edgeitems=3)

    dtype = torch.float32

    benchmark = False
    nnodes = 1000
    nfeatures = 128
    max_ell = 3

    irreps1, irreps2, target_irreps = (
        o3.Irreps(f"0e + 1o + 2e + 3o"),
        o3.Irreps("0e + 1o + 2e + 3o"),
        o3.Irreps(f"0e + 1o + 2e + 3o"),
    )


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

    print (irreps_mid)
    print (instructions)

    import e3nn
    import numpy as np

    sum = 0

    weight_indices =  []
    output_indices = []

    for irrep, ins in zip(irreps_mid, instructions):
        l = ins[1]

        ir = e3nn.o3.Irreps(str(irrep))

        output_indices.append(np.arange(ir.lmax ** 2, ir.lmax ** 2 + 2*ir.lmax+1))

        sum += 2 * ir.lmax + 1

        weight_indices.append([ir.lmax] * (2 * ir.lmax + 1))


    weight_indices = torch.tensor(np.hstack(weight_indices), device='cuda', dtype=torch.int32)
    output_indices =  torch.tensor(np.hstack(output_indices), device='cuda', dtype=torch.int32)
   
    print (weight_indices, weight_indices.shape)
    print (output_indices, output_indices.shape)
    
    """   
    torch::Tensor forward_gpu(
        torch::Tensor X,
        torch::Tensor weights,
        torch::Tensor weight_indices,
        torch::Tensor output_indices,
        int64_t noutputs,
        int64_t nthreadx,
        int64_t nthready,
        int64_t nthreadz)
    """

    X = torch.randn(nnodes, len(weight_indices), nfeatures, device='cuda', requires_grad=True)
    weights = torch.randn(max_ell +1, nfeatures, device='cuda', requires_grad=False)

    print (X.shape)

    start = time()
    for i in range (1000):
        output = torch.ops.linear.forward(X,weights, weight_indices, output_indices, (max_ell+1) **2, 64, 4, 1)

    torch.cuda.synchronize()
    
    end = time()
    print (output)
    print (output.shape)

    print ("forward time:", end- start)

    ls = output.sum()

    ls.backward()

    X_grad_cuda = X.grad.clone()

    start = time()
    for i in range (1000):
        output = torch.ops.linear.forward(X,weights, weight_indices, output_indices, (max_ell+1) **2, 64, 4, 1)
        ls = output.sum()

        ls.backward()
    torch.cuda.synchronize()
    
    end = time()

    print ("backward time:", end- start)

    X_ref = X.clone().detach().requires_grad_(True)
    weights_ref = weights.clone().detach().requires_grad_(True)
    ref_out = reference(X_ref, weights_ref, weight_indices, output_indices,  (max_ell+1) **2)

    print (ref_out)

    ls_ref = ref_out.sum()

    ls_ref.backward()

    print (X_ref.grad)

    idx = torch.where(output - ref_out > 1e-6)

    print (output[idx], ref_out[idx])

    assert torch.allclose(output, ref_out, atol=1e-6), "output assert failed."
    assert torch.allclose(X_grad_cuda, X_ref.grad, atol=1e-6), "grad assert failed."
