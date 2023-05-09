import torch
import numpy as np
from time import time
from typing import Tuple, List

from e3nn import o3
from mace_ops.cuda import TensorProduct
from TensorProductReference import TensorProductReference

if __name__ == "__main__":

    torch.set_printoptions(edgeitems=3)
    
    dtype = torch.float32

    nchannels = 96

    l1 = 1
    l2 = 3

    n_atoms = 500

    irreps1, irreps2, target_irreps = (
        o3.Irreps(f"0e + 1o"),
        o3.Irreps("0e + 1o + 2e + 3o"),
        o3.Irreps(f"0e + 1o + 2e + 3o"),
    )

    print ("natoms: ", n_atoms)
    print ("nchannels: ", nchannels)
    print ("irreps1: ", irreps1)
    print ("irreps2: ", irreps2)
    print ("target_irreps: ", target_irreps)
    print ("dtype: ", dtype)

    X1 = torch.randn(n_atoms, irreps1.dim, nchannels, requires_grad=True, device='cuda', dtype=dtype)
    X2 = torch.randn(n_atoms, irreps2.dim, 1, requires_grad=True, device='cuda', dtype=dtype)

    tp_cuda = TensorProduct(irreps1,irreps2,target_irreps, device="cuda", dtype=dtype)
    tp_reference = TensorProductReference(irreps1,irreps2,target_irreps,nchannels, device="cuda", dtype=dtype)

    X1_ref = X1.detach().clone()
    X2_ref = X2.detach().clone()
    X1_ref.requires_grad = True
    X2_ref.requires_grad = True

    output = tp_cuda.forward(X1, X2)

    output_ref = tp_reference.forward(X1_ref, X2_ref)

    if (X1.requires_grad or X2.requires_grad):
        (output_ref).sum().backward()
        (output).sum().backward()

    if (not torch.allclose(output[0], output_ref[0], atol=1e-7)):
        idx = torch.where(output[0] - output_ref[0] > 1e-7)
        print ("possible issue with output at indices:")
        print (idx)
        print ("--diff_tensor--")
        print (output[0][idx] - output[0][idx])

    if (X1.requires_grad and not torch.allclose(X1.grad[0],  X1_ref.grad[0], atol=1e-7)):
        idx = torch.where(X1.grad[0] -  X1_ref.grad[0] > 1e-7)
        print ("possible issue with grad X1 at indices:")
        print (idx)
        print ("--diff_tensor--")
        print (X1.grad[0][idx] - X1_ref.grad[0][idx])

    if (X2.requires_grad and not torch.allclose(X2.grad[0],  X2_ref.grad[0], atol=1e-7)):
        idx = torch.where(X2.grad[0] -  X2_ref.grad[0] > 1e-7)
        print ("possible issue with grad X2 at indices:")
        print (idx)
        print ("--diff_tensor--")
        print (X2.grad[0][idx] - X2_ref.grad[0][idx])

    niter = 1000

    duration_fwd = 0
    duration_bwd = 0

    for i in range (niter):
        
        start = time()
        output = tp_cuda.forward(X1, X2)
        end  = time()
        duration_fwd += end - start

        os = output.sum()
        
        start = time()
        os.backward()
        end = time()

        duration_bwd += end - start
    
    print ("--forward time-- (us)")
    print (duration_fwd * 1000)
    print ("--backward time-- (us)")
    print (duration_bwd * 1000)

    






