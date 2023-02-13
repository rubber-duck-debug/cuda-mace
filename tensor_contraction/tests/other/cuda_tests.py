import numpy as np
import numba
from numba import cuda, float32, int32, int64
from time import perf_counter_ns, time
import torch
from opt_einsum import contract
import math
from numba_cuda import *
import os
from tensor_contraction.cuda.tensor_contraction import *

os.environ['NUMBA_ENABLE_CUDASIM'] = "1"

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

X = np.fromfile('../../data/X.npy').reshape(21, 128, 16)
Y = np.fromfile('../../data/Y.npy').reshape(21, 3)
U3 = np.fromfile('../../data/U_3.npy').reshape(16,16,16,23)
W3 = np.fromfile('../../data/W_3.npy').reshape(3,23,128)

U2 = np.fromfile('../../data/U_2.npy').reshape(16,16, 4)
W2 = np.fromfile('../../data/W_2.npy').reshape(3,4,128)

U1 = np.fromfile('../../data/U_1.npy').reshape(16,1)

W1 = np.fromfile('../../data/W_1.npy').reshape(3,1,128)

U3_torch = torch.from_numpy(U3).float().cuda()
W3_torch =  torch.from_numpy(W3).float().cuda()

U2_torch = torch.from_numpy(U2).float().cuda()
W2_torch =  torch.from_numpy(W2).float().cuda()

U1_torch = torch.from_numpy(U1).float().cuda()
W1_torch =  torch.from_numpy(W1).float().cuda()

#self.equation_main = "...ik,ekc,bci,be -> bc..."
#self.equation_weighting = "...k,ekc,be->bc..."
#self.equation_contract = "bc...i,bci->bc..."

X_torch = torch.from_numpy(X).float().cuda().transpose(-1, -2) # (21, 16, 128)
Y_torch = torch.from_numpy(Y).float().cuda()

atom_types = np.array([1,1,1,1,1,1,1,2,2,2,1,1,2,0,0,0,0,0,0,0,0])

atom_types_torch = torch.from_numpy(atom_types).int().cuda()

U_tensors = {3: U3_torch.float(), 2:  U2_torch.float(), 1: U1_torch.float()}
W_tensors = {3: W3_torch.float(), 2: W2_torch.float(), 1: W1_torch.float()}

U3W_non_sparse_indices = torch.zeros((U3.shape[0], U3.shape[1], 3), dtype=torch.int32).cuda()
U3W_num_nonsparse = torch.zeros((U3.shape[0], U3.shape[1]), dtype=torch.int32).cuda()

equation_main = "...ik,ekc,bic,be -> b...c"

UW_torch = contract('...ik, ekc -> ...iec', U_tensors[3],W_tensors[3])

U3W3X_torch = contract(equation_main, U_tensors[3],W_tensors[3], X_torch, Y_torch)

timings = np.zeros(100)

for i in range (timings.shape[0]):
    start = time()
    U3W3X_torch = contract("...iec,bic,be -> b...c", UW_torch, X_torch, Y_torch)
    torch.cuda.synchronize()
    end = time()

    timings[i] = end - start

print ("opt_eimsum: %.5f ms" % (np.mean(timings[50:]) * 1000.0))

print (UW_torch.shape) # torch.Size([16, 16, 16, 3, 128])
print (U3W3X_torch.shape)

for i in range(UW_torch.shape[0]):
    for j in range(UW_torch.shape[1]):

        idx, edx, ldx  = torch.where(UW_torch[i, j] != 0.0)
        
        idx = torch.unique(idx)

        if (idx.shape[0] > 0):

            U3W_non_sparse_indices[i, j, :idx.shape[0]] = idx
            U3W_num_nonsparse[i, j] = idx.shape[0]

U2_non_sparse_indices = torch.zeros((U2_torch.shape[0], 3), dtype=torch.int32).cuda()
U2_nonsparse = torch.zeros((U2_torch.shape[0]), dtype=torch.float).cuda()

count = 0
for i in range(U2_torch.shape[0]):

    for j in range (U2_torch.shape[1]):

        kdx,   = torch.where(U2_torch[i, j] != 0.0)
        
        if (len(kdx) > 0):
            U2_nonsparse[count] = U2_torch[i, j][kdx]
            U2_non_sparse_indices[count, 0] = i
            U2_non_sparse_indices[count, 1] = j
            U2_non_sparse_indices[count, 2] = kdx

            count +=1

U1_non_sparse_indices = torch.zeros((1, 2), dtype=torch.int32).cuda()
U1_nonsparse = torch.zeros(1, dtype=torch.float).cuda()

count = 0
for i in range(U2_torch.shape[0]):

    kdx,   = torch.where(U1_torch[i] != 0.0)
    
    if (len(kdx) > 0):
        U1_nonsparse[count] = U1_torch[i, kdx]
        U1_non_sparse_indices[count, 0] = i
        U2_non_sparse_indices[count, 1] = kdx

        count +=1

out_u3w3x = torch.zeros((21, 16, 16, 128), device='cuda', dtype=torch.float32)

for i in range(X_torch.shape[0]):

    atom_type = atom_types[i]

    for j in range (UW_torch.shape[0]):
        for k in range (UW_torch.shape[1]):

            for l in range(U3W_num_nonsparse[j, k]):

                ldx = U3W_non_sparse_indices[j,k,l]

                out_u3w3x[i, j, k, :] += UW_torch[j,k,ldx, atom_type, :] * X_torch[i, ldx, :]

print ("X:", X_torch.shape)

UW2 = contract("...k,ekc->e...c", U_tensors[2], W_tensors[2])
UW1 = contract("...k,ekc->e...c", U_tensors[1], W_tensors[1])

print (UW1.shape, UW2.shape)

for i in range (timings.shape[0]):
    start = time()
    out = correlation_3_main(UW_torch,
                            U3W_non_sparse_indices,
                            U3W_num_nonsparse,
                            X_torch,
                            atom_types_torch, 21, 16, 1, 32, 16, 1)

    out = correlation_2_contraction(U2_non_sparse_indices, UW2, out, 
                                    X_torch, atom_types_torch, 
                                    21, 1, 1,32,16,1)

    out = correlation_1_contraction(U1_non_sparse_indices, UW1, out, 
                                    X_torch, atom_types_torch,
                                    21, 1, 1,32,1,1)

    torch.cuda.synchronize()
    end = time()

    timings[i] = end - start

print (out)

print ("cuda time: %.5f ms" % (np.mean(timings[50:]) * 1000.0))

equation_main = "...ik,ekc,bic,be -> b...c"
equation_weighting = "...k,ekc,be->b...c"
equation_contract = "bi...c,bic->b...c"

correlation = 3

for i in range (timings.shape[0]):
    start = time()
    out = contract(
                        equation_main,
                        U_tensors[correlation],
                        W_tensors[correlation],
                        X_torch,
                        Y_torch,
                    ) 

    for corr in range(correlation - 1, 0, -1):

        c_tensor = contract(
            equation_weighting,
            U_tensors[corr],
            W_tensors[corr],
            Y_torch,
        )
        c_tensor = c_tensor + out
        out = contract(equation_contract, c_tensor, X_torch)

        torch.cuda.synchronize()
        end = time()

        timings[i] = end - start


print (out)

print ("torch time: %.5f ms" % (np.mean(timings[50:]) * 1000.0))