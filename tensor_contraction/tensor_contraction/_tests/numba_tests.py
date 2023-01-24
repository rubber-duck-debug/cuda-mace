import numpy as np
import numba
from numba import cuda, float32, int32, int64
from time import perf_counter_ns, time
import torch
from opt_einsum import contract
import math
from numba_funcs import *
import os

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

U3_torch = torch.from_numpy(U3).cuda()
W3_torch =  torch.from_numpy(W3).cuda()

U2_torch = torch.from_numpy(U2).cuda()
W2_torch =  torch.from_numpy(W2).cuda()

U1_torch = torch.from_numpy(U1).cuda()
W1_torch =  torch.from_numpy(W1).cuda()

X_torch = torch.from_numpy(X.reshape(21, 128, 16)).float().cuda()
Y_torch = torch.from_numpy(Y).float().cuda()

atom_types = np.array([1,1,1,1,1,1,1,2,2,2,1,1,2,0,0,0,0,0,0,0,0])
X = cuda.to_device(X.reshape(21, 16, 128))

U3_non_sparse = np.zeros((U3.shape[0], U3.shape[1], 3), dtype=np.float32)
U3_non_sparse_indices = np.zeros((U3.shape[0], U3.shape[1], 2, 3), dtype=np.int32)

for i in range(U3.shape[0]):
    for j in range(U3.shape[1]):

        idx1, idx2 = np.where(U3[i, j] != 0.0)

        #print (idx1, idx2, U3[i, j, idx1, idx2])
            
        if (idx1.shape[0] > 0):
            #print (idx1, idx2)
            for k in range(idx1.shape[0]):
                U3_non_sparse[i, j, k] = U3[i, j, idx1[k], idx2[k]]
                U3_non_sparse_indices[i, j, 0, k] = idx1[k]
                U3_non_sparse_indices[i, j, 1, k] = idx2[k]


u3w3x = np.zeros((21, 128, 16, 16))


U_tensors = {3: U3_torch.float(), 2:  U2_torch.float(), 1: U1_torch.float()}
W_tensors = {3: W3_torch.float(), 2: W2_torch.float(), 1: W1_torch.float()}

print (U_tensors[3].shape)
print (W_tensors[3].shape)

UW_torch = contract('...ik, ekc -> ...iec', U_tensors[3],W_tensors[3])
print (UW_torch.shape)

equation_main = "...ik,ekc,bci,be -> bc..."
equation_weighting = "...k,ekc,be->bc..."
equation_contract = "bc...i,bci->bc..."

out_contract = contract(equation_main, U_tensors[3], W_tensors[3], X_torch, Y_torch)

out = torch.zeros(out_contract.shape, device=out_contract.device)

print (out_contract.shape)

for i in range (X.shape[0]):

    element_idx = atom_types[i]

    for j in range (UW_torch.shape[0]):

        for k in range (UW_torch.shape[1]):

            for l in range(UW_torch.shape[2]):

                out [i, :, j, k] += UW_torch[j,k,l, element_idx, :] * X_torch[i, :, l]

print (out[0])


U3_non_sparse_indices = cuda.to_device(U3_non_sparse_indices)
U3_non_sparse = cuda.to_device(U3_non_sparse)             

blocks_per_grid = (21,1)
threads_per_block= (16,16)

W_dim_1 = math.ceil(W3.shape[1]/threads_per_block[0])
W_dim_2 = math.ceil(W3.shape[2]/threads_per_block[1])

out_u3w3x = np.zeros((21, 16, 16, 128))
out_u3w3x = cuda.to_device(out_u3w3x)


atom_types = cuda.to_device(atom_types)


UW_torch = contract('...ik, ekc -> ...iec', U_tensors[3],W_tensors[3])


#torch.Size([16, 16, 16, 3, 128])
# torch.Size([21, 128, 16, 16])

UW = UW_torch.cpu().numpy()

UW_nonzero_indices = np.zeros((3, UW.shape[0], UW.shape[1]))
UW_num_nonzero = np.zeros((UW.shape[0], UW.shape[1]))

for i in range (UW.shape[0]):
    for j in range (UW.shape[1]):
        idx1, idx2, idx3 = np.where(UW[i, j] != 0.0)

        unique_idx = np.unique(idx1)

        UW_num_nonzero[i,j] =unique_idx.shape[0]

        UW_nonzero_indices[:unique_idx.shape[0], i, j] = unique_idx

UWX = contract('...iec, bci -> bec...', UW_torch, X_torch)

UWXY = contract('bec..., be -> bc...', UWX, Y_torch)

out_contract = contract(equation_main, U_tensors[3], W_tensors[3], X_torch, Y_torch)

for i in range (50):
    start = time()
    out_contract = contract(equation_main, U_tensors[3], W_tensors[3], X_torch, Y_torch)
    end = time()

print (out_contract.shape)

print ("einsum time %.3f ms" % ((end - start) * 1000.0))


# '...ik, ekc -> ...iec'                    
#                                           i  k
#U3 = np.fromfile('U_3.npy').reshape(16,16,16,23)
#                                    e  k   c        
#W3 = np.fromfile('W_3.npy').reshape(3,23,128)

out = np.zeros((16, 16, 16, 3, 128))
out = cuda.to_device(out)

i_val = 3
j_val = 3
k_val = 8
l_val = 11


#sparse_accumulate_U3W[16, (16,16)](U3_non_sparse, U3_non_sparse_indices, W3, out)

#for i in range (50):
#    start = time()
#    sparse_accumulate_U3W[16, (16,16)](U3_non_sparse, U3_non_sparse_indices, W3, out)
#    torch.cuda.synchronize()
#    end = time()

#print ("kernel time %.3f ms" % ((end - start) * 1000.0))

W3 = cuda.to_device(W3)

for i in range (50):
    start = time()
    sparse_accumulate_U3W3X[(21, 16), (16,16)](U3_non_sparse, U3_non_sparse_indices, W3,X, atom_types, out_u3w3x)
    torch.cuda.synchronize()
    end = time()

out_u3w3x = np.zeros((21, 16, 16, 128))
out_u3w3x = cuda.to_device(out_u3w3x)

sparse_accumulate_U3W3X[(21, 16, 16), (32)](U3_non_sparse, U3_non_sparse_indices, W3,X, atom_types, out_u3w3x)

print (out_contract[0])
#print (out_u3w3x.copy_to_host().reshape(21, 128, 16, 16)[0])
#print ("U3U3X kernel time %.3f ms" % ((end - start) * 1000.0))