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

print (X.shape)

X_torch = torch.from_numpy(X).float().cuda().transpose(-1, -2) # (21, 16, 128)
Y_torch = torch.from_numpy(Y).float().cuda()

atom_types = np.array([1,1,1,1,1,1,1,2,2,2,1,1,2,0,0,0,0,0,0,0,0])

atom_types_torch = torch.from_numpy(atom_types).int().cuda()

X = cuda.to_device(X.reshape(21, 16, 128))

U_tensors = {3: U3_torch.float(), 2:  U2_torch.float(), 1: U1_torch.float()}
W_tensors = {3: W3_torch.float(), 2: W2_torch.float(), 1: W1_torch.float()}

U3W_non_sparse_indices = torch.zeros((U3.shape[0], U3.shape[1], 3), dtype=torch.int32).cuda()
U3W_num_nonsparse = torch.zeros((U3.shape[0], U3.shape[1]), dtype=torch.int32).cuda()

equation_main = "...ik,ekc,bic,be -> b...c"

UW_torch = contract('...ik, ekc -> ...iec', U_tensors[3],W_tensors[3])

U3W3X_torch = contract(equation_main, U_tensors[3],W_tensors[3], X_torch, Y_torch)

timings = np.zeros(50)

for i in range (timings.shape[0]):
    start = time()
    U3W3X_torch = contract("...iec,bic,be -> b...c", UW_torch, X_torch, Y_torch)
    torch.cuda.synchronize()
    end = time()

    timings[i] = end - start

print ("opt_eimsum: %.5f ms" % (np.mean(timings[5:]) * 1000.0))

print (UW_torch.shape) # torch.Size([16, 16, 16, 3, 128])
print (U3W3X_torch.shape)

for i in range(UW_torch.shape[0]):
    for j in range(UW_torch.shape[1]):

        idx, edx, ldx  = torch.where(UW_torch[i, j] != 0.0)
        
        idx = torch.unique(idx)

        if (idx.shape[0] > 0):

            U3W_non_sparse_indices[i, j, :idx.shape[0]] = idx
            U3W_num_nonsparse[i, j] = idx.shape[0]

out_u3w3x = torch.zeros((21, 16, 16, 128), device='cuda', dtype=torch.float32)

for i in range(X_torch.shape[0]):

    atom_type = atom_types[i]

    for j in range (UW_torch.shape[0]):
        for k in range (UW_torch.shape[1]):

            for l in range(U3W_num_nonsparse[j, k]):

                ldx = U3W_non_sparse_indices[j,k,l]

                out_u3w3x[i, j, k, :] += UW_torch[j,k,ldx, atom_type, :] * X_torch[i, ldx, :]


#print (U3W3X_torch[0] )
atom_types = np.array([1,1,1,1,1,1,1,2,2,2,1,1,2,0,0,0,0,0,0,0,0])

atom_types = cuda.to_device(atom_types)
UW_numba =  cuda.to_device(UW_torch.contiguous().cpu().numpy())

U3W_non_sparse_indices_numba =  cuda.to_device(U3W_non_sparse_indices.transpose(-1, -2).transpose(-2, -3).contiguous().cpu().numpy())
U3W_num_nonsparse_numba = cuda.to_device(U3W_num_nonsparse.cpu().numpy())
X_numba = cuda.to_device(X_torch.contiguous().cpu().numpy())



out_u3w3x_numba = np.zeros((21, 16, 16, 128),  dtype=np.float32)
out_u3w3x_numba = cuda.to_device(out_u3w3x_numba)

timings = np.zeros(50)
for i in range (timings.shape[0]):
    start = time()
    sparse_accumulate_U3W3_X[(21, 16), (32, 16)](UW_numba, U3W_non_sparse_indices_numba ,#
                        U3W_num_nonsparse_numba, X_numba, atom_types, out_u3w3x_numba)
    cuda.synchronize()
    end = time()

    timings[i] = end - start

print ("numba kernel: %.5f ms" % (np.mean(timings[5:]) * 1000.0))


from tensor_contraction import U3W3_X_contraction

print (UW_torch.shape)
print (U3W_non_sparse_indices.shape)
print (U3W_num_nonsparse.shape)
print (X_torch.shape)

for i in range (timings.shape[0]):
    start = time()
    U3W3X_cuda = U3W3_X_contraction(
                                    UW_torch,
                                    U3W_non_sparse_indices,
                                    U3W_num_nonsparse,
                                    X_torch,
                                    atom_types_torch)


    torch.cuda.synchronize()
    end = time()

    timings[i] = end - start

print ("cuda kernel: %.5f ms" % (np.mean(timings[5:]) * 1000.0))

#print (U3W3X_torch[0])
#print (U3W3X_cuda[0].shape)
#print (U3W3X_cuda[0])

out_u3w3x_numba = np.zeros((21, 16, 16, 128),  dtype=np.float32)
out_u3w3x_numba = cuda.to_device(out_u3w3x_numba)


sparse_accumulate_U3W3_X[(21, 16,16), 64](UW_numba, U3W_non_sparse_indices_numba ,#
                        U3W_num_nonsparse_numba, X_numba, atom_types, out_u3w3x_numba)


out_u3w3x_numba = out_u3w3x_numba.copy_to_host()

#print (out_u3w3x_numba[0])


""" u3w3x = np.zeros((21, 128, 16, 16))



print (U_tensors[3].shape)
print (W_tensors[3].shape)

UW_torch = contract('...ik, ekc -> ...iec', U_tensors[3],W_tensors[3])
print (UW_torch.shape)



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
#print ("U3U3X kernel time %.3f ms" % ((end - start) * 1000.0)) """