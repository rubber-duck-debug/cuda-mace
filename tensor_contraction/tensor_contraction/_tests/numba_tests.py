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

X = np.fromfile('X.npy').reshape(21, 128, 16)
Y = np.fromfile('Y.npy').reshape(21, 3)
U3 = np.fromfile('U_3.npy').reshape(16,16,16,23)
W3 = np.fromfile('W_3.npy').reshape(3,23,128)

U2 = np.fromfile('U_2.npy').reshape(16,16, 4)
W2 = np.fromfile('W_2.npy').reshape(3,4,128)

U1 = np.fromfile('U_1.npy').reshape(16,1)
W1 = np.fromfile('W_1.npy').reshape(3,1,128)

U3_torch = torch.from_numpy(U3).cuda()
W3_torch =  torch.from_numpy(W3).cuda()

U2_torch = torch.from_numpy(U2).cuda()
W2_torch =  torch.from_numpy(W2).cuda()

U1_torch = torch.from_numpy(U1).cuda()
W1_torch =  torch.from_numpy(W1).cuda()

W_2 = np.fromfile('W_2.npy').reshape(3,4,128)
X_torch = torch.from_numpy(X.reshape(21, 128, 16)).float().cuda()
Y_torch = torch.from_numpy(Y).float().cuda()



indices = torch.where(U3_torch !=0)
U3_sparse = U3_torch[indices].cpu().numpy()
U3_sparse_indices = torch.nonzero(U3_torch).cpu().numpy()


U3_sparse = cuda.to_device(U3_sparse)
U3_sparse_indices = cuda.to_device(U3_sparse_indices)

W3 = cuda.to_device(W3)
X = cuda.to_device(X.reshape(21, 16, 128))

U3_non_sparse = np.zeros((3, U3.shape[0], U3.shape[1]), dtype=np.float32)
U3_non_sparse_indices = np.zeros((2, 3, U3.shape[0], U3.shape[1]), dtype=np.int32)

print (U3[1,1, 0])
print (U3[1,1, 6])
print (U3[1,1, 8])


for i in range(U3.shape[0]):
    for j in range(U3.shape[1]):

        idx1, idx2 = np.where(U3[i, j] != 0.0)

        #print (idx1, idx2, U3[i, j, idx1, idx2])
        
        if (i == 1 and j == 1):
            print (idx1, idx2)
            
        if (idx1.shape[0] > 0):
            #print (idx1, idx2)
            for k in range(idx1.shape[0]):
                U3_non_sparse[k, i,j] = U3[i, j, idx1[k], idx2[k]]
                U3_non_sparse_indices[0, k, i, j] = idx1[k]
                U3_non_sparse_indices[1, k, i, j] = idx2[k]
        


U3_non_sparse_indices = cuda.to_device(U3_non_sparse_indices)
U3_non_sparse = cuda.to_device(U3_non_sparse)             

blocks_per_grid = (21,1)
threads_per_block= (16,16)

W_dim_1 = math.ceil(W3.shape[1]/threads_per_block[0])
W_dim_2 = math.ceil(W3.shape[2]/threads_per_block[1])

out_u3w3x = np.zeros((21, 16, 16, 128))
out_u3w3x = cuda.to_device(out_u3w3x)

print (Y)

print (W_dim_1, W_dim_2)

atom_types = np.array([1,1,1,1,1,1,1,2,2,2,1,1,2,0,0,0,0,0,0,0,0])
atom_types = cuda.to_device(atom_types)

U_tensors = {3: U3_torch.float(), 2:  U2_torch.float(), 1: U1_torch.float()}
W_tensors = {3: W3_torch.float(), 2: W2_torch.float(), 1: W1_torch.float()}

equation_main = "...ik,ekc,bci,be -> bc..."
equation_weighting = "...k,ekc,be->bc..."
equation_contract = "bc...i,bci->bc..."

UW = contract('...ik, ekc -> ...iec', U_tensors[3],W_tensors[3])

print (UW.shape)
UWX = contract('...iec, bci -> bec...', UW,X_torch)


UWXY = contract('bec..., be -> bc...', UWX, Y_torch)

out_contract = contract(equation_main, U_tensors[3], W_tensors[3], X_torch, Y_torch)

for i in range (50):
    start = time()
    out_contract = contract(equation_main, U_tensors[3], W_tensors[3], X_torch, Y_torch)
    end = time()

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

@cuda.jit
def sparse_accumulate_U3W(U3_non_sparse, U3_non_sparse_indices, W, out):
    
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bx = cuda.blockIdx.x

    U_nonsparse_shared = cuda.shared.array(shape=(3, 16,16), dtype=float32)
    U_nonsparse_indices_shared = cuda.shared.array(shape=(2,3, 16,16), dtype=int32)

    for i in range(3):
        U_nonsparse_shared[i, ty,tx] = U3_non_sparse[i, ty, tx]
        U_nonsparse_indices_shared[0, i, ty, tx] = U3_non_sparse_indices[0, i, ty, tx]
        U_nonsparse_indices_shared[1, i, ty, tx] = U3_non_sparse_indices[1, i, ty, tx]
    
    cuda.syncthreads()

    for element in range(3): 
    
        #if (bx == i_val and tx == 0 and ty == 0):   
        #    print (W[element, 11, 0], W_shared[11, 0])
                  
        for r in range(3):
            
            idx = U_nonsparse_indices_shared[0, r, bx, ty]
            jdx = U_nonsparse_indices_shared[1, r, bx, ty]
            
            U_val = U_nonsparse_shared[r, bx, ty]
        
            if (U_val != 0.0):
                
                #if (tx == 0 and bx == i_val and ty == j_val and idx == k_val):
                #    print (bx, ty, idx, jdx, U_val, W_shared[jdx,0])
                
                for i in range(8):
                    idx_i = i * 16 + tx
                    out[bx, ty, idx, element, idx_i]  =  U_val * W[element, jdx,idx_i]
                

sparse_accumulate_U3W[16, (16,16)](U3_non_sparse, U3_non_sparse_indices, W3, out)

for i in range (50):
    start = time()
    sparse_accumulate_U3W[16, (16,16)](U3_non_sparse, U3_non_sparse_indices, W3, out)
    torch.cuda.synchronize()
    end = time()

print ("kernel time %.3f ms" % ((end - start) * 1000.0))

hout = out.copy_to_host()

#print (UW[0, :, :, 0, :])
#print (hout[0, :, :, 0, :])

diff= UW.cpu().numpy() - hout 

print ("----")
print (U3[i_val,j_val,k_val])
print (U3_non_sparse.copy_to_host()[:, i_val,j_val])
print (U3_non_sparse_indices.copy_to_host()[..., i_val,j_val])

print ("weights:")
print (W3[0, l_val, 0])
print (W3[1, l_val, 0])
print (W3[2, l_val, 0])

print ()

test = U3[i_val,j_val,k_val, l_val] * W3[:, l_val, :]

print (hout[i_val,j_val,k_val, 0])
print ("--test--")

print (test[0])
print (UW[i_val,j_val,k_val, 0])

idx, jdx, kdx, edx, ldx = np.where(diff>  1e-5)

print (idx)
print (jdx)
print (kdx)
print (edx)

sparse_accumulate_U3W3X[(21, 16), (16,16)](U3_non_sparse, U3_non_sparse_indices, W3,X, atom_types, out_u3w3x)

for i in range (50):
    start = time()
    sparse_accumulate_U3W3X[(21, 16), (16,16)](U3_non_sparse, U3_non_sparse_indices, W3,X, atom_types, out_u3w3x)
    torch.cuda.synchronize()
    end = time()

print (out_contract)
print (out_u3w3x.copy_to_host())
print ("U3U3X kernel time %.3f ms" % ((end - start) * 1000.0))

#for i in range(5):
#    print (idx[i*128 * 3], jdx[i*128*3], kdx[i*128*3] )



