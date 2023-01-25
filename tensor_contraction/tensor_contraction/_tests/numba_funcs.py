import numpy as np
import numba
from numba import cuda, float32, int32, int64
from time import perf_counter_ns, time
import torch
from opt_einsum import contract
import math


# '...ik, ekc -> ...iec'                    
#                                           i  k
#U3 = np.fromfile('U_3.npy').reshape(16,16,16,23)
#                                    e  k   c        
#W3 = np.fromfile('W_3.npy').reshape(3,23,128)

#out = np.zeros((16, 16, 16, 3, 128))

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

\
# equation_main = "...ik,ekc,bci,be -> bc..."

#                         i   e  c
#out = np.zeros((16, 16, 16, 3, 128))
#X = np.zeros((21,16, 128))
# iec, bic -> bc
#out = np.zeros((21, 16, 16, 128))
@cuda.jit
def sparse_accumulate_U3W3_X(UW, U3W_non_sparse_indices, U3W_num_nonsparse, X, atom_types, out):
   
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y
    bz = cuda.blockIdx.z
    
    gx = cuda.gridDim.x
    gy = cuda.gridDim.y
    gz = cuda.gridDim.z

    cuda.syncthreads()
    
    element = atom_types[bx]

    num_threads_x = cuda.blockDim.x

    #X_shared = cuda.shared.array(shape=(128), dtype=float32)

    n_iter_x = 128 / num_threads_x

    for y in range(by, 16, gy):
    
        num_non_sparse = U3W_num_nonsparse[y, ty]

        #for fi in range((n_iter_x)):
        #    idx_i = fi * num_threads_x + tx
        #    X_shared[idx_i] = 0.0

        for l in range(num_non_sparse):

            ldx = U3W_non_sparse_indices[l, y,ty]
        
            for fi in range(n_iter_x):
                idx_i = fi * num_threads_x + tx

                uw = UW[y, ty, ldx, element, idx_i]

                x = X[bx, ldx, idx_i]

                numba.cuda.atomic.add(out, (bx, y, ty, idx_i),uw * x)

            #for fi in range(n_iter_x):
            #    idx_i = fi * num_threads_x + tx
            #    out[bx, y, z, idx_i] = X_shared[idx_i]
                
            
            #out[bx, ty, j, idx_i] +=  U_nonsparse_shared[r, j, ty]  * X[bx, idx, idx] * W_shared[jdx,idx_i]






