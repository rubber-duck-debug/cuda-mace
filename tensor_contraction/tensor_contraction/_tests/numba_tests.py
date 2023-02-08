import numpy as np
import numba
from numba import cuda, float32, int32, int64
from time import perf_counter_ns, time
import torch
from opt_einsum import contract
import math
from numba_cuda import *
import os
from tensor_contraction.cuda.tensor_contraction import U3W3_X_contraction

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

print (U1)
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

X = cuda.to_device(X.reshape(21, 16, 128))

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


#print (U3W3X_torch[0] )
atom_types = np.array([1,1,1,1,1,1,1,2,2,2,1,1,2,0,0,0,0,0,0,0,0])

atom_types = cuda.to_device(atom_types)
UW_numba =  cuda.to_device(UW_torch.contiguous().cpu().numpy())

U3W_non_sparse_indices_numba =  cuda.to_device(U3W_non_sparse_indices.transpose(-1, -2).transpose(-2, -3).contiguous().cpu().numpy())
U3W_num_nonsparse_numba = cuda.to_device(U3W_num_nonsparse.cpu().numpy())
X_numba = cuda.to_device(X_torch.contiguous().cpu().numpy())

out_u3w3x_numba = np.zeros((21, 16, 16, 128),  dtype=np.float32)
out_u3w3x_numba = cuda.to_device(out_u3w3x_numba)

for i in range (timings.shape[0]):
    start = time()
    sparse_accumulate_U3W3_X[(21, 16), (32, 16)](UW_numba, U3W_non_sparse_indices_numba ,#
                        U3W_num_nonsparse_numba, X_numba, atom_types, out_u3w3x_numba)
    cuda.synchronize()
    end = time()

    timings[i] = end - start

print ("numba kernel: %.5f ms" % (np.mean(timings[50:]) * 1000.0))



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
                                    atom_types_torch, 21, 16, 1, 32, 16, 1)


    torch.cuda.synchronize()
    end = time()

    timings[i] = end - start

print ("cuda kernel: %.5f ms" % (np.mean(timings[50:]) * 1000.0))

out_u3w3x_numba = np.zeros((21, 16, 16, 128),  dtype=np.float32)
out_u3w3x_numba = cuda.to_device(out_u3w3x_numba)


sparse_accumulate_U3W3_X[(21, 16,16), 64](UW_numba, U3W_non_sparse_indices_numba ,#
                        U3W_num_nonsparse_numba, X_numba, atom_types, out_u3w3x_numba)



U2_nonsparse_numba =  cuda.to_device(U2_nonsparse.contiguous().cpu().numpy())
U2_nonsparse_indices_numba =  cuda.to_device(U2_non_sparse_indices.contiguous().cpu().numpy())
W2_numba = cuda.to_device(W2_torch.contiguous().cpu().numpy()) 

out_u2w2_numba = np.zeros((21, 16, 16, 128),  dtype=np.float32)
out_u2w2_numba = cuda.to_device(out_u2w2_numba)


#U2 = np.fromfile('../../data/U_2.npy').reshape(16,16, 4)
#W2 = np.fromfile('../../data/W_2.npy').reshape(3,4,128)

for i in range (timings.shape[0]):
    start = time()

    c_tensor = contract("...k,ekc,be->b...c", U2_torch, W2_torch, Y_torch)
    out = U3W3X_torch + c_tensor
    out = contract("bi...c,bic->b...c", out, X_torch)
    torch.cuda.synchronize()

    end = time()

    timings[i] = end - start

print ("torch contract weighting: %.5f ms" % (np.mean(timings[50:]) * 1000.0))

for i in range (timings.shape[0]):
    start = time()
    sparse_accumulate_U2W2_weighting[21, (32, 16)](U2_nonsparse_numba, U2_nonsparse_indices_numba, W2_numba, atom_types, out_u2w2_numba )
    cuda.synchronize()
    end = time()

    timings[i] = end - start

print ("numba U2W2 weighting kernel: %.5f ms" % (np.mean(timings[50:]) * 1000.0))

out_u2w2_numba = np.zeros((21, 16, 16, 128),  dtype=np.float32)
out_u2w2_numba = cuda.to_device(out_u2w2_numba)
sparse_accumulate_U2W2_weighting[(21, 1, 1), (32, 16, 1)](U2_nonsparse_numba, U2_nonsparse_indices_numba, W2_numba, atom_types, out_u2w2_numba )

out_u2w2_x = np.zeros((21, 16, 128),  dtype=np.float32)
out_u2w2_x = cuda.to_device(out_u2w2_x)

for i in range (timings.shape[0]):
    start = time()
    sparse_accumulate_U2W2_contract[(21, 1, 1), (32, 16, 1)](U2_nonsparse_indices_numba, out_u2w2_numba, X_numba, out_u2w2_x )

    cuda.synchronize()
    end = time()

    timings[i] = end - start

print ("numba U2W2 contraction kernel: %.5f ms" % (np.mean(timings[50:]) * 1000.0))

out_u2w2_x = np.zeros((21, 16, 128),  dtype=np.float32)
out_u2w2_x = cuda.to_device(out_u2w2_x)

#out_u2w2_x = out_u2w2_numba + out_u3w3x_numba

sparse_accumulate_U2W2_contract[(21, 1, 1), (32, 16, 1)](U2_nonsparse_indices_numba, out_u2w2_numba, X_numba, out_u2w2_x )


W1_numba = cuda.to_device(W1_torch.contiguous().cpu().numpy()) 

U1_nonsparse_numba =  cuda.to_device(U1_nonsparse.contiguous().cpu().numpy())
U1_nonsparse_indices_numba =  cuda.to_device(U1_non_sparse_indices.contiguous().cpu().numpy())

out_u1w1_numba = np.zeros((21, 16, 128),  dtype=np.float32)
out_u1w1_numba = cuda.to_device(out_u1w1_numba)

for i in range (timings.shape[0]):
    start = time()
    
    sparse_accumulate_U1W1_weighting[(21, 1, 1), (32, 1, 1)](U1_nonsparse_numba, U1_nonsparse_indices_numba, W1_numba, atom_types, out_u1w1_numba )

    cuda.synchronize()
    end = time()

    timings[i] = end - start

print ("numba U1W1 weighting kernel: %.5f ms" % (np.mean(timings[50:]) * 1000.0))

out_u1w1_numba = np.zeros((21, 16, 128),  dtype=np.float32)
out_u1w1_numba = cuda.to_device(out_u1w1_numba)
sparse_accumulate_U1W1_weighting[(21, 1, 1), (32, 1, 1)](U1_nonsparse_numba, U1_nonsparse_indices_numba, W1_numba, atom_types, out_u1w1_numba )

c_tensor = contract("...k,ekc,be->b...c", U1_torch, W1_torch, Y_torch)

#print (out_u1w1_numba.copy_to_host())

#print (c_tensor)


out_u1w1_x_numba = np.zeros((21, 128),  dtype=np.float32)
out_u1w1_x_numba = cuda.to_device(out_u1w1_x_numba)


sparse_accumulate_U1W1_contract[(21, 1, 1), (32, 1, 1)](U1_nonsparse_indices_numba, out_u1w1_numba, X_numba, out_u1w1_x_numba )

## (21, 16, 128)


print (c_tensor.shape)

out = contract("bic..., bic->bc...",c_tensor, X_torch)


print (out_u1w1_x_numba.copy_to_host())

print (out)
#self.equation_contract = "bc...i,bci->bc..."
#
#c_tensor = c_tensor + out
#out = contract(self.equation_contract, c_tensor, x)

#"b...ic, bic -> b...c"