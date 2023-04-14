import numpy as np
from time import time
from opt_einsum import contract
import torch
from tensor_contraction.cuda.tensor_contraction import *
from numba import cuda
import logging
import traceback
        
X = np.fromfile('../../data/X.npy').reshape(21, 128, 16)


Y = np.fromfile('../../data/Y.npy').reshape(21, 3)


U_3 = np.fromfile('../../data/U_3.npy').reshape(16,16,16,23)
U_2 = np.fromfile('../../data/U_2.npy').reshape(16,16, 4)
U_1 = np.fromfile('../../data/U_1.npy').reshape(16,1)

U_3 = torch.from_numpy(U_3).float().cuda()
U_2 = torch.from_numpy(U_2).float().cuda()
U_1 = torch.from_numpy(U_1).float().cuda()

W_3 = np.fromfile('../../data/W_3.npy').reshape(3,23,128)
W_2 = np.fromfile('../../data/W_2.npy').reshape(3,4,128)
W_1 = np.fromfile('../../data/W_1.npy').reshape(3,1,128)

W_3 = torch.from_numpy(W_3).float().cuda()
W_2 = torch.from_numpy(W_2).float().cuda()
W_1 = torch.from_numpy(W_1).float().cuda()


equation_main = "...ik,ekc,bci,be -> bc..."
equation_weighting = "...k,ekc,be->bc..."
equation_contract = "bc...i,bci->bc..."
            
correlation = 3

U_tensors = {3: U_3, 2:  U_2, 1: U_1}
W_tensors = {3: W_3, 2: W_2, 1: W_1}

nrepeats = 1

X = torch.from_numpy(X).float().cuda().repeat(nrepeats, 1, 1)
Y = torch.from_numpy(Y).float().cuda().repeat(nrepeats, 1)

X_torch = X.transpose(-1, -2).contiguous()
atom_types = np.array([1,1,1,1,1,1,1,2,2,2,1,1,2,0,0,0,0,0,0,0,0])
atom_types_torch = torch.from_numpy(atom_types).repeat(nrepeats).int().cuda()

#print (X.shape, Y.shape, atom_types.shape)


def mace_version_1(U_tensors, W_tensors, X, Y, requires_grad=False):

    equation_main = "...ik,ekc,bci,be -> bc..."
    equation_weighting = "...k,ekc,be->bc..."
    equation_contract = "bc...i,bci->bc..."

    if requires_grad:
        X.requires_grad = True

    outputs = {}

    out = contract(equation_main, U_tensors[3],W_tensors[3], X, Y)
    outputs[3] = out

    for corr in range(correlation - 1, 0, -1):      

        c_tensor = contract(
            equation_weighting,
            U_tensors[corr],
            W_tensors[corr],
            Y,
        )
        
        print ("c_tensor shape:", c_tensor.shape)
        print ("out shape:", out.shape)
        print ("v1 corr: ", corr, c_tensor[0][0])

        c_tensor = c_tensor + out
        
        out = contract(equation_contract, c_tensor, X)

        outputs[corr] = out

    if (requires_grad):
        out.sum().backward()

    return outputs

equation_contract_weights = '...ik, ekc -> ...iec'

UW_tensors = {}

for corr in range(correlation, 0, -1):
    uw_torch = contract(equation_contract_weights, U_tensors[corr],W_tensors[corr])
    UW_tensors[corr] = uw_torch

    print (uw_torch.shape)
    
def mace_version_2(UW_tensors, X, Y):

    outputs = {}

    out = contract ('...iec, bci, be -> bc...', UW_tensors[3], X, Y)
    
    outputs[3] = out

    for corr in range(correlation - 1, 0, -1):      
        
        #equation_weighting = "...k,ekc,be->bc..."

        c_tensor = contract('...iec, be -> bc...i', UW_tensors[corr], Y)
        

        c_tensor = contract(
            equation_weighting,
            U_tensors[corr],
            W_tensors[corr],
            Y,
        )

        print (c_tensor.shape)
        print (out.shape)

        print ("v2 corr: ", corr, c_tensor[0][0])

        

        c_tensor = c_tensor + out
        
        out = contract(equation_contract, c_tensor, X)

        outputs[corr] = out

    return outputs

def cuda_version_1(UW_tensors, U3W_non_sparse_indices,U3W_num_nonsparse, X, atom_types):

    outputs = {}

    out3, grad3 = correlation_3_main(UW_tensors[3], U3W_non_sparse_indices,U3W_num_nonsparse,
                        X_torch, atom_types_torch, X.shape[0], 4, 1, 64, 2, 1)

    out2, grad2 = correlation_2_contraction(UW_tensors[2], out3,  X_torch, grad3, atom_types_torch, X.shape[0], 1, 1, 64, 1, 1)


    out1, grad1 = correlation_1_contraction(UW_tensors[1], out2, X_torch, grad2, atom_types_torch, X.shape[0], 1, 1, 64, 1, 1)

    outputs[3] = out3
    outputs[2] = out2
    outputs[1] = out1

    return outputs

torch.matmul(torch.rand(1024, 1024, device='cuda'),torch.rand(1024, 1024, device='cuda'))

# start = time()

# for i in range(1000):
#     outputs_v1 = mace_version_1(U_tensors, W_tensors, X, Y, requires_grad=False)

# end = time()

# print (end - start)

# start = time()
# for i in range(1000):
#     outputs_v2 =mace_version_2(UW_tensors, X, Y)

# end = time()

# print (end - start)



U3W_non_sparse_indices = torch.zeros((U_tensors[3].shape[0], U_tensors[3].shape[1], 3), dtype=torch.int32).cuda()
U3W_num_nonsparse = torch.zeros((U_tensors[3].shape[0], U_tensors[3].shape[1]), dtype=torch.int32).cuda()

for i in range(UW_tensors[3].shape[0]):
    for j in range(UW_tensors[3].shape[1]):

        idx, edx, ldx  = torch.where(UW_tensors[3][i, j] != 0.0)
        
        idx = torch.unique(idx)

        if (idx.shape[0] > 0):

            U3W_non_sparse_indices[i, j, :idx.shape[0]] = idx
            U3W_num_nonsparse[i, j] = idx.shape[0]




# with open("log.txt", "w") as log:
#     try:
#         outputs_cuda = cuda_version_1(UW_tensors,U3W_non_sparse_indices,U3W_num_nonsparse, X_torch, atom_types_torch)
#     except Exception:
#         traceback.print_exc(file=log)




""" start = time()
for i in range(1000):
    outputs_cuda = cuda_version_1(UW_tensors,U3W_non_sparse_indices,U3W_num_nonsparse, X_torch, atom_types_torch)

end = time()
print ("cuda total time:", end - start) """

X_torch.requires_grad = True

start = time()
for i in range(1000):
    out3, grad3 = correlation_3_main(UW_tensors[3], U3W_non_sparse_indices,U3W_num_nonsparse,
                        X_torch, atom_types_torch, X.shape[0], 4, 1, 64, 2, 1)
end = time()
#print ("cuda correlation 3 time:", end - start)

X_torch.requires_grad = True
start = time()
for i in range(1000):
    out2, grad2 = correlation_2_contraction(UW_tensors[2], out3,  X_torch, grad3, atom_types_torch, X.shape[0], 1, 1, 64, 1, 1)
end = time()
#print ("cuda correlation 2 time:", end - start)

start = time()
for i in range(1000):
    out1, grad1 = correlation_1_contraction(UW_tensors[1], out2, X_torch, grad2, atom_types_torch, X.shape[0], 1, 1, 64, 1, 1)
end = time()
#print ("cuda correlation 1 time:", end - start)





#hello(32, 2, 1)

#X.requires_grad = True
#out = contract ('...iec, bci, be -> bc...', UW_tensors[3], X, Y)

#out.sum().backward()

X_torch = X.transpose(-1, -2)  # (21, 16, 128)

test_3 = torch.zeros((X_torch.shape[0], 128, 16, 16), device='cuda') #torch.Size([21, 128, 16, 16])
deriv_3 = torch.zeros((X_torch.shape[0], 128, 16), device='cuda')
# torch.Size([16, 16, 16, 3, 128]), torch.Size([21, 16, 128]) -> torch.Size([21, 16, 16, 128])
for atom in range (X_torch.shape[0]):

    atom_type = atom_types_torch[atom]

    for i in range (X_torch.shape[1]):

        uw3 = UW_tensors[3][..., i, atom_type, :]

        #print (uw3.shape) #torch.Size([16, 16, 128])

        #                      [16, 16, 128]                  # [128]
        v = uw3 *  X_torch[atom, i, :]

        deriv_3[atom, :, i] = uw3.sum(axis=0).sum(axis=0)

        test_3[atom] += v.transpose(-1, -2).transpose(-2, -3)

deriv_2 = torch.zeros((X_torch.shape[0], 128, 16), device='cuda')
test_2 = torch.zeros((X_torch.shape[0], 128, 16), device='cuda')

for atom in range (X_torch.shape[0]):

    atom_type = atom_types_torch[atom]

    for i in range (X_torch.shape[1]):
        for j in range (X_torch.shape[1]):

            uw2 = UW_tensors[2][j, i, atom_type, :]

            prev_layer = test_3[atom, :, j, i]

            uw2 += prev_layer 

            out = uw2 *  X_torch[atom, i, :]

            deriv_2[atom, :, i] += uw2
            test_2[atom, :, i] += out

#print (deriv_3[atom_check])



#print (test_2[atom_check])


X.requires_grad = True
outputs_v1 = mace_version_1(U_tensors, W_tensors, X, Y)

#print (outputs_v1[2][atom_check])

outputs_v1[2].sum().backward()

v1_grads = X.grad.clone()

X.grad.zero_()

outputs_v2 = mace_version_2(UW_tensors, X, Y)

# outputs_v2[2].sum().backward()

# v2_grads = X.grad.clone()

# X.grad.zero_()

#print (outputs_v1[1] - outputs_v2[1])
#print (deriv_2[atom_check])
#print (v1_grads[atom_check])
#print (v2_grads[atom_check])
#print (grad2[atom_check].transpose(-1, -2) - X.grad[atom_check])
#print (outputs[2][atom_check]- out2[atom_check].transpose(-1, -2))

outputs_cuda = cuda_version_1(UW_tensors,U3W_non_sparse_indices,U3W_num_nonsparse, X_torch, atom_types_torch)

atom_check = 0
print ("output 3 diff:", outputs_v1[3][atom_check] - outputs_cuda[3][atom_check].transpose(-1, -2).transpose(-2, -3))
print ("output 2 diff:",outputs_v1[2][atom_check] - outputs_cuda[2][atom_check].transpose(-1, -2))
print ("output 1 diff:",outputs_v1[1][atom_check] - outputs_cuda[1][atom_check])
