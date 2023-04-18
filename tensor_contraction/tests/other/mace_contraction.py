import numpy as np
from time import time
from opt_einsum import contract
import torch
from tensor_contraction.cuda.tensor_contraction import *
from tensor_contraction.cuda.symmetric_contraction import *
from numba import cuda
import logging
import traceback

n_equivariants = 4

X = np.fromfile('../../data/X.npy').reshape(21, 128, 16)

Y = np.fromfile('../../data/Y.npy').reshape(21, 3)

U_3 = np.fromfile('../../data/U_3.npy').reshape(16,16,16,23)
U_2 = np.fromfile('../../data/U_2.npy').reshape(16,16, 4)
U_1 = np.fromfile('../../data/U_1.npy').reshape(16,1)

U_3 = torch.from_numpy(U_3).float().cuda()
U_2 = torch.from_numpy(U_2).float().cuda()
U_1 = torch.from_numpy(U_1).float().cuda()

W_3 = np.fromfile('../../data/W_3.npy').reshape(3,23,128)

W_2 = np.fromfile('../../data/W_2.npy').reshape(3,4, 128)

W_1 = np.fromfile('../../data/W_1.npy').reshape(3,1, 128)


W_3 = torch.from_numpy(W_3).float().cuda()
W_3 = W_3.repeat(1, 1, n_equivariants)
W_2 = torch.from_numpy(W_2).float().cuda()
W_2 = W_2.repeat(1, 1, n_equivariants)
W_1 = torch.from_numpy(W_1).float().cuda()
W_1 = W_1.repeat(1, 1, n_equivariants)

        
correlation = 3

U_tensors = {3: U_3, 2:  U_2, 1: U_1}
W_tensors = {3: W_3, 2: W_2, 1: W_1}

nrepeats = 25

X = torch.from_numpy(X).float().cuda().repeat(nrepeats, n_equivariants, 1)
Y = torch.from_numpy(Y).float().cuda().repeat(nrepeats, 1)

X_torch = X.transpose(-1, -2).contiguous()
atom_types = np.array([1,1,1,1,1,1,1,2,2,2,1,1,2,0,0,0,0,0,0,0,0])
atom_types_torch = torch.from_numpy(atom_types).repeat(nrepeats).int().cuda()


equation_main = "...ik,ekc,bci,be -> bc..."
equation_weighting = "...k,ekc,be->bc..."
equation_contract = "bc...i,bci->bc..."

equation_contract_weights = '...ik, ekc -> ...iec'

U3W_non_sparse_indices = torch.zeros((16,16, 3), dtype=torch.uint8).cuda()
U3W_num_nonsparse = torch.zeros((16, 16), dtype=torch.uint8).cuda()

U2W_non_sparse_indices = torch.zeros((16,1), dtype=torch.uint8).cuda()
U2W_num_nonsparse = torch.zeros((16), dtype=torch.uint8).cuda()

UW_tensors = {}
for corr in range(correlation, 0, -1):
    uw_torch = contract(equation_contract_weights, U_tensors[corr],W_tensors[corr])
    UW_tensors[corr] = uw_torch

    print (f"uw_torch {corr}", uw_torch.shape)

UW3_deriv_factors = torch.zeros_like(UW_tensors[3])

for i in range(UW_tensors[3].shape[0]):
    for j in range(UW_tensors[3].shape[1]):

        idx, edx, ldx  = torch.where(UW_tensors[3][i, j] != 0.0)
        
        idx = torch.unique(idx)

        if (idx.shape[0] > 0):

            U3W_non_sparse_indices[i, j, :idx.shape[0]] = idx
            U3W_num_nonsparse[i, j] = idx.shape[0]

            for e in range(UW_tensors[3].shape[3]):
            
                #deriv_1_j_tmp2 += (uw3_ijk + uw3_jki + uw3_jik) * Xk;
                UW3_deriv_factors[i,j,idx,e,:] = UW_tensors[3][i,j,idx,e,:] + UW_tensors[3][j,idx,i,e,:] + UW_tensors[3][j,i,idx,e,:]

for i in range(UW_tensors[3].shape[0]):

    idx, edx, ldx  = torch.where(UW_tensors[2][i] != 0.0)
    
    idx = torch.unique(idx)

    if (idx.shape[0] > 0):

        U2W_non_sparse_indices[i, :idx.shape[0]] = idx
        U2W_num_nonsparse[i] = idx.shape[0]




def mace_v1(U_tensors, W_tensors, X, Y, correlation, correlation_min=0, requires_grad = True):

    X_copy = X.clone()
    X_copy.requires_grad = requires_grad
    Y_copy = Y.clone()

    outputs = {}

    out_v1 = contract(equation_main, U_tensors[correlation],W_tensors[correlation], X_copy, Y_copy)

    #out_v1 = torch.ones_like(out_v1)

    outputs[correlation] = out_v1

    for corr in range(correlation - 1, correlation_min, -1):      

        c_tensor_v1 = contract(
            equation_weighting,
            U_tensors[corr],
            W_tensors[corr],
            Y_copy,
        )

        c_tensor_v1 = c_tensor_v1 + out_v1
        
        #equation_contract = "bc...i,bci->bc..."
        out_v1 = contract(equation_contract, c_tensor_v1, X_copy)

        outputs[corr] = out_v1

    if (requires_grad):
        out_v1.sum().backward()

    if (requires_grad):
        return outputs, X_copy.grad.clone()


def mace_v2(UW_tensors, X, Y, correlation, correlation_min=0, requires_grad = False):

    X_copy = X.clone()
    X_copy.requires_grad = requires_grad
    Y_copy = Y.clone()

    outputs = {}

    out_v2 = contract ('...iec, bci, be -> bc...', UW_tensors[correlation], X_copy, Y_copy)

    #out_v2 = torch.ones_like(out_v2)

    outputs[correlation] = out_v2

    for corr in range(correlation - 1, correlation_min, -1):      

        c_tensor_v2 = contract('...iec, be -> bc...i', UW_tensors[corr], Y_copy)

        c_tensor_v2 = c_tensor_v2 + out_v2
        
        #equation_contract = "bc...i,bci->bc..."
        out_v2 = contract(equation_contract, c_tensor_v2, X_copy)

        outputs[corr] = out_v2

    if (requires_grad):
        out_v2.sum().backward()

    if (requires_grad):
        return outputs, X_copy.grad.clone()

    return outputs

outputs_v1, grad_v1 = mace_v1(U_tensors, W_tensors, X, Y, 3, 0, requires_grad = True)
outputs_v2, grad_v2 = mace_v2(UW_tensors, X, Y, 3, 0, requires_grad = True)

atom_check = 0

#print(outputs_v1[2][atom_check])
#print(outputs_v2[2][atom_check])

X_torch = X.transpose(-1, -2).contiguous()  # (21, 16, 128)

start = time()
for i in range (1000):
    outputs_v1, grad_v1 = mace_v1(U_tensors, W_tensors, X, Y, 3, 0, requires_grad = True)

end = time()

print (end - start)


start = time()
for i in range (1000):
    out = sparse_symmetric_contraction(U3W_non_sparse_indices, U3W_num_nonsparse, UW_tensors[3], UW_tensors[2], UW_tensors[1], X_torch, atom_types_torch,1,1,1,8,16,1)

end = time()

print ("forwards only:", end - start)

start = time()
for i in range (1000):
    out, grad = sparse_symmetric_contraction_derivative(U3W_non_sparse_indices, U3W_num_nonsparse, 
                                                        UW_tensors[3], UW3_deriv_factors, UW_tensors[2], UW_tensors[1], 
                                                        X_torch, atom_types_torch,X_torch.shape[0],1,1,8,16,1)

end = time()
print ("grad time:", end - start)

print ("outputs 1")
print (outputs_v1[1][atom_check])
print (out[atom_check])

print ("grads")

idxs = torch.where(torch.abs(grad_v1[atom_check]-grad[atom_check].transpose(-1, -2) )  > 1e-6)

print (len(idxs[0]))

print (grad_v1[atom_check][idxs])
print (grad[atom_check].transpose(-1, -2)[idxs])


#atom_types_torch_i8 = atom_types_torch.type(torch.uint8)


start = time()
for i in range (1000):
    out3, out2, out1 = uw3_contraction(U3W_non_sparse_indices, U3W_num_nonsparse, UW_tensors[3], UW_tensors[2], UW_tensors[1], atom_types_torch, X_torch, X.shape[0], 1, 1, 16, 1, 8)

end = time()


print (end - start)

#print (outputs_v1[3][atom_check])
#print (out3[atom_check].transpose(-1, -2).transpose(-2, -3))

idxs = torch.where((torch.abs(outputs_v1[3][atom_check] - out3[atom_check].transpose(-1, -2).transpose(-2, -3) ))  > 1e-8)

print (len(idxs[0]))

idxs = torch.where((torch.abs(outputs_v1[2][atom_check] - out2[atom_check].transpose(-1, -2) ))  > 1e-8)

print (len(idxs[0]))

idxs = torch.where((torch.abs(outputs_v1[1][atom_check] - out1[atom_check] ))  > 1e-8)

print (len(idxs[0]))


#print (outputs_v1[1][atom_check])
#print (out1[atom_check])






