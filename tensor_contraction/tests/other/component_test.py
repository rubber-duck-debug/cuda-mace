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

correlation = 3

U_tensors = {3: U_3, 2:  U_2, 1: U_1}
W_tensors = {3: W_3, 2: W_2, 1: W_1}

nrepeats = 1

X = torch.from_numpy(X).float().cuda().repeat(nrepeats, 1, 1)
Y = torch.from_numpy(Y).float().cuda().repeat(nrepeats, 1)

X_torch = X.transpose(-1, -2).contiguous()
atom_types = np.array([1,1,1,1,1,1,1,2,2,2,1,1,2,0,0,0,0,0,0,0,0])
atom_types_torch = torch.from_numpy(atom_types).repeat(nrepeats).int().cuda()


equation_main = "...ik,ekc,bci,be -> bc..."
equation_weighting = "...k,ekc,be->bc..."
equation_contract = "bc...i,bci->bc..."

equation_contract_weights = '...ik, ekc -> ...iec'

UW_tensors = {}
for corr in range(correlation, 0, -1):
    uw_torch = contract(equation_contract_weights, U_tensors[corr],W_tensors[corr])
    UW_tensors[corr] = uw_torch

    print (f"uw {corr}", uw_torch.shape)

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

nradial = 2
nl = 2

UW_tensors[3] = torch.rand((nl,nl,nl,3,nradial), device='cuda')
UW_tensors[2] = torch.rand((nl,nl,3,nradial), device='cuda')
UW_tensors[1] = torch.rand((nl,3,nradial), device='cuda')
X = torch.rand((21, nradial, nl), device='cuda')

#outputs_v1, grad_v1 = mace_v1(U_tensors, W_tensors, X, Y, 3, 1, requires_grad = True)
outputs_v2, grad_v2 = mace_v2(UW_tensors, X, Y, 3, 1, requires_grad = True)

atom_check = 0

X_torch = X.transpose(-1, -2)  # (21, 16, 128)

test_3 = torch.zeros((X_torch.shape[0], nradial, nl, nl), device='cuda') #torch.Size([21, 128, 16, 16])
deriv_3 = torch.zeros((X_torch.shape[0], nradial, nl), device='cuda')


atom_type = atom_types_torch[atom_check]

v3 =  torch.zeros((nradial, nl, nl), device='cuda')
deriv3 =  torch.zeros((nradial, nl), device='cuda')
deriv2 =  torch.zeros((nradial, nl), device='cuda')

v2 =  torch.zeros((nradial, nl), device='cuda')

# UW_tensors[3] # torch.Size([16, 16, 16, 3, 128])

for j in range (UW_tensors[3].shape[0]):

    for k in range(UW_tensors[3].shape[1]):

        for i in range (UW_tensors[3].shape[2]):

            uw3 = UW_tensors[3][j, k, i, atom_type, :]

            v3[:, j, k] += uw3 *  X_torch[atom_check, i, :]

            deriv3[:, i] += uw3

for j in range (UW_tensors[2].shape[0]):
    for k in range (UW_tensors[2].shape[1]):

        uw2 = UW_tensors[2][j, k, atom_type, :]
        uw3 = UW_tensors[3][j, k, k, atom_type, :]

        v2[:, j] += (v3[:, j, k] + uw2) *  X_torch[atom_check, k, :] # (f(x) + c) * x = f(x)x + cx, d/dx = f'(x)x + f(x) + c 

        deriv2[:, k] += deriv3[:, k] + uw3 *  X_torch[atom_check, k, :]  + uw2


print ("v3 diff:", v3 - outputs_v2[3][atom_check])
print ("v2_diff:", v2 - outputs_v2[2][atom_check])

print (grad_v2[atom_check])
#print (deriv3)
print (deriv2)

