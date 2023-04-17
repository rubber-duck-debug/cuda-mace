import numpy as np
from time import time
from opt_einsum import contract
import torch
from tensor_contraction.cuda.tensor_contraction import *
from tensor_contraction.cuda.symmetric_contraction import *
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

    print (f"uw_torch {corr}", uw_torch.shape)


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


atom_check = 0

outputs_v1, grad_v1 = mace_v1(U_tensors, W_tensors, X, Y, 3, 0, requires_grad = True)



def forward(x,atom_types, UW3, UW2, UW1):
    b1 = torch.einsum('bde,bg,eicgd->bdic', x, atom_types, UW3)
    b1 = b1 + torch.einsum('bg,icgd->bdic', atom_types, UW2)
    b2 = torch.einsum('bdi,bdic->bdc', x, b1)
    b2 = b2 + torch.einsum('bg,cgd->bdc', atom_types, UW1)
    b3 = torch.einsum('bdc,bdc->bd', x, b2)
    return b3, b2, b1

out1, out2, out3 = forward(X, Y, UW_tensors[3], UW_tensors[2], UW_tensors[1])


class SymmetricContraction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, atom_types, UW3, UW2, UW1):
        b1 = torch.einsum('bde,bg,eicgd->bdic', x, atom_types, UW3)
        b1 = b1 + torch.einsum('bg,icgd->bdic', atom_types, UW2)
        b2 = torch.einsum('bdi,bdic->bdc', x, b1)
        b2 = b2 + torch.einsum('bg,cgd->bdc', atom_types, UW1)
        b3 = torch.einsum('bdc,bdc->bd', x, b2)

        ctx.save_for_backward(X, b2, b1, UW3, UW2, UW1, atom_types)

        return b3

    @staticmethod
    def backward(ctx, grad_output):
        X, b2, b1, UW3, UW2, UW1, atom_types = ctx.saved_tensors

        grad_b2 = torch.einsum('bd,bdc->bdc', grad_output, X)
        grad_b1 = torch.einsum('bdc,bdi->bdic', grad_b2, X)
        grad_x = torch.einsum('bdic,bg,eicgd->bde', grad_b1, atom_types, UW3)
        grad_x +=  torch.einsum('bdc,bdic->bdi', grad_b2, b1)
        grad_x +=  torch.einsum('bd,bdc->bdc', grad_output, b2)
        return grad_x, None, None, None, None

X.requires_grad = True
out_fn = SymmetricContraction.apply(X, Y, UW_tensors[3], UW_tensors[2], UW_tensors[1])
out_fn.sum().backward()



print (out_fn[atom_check])
print (outputs_v1[1][atom_check])

print (X.grad[atom_check])
print (grad_v1[atom_check])


def backward(grad_output, x, UW3, UW2, UW1, b1, b2):
    grad_b2 = torch.einsum('bd,bdc->bdc', grad_output, x)
    grad_b1 = torch.einsum('bdc,bdi->bdic', grad_b2, x)
    grad_x = torch.einsum('bdic,eic->bde', grad_b1, w1)
    grad_x +=  torch.einsum('bdc,bdic->bdi', grad_b2, b1)
    grad_x +=  torch.einsum('bd,bdc->bdc', grad_output, b2)
    return grad_x