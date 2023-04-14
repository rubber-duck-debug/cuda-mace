import numpy as np
from time import time
from opt_einsum import contract
import torch
from tensor_contraction.cuda.tensor_contraction import *
from numba import cuda
import logging
import traceback

X = torch.arange(3).float().repeat(21, 2, 1)
atom_types = torch.zeros(21)

UW3 = torch.Tensor([0.1, 0.5]).repeat(3, 3, 3, 1)
UW2 = torch.Tensor([0.1, 0.5]).repeat(3, 3, 1)

print (X.shape)
print (UW3.shape)
print (UW2.shape)

equation_main = "...ik,ekc,bci,be -> bc..."
equation_weighting = "...k,ekc,be->bc..."
equation_contract = "bc...i,bci->bc..."

out3 = contract ('...ic, bci -> bc...', UW3, X)

c_tensor = contract('...ic, b... -> bc...i', UW2, atom_types)

c_tensor = c_tensor + out3

out2 = contract(equation_contract, c_tensor, X)

print (X.shape)
print (out3.shape)


test2 = torch.zeros_like(out2)

for atom in range (X.shape[0]):

    for i in range (X.shape[2]):

        uw2 = UW2[..., i, :]
        uw3 = UW3[..., i, :]

        #print (uw2.shape, out3[atom, ..., i].transpose(-1, -2).shape, X[atom, :, i].shape)

        v = (uw2 + out3[atom, ..., i].transpose(-1, -2)) *  X[atom, :, i]

        

        test2[atom] += v.transpose(-1, -2)

        #deriv_2[atom, :, i] = uw2.sum(axis=0) + outputs_v1[3][atom, ..., i].sum(axis=-1) +  X_torch[atom, i, :] * deriv_3[atom, :, i]

print (out2[atom])
print (test2[atom])