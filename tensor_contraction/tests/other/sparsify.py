import numpy as np
from time import time
from opt_einsum import contract
import torch
#from tensor_contraction.cuda.tensor_contraction import *
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
            
equation_contract_weights = '...ik, ekc -> ...iec'
correlation = 3

U_tensors = {3: U_3, 2:  U_2, 1: U_1}
W_tensors = {3: W_3, 2: W_2, 1: W_1}

UW_tensors = {}
for corr in range(correlation, 0, -1):
    uw_torch = contract(equation_contract_weights, U_tensors[corr],W_tensors[corr])
    UW_tensors[corr] = uw_torch

nrepeats = 50

X = torch.from_numpy(X).float().cuda().repeat(nrepeats, 1, 1)
Y = torch.from_numpy(Y).float().cuda().repeat(nrepeats, 1)

X_torch = X.transpose(-1, -2).contiguous()
atom_types = np.array([1,1,1,1,1,1,1,2,2,2,1,1,2,0,0,0,0,0,0,0,0])
atom_types_torch = torch.from_numpy(atom_types).repeat(nrepeats).int().cuda()

print (X.shape, Y.shape, atom_types.shape)

test_dict = {}


for i in range(U_3.shape[0]):
    for j in range(U_3.shape[1]):

        kdx, ldx   = torch.where(U_3[i, j] != 0.0)
        
        print (i, j, kdx, ldx)

print ("--U2 Sparsity--")
for i in range(U_2.shape[0]):
    kdx, ldx   = torch.where(U_2[i] != 0.0)
    
    print (i, kdx, ldx)





