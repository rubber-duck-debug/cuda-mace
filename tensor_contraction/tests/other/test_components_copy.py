import numpy as np
from time import perf_counter_ns, time
import torch
from opt_einsum import contract
import math
import os
from tensor_contraction.cuda.tensor_contraction import *

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

U3_torch = torch.from_numpy(U3).float().cuda()
W3_torch =  torch.from_numpy(W3).float().cuda()

U2_torch = torch.from_numpy(U2).float().cuda()
W2_torch =  torch.from_numpy(W2).float().cuda()

U1_torch = torch.from_numpy(U1).float().cuda()
W1_torch =  torch.from_numpy(W1).float().cuda()

#self.equation_main = "...ik,ekc,bci,be -> bc..."
#self.equation_weighting = "...k,ekc,be->bc..."
#self.equation_contract = "bc...i,bci->bc..."

equation_contract_weights = '...ik, ekc -> ...iec'
#equation_main = "...iec,bic,be -> b...c"
equation_weighting = "...k,ekc,be->b...c"
equation_contract = "bi...c,bic->b...c"
equation_main = "...ik,ekc,bci,be -> bc..."


X_torch = torch.from_numpy(X).float().cuda()
Y_torch = torch.from_numpy(Y).float().cuda()

atom_types = np.array([1,1,1,1,1,1,1,2,2,2,1,1,2,0,0,0,0,0,0,0,0])

atom_types_torch = torch.from_numpy(atom_types).int().cuda()

U_tensors = {3: U3_torch.float(), 2:  U2_torch.float(), 1: U1_torch.float()}
W_tensors = {3: W3_torch.float(), 2: W2_torch.float(), 1: W1_torch.float()}

UW3_torch = contract(equation_contract_weights, U_tensors[3],W_tensors[3])
UW2_torch = contract(equation_contract_weights, U_tensors[2],W_tensors[2])
UW1_torch = contract(equation_contract_weights, U_tensors[1],W_tensors[1])

print ("--UW3--")
print (UW3_torch.shape) # torch.Size([16, 16, 16, 3, 128])
print ("--UW2--")
print (UW2_torch.shape)
print ("--UW1--")
print (UW1_torch.shape)

print (X_torch.shape) #torch.Size([21, 16, 128])
print (Y_torch.shape) #torch.Size([21, 3])

outputs = {}

out = contract(equation_main,  U_tensors[3],W_tensors[3], X_torch, Y_torch)
outputs[3] = out

correlation = 3

for corr in range(correlation - 1, 0, -1):      

    
    c_tensor = contract(
        equation_weighting,
        U_tensors[corr],
        W_tensors[corr],
        Y_torch,
    )

    print (c_tensor.shape, out.shape)
    

    c_tensor = c_tensor + out
    
    out = contract(equation_contract, c_tensor, X)

    outputs[corr] = out


test_3 = torch.zeros_like(out3_torch)

X_torch = X_torch.transpose(-1, -2)  # (21, 16, 128)

# torch.Size([16, 16, 16, 3, 128]), torch.Size([21, 16, 128]) -> torch.Size([21, 16, 16, 128])
for atom in range (X_torch.shape[0]):

    atom_type = atom_types_torch[atom]

    for i in range (X_torch.shape[1]):

        #                      [16, 16, 128]                  # [128]                                                       
        test_3[atom] += UW3_torch[..., i, atom_type, :] *  X_torch[atom, i, :]

test_2 = torch.zeros_like(outputs[2])

for atom in range (X_torch.shape[0]):

    atom_type = atom_types_torch[atom]

    uw2 = UW2_torch[..., atom_type, :]

    for i in range (X_torch.shape[1]):

        #                      [16, 16, 128]                  # [128]                                                       
        test_2[atom] += uw2 *  X_torch[atom, i, :]