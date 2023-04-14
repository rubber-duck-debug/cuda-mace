import numpy as np
from time import time
from opt_einsum import contract
import torch
from tensor_contraction.cuda.tensor_contraction import *
from numba import cuda

        
X = np.fromfile('../../data/X.npy').reshape(21, 128, 16)
atom_types = np.array([1,1,1,1,1,1,1,2,2,2,1,1,2,0,0,0,0,0,0,0,0])

atom_types_torch = torch.from_numpy(atom_types).int().cuda()

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

X = torch.from_numpy(X).float().cuda()
Y = torch.from_numpy(Y).float().cuda()


torch.matmul(torch.rand(1024, 1024, device='cuda'),torch.rand(1024, 1024, device='cuda'))
start = time()


outputs = {}

out = contract(equation_main, U_3,W_3, X, Y)
outputs[3] = out

for corr in range(correlation - 1, 0, -1):      

    
    c_tensor = contract(
        equation_weighting,
        U_tensors[corr],
        W_tensors[corr],
        Y,
    )
    
    c_tensor = c_tensor + out
    
    out = contract(equation_contract, c_tensor, X)

    outputs[corr] = out

equation_contract_weights = '...ik, ekc -> ...iec'

UW3_torch = contract(equation_contract_weights, U_tensors[3],W_tensors[3])
UW2_torch = contract(equation_contract_weights, U_tensors[2],W_tensors[2])
UW1_torch = contract(equation_contract_weights, U_tensors[1],W_tensors[1])

test_3 = torch.zeros_like(outputs[3]) #torch.Size([21, 128, 16, 16])

print ("--out 1 shape--")
print ("output:",test_3.shape)
print ("uw3:",UW3_torch.shape)

X_torch = X.transpose(-1, -2)  # (21, 16, 128)

# torch.Size([16, 16, 16, 3, 128]), torch.Size([21, 16, 128]) -> torch.Size([21, 16, 16, 128])
for atom in range (X_torch.shape[0]):

    atom_type = atom_types_torch[atom]

    for i in range (X_torch.shape[1]):
        uw3 = UW3_torch[..., i, atom_type, :]

        #                      [16, 16, 128]                  # [128]
        v = UW3_torch[..., i, atom_type, :] *  X_torch[atom, i, :]


        test_3[atom] += v.transpose(-1, -2).transpose(-2, -3)
    

idx =  torch.where(outputs[3][atom] - test_3[atom] > 1e-7)
print (idx)

test_2 = torch.zeros_like(outputs[2])


print ("--out 2 shape--")
print ("output:", test_2.shape)
print ("uw2:", UW2_torch.shape) #torch.Size([16, 16, 3, 128])

for atom in range (X_torch.shape[0]):

    atom_type = atom_types_torch[atom]

    for i in range (X_torch.shape[1]):

        uw2 = UW2_torch[..., i, atom_type, :] +  test_3[atom, :, :, i].transpose(-1, -2) #torch.Size([16, 128]) 

        v = uw2 *  X_torch[atom, i, :]

        test_2[atom] += v.transpose(-1, -2)

idx =  torch.where(outputs[2][atom] - test_2[atom] > 1e-7)
print (idx)

test_1 = torch.zeros_like(outputs[1])

for atom in range (X_torch.shape[0]):

    atom_type = atom_types_torch[atom]

    for i in range (X_torch.shape[1]):
        
        uw1 = UW1_torch[i, atom_type, :] +  test_2[atom, :, i]

        v = uw1 *  X_torch[atom, i, :]

        test_1[atom] += v

idx =  torch.where(outputs[1][atom] - test_1[atom] > 1e-7)
print (idx)

U3W_non_sparse_indices = torch.zeros((U_tensors[3].shape[0], U_tensors[3].shape[1], 3), dtype=torch.int32).cuda()
U3W_num_nonsparse = torch.zeros((U_tensors[3].shape[0], U_tensors[3].shape[1]), dtype=torch.int32).cuda()


for i in range(UW3_torch.shape[0]):
    for j in range(UW3_torch.shape[1]):

        idx, edx, ldx  = torch.where(UW3_torch[i, j] != 0.0)
        
        idx = torch.unique(idx)

        if (idx.shape[0] > 0):

            U3W_non_sparse_indices[i, j, :idx.shape[0]] = idx
            U3W_num_nonsparse[i, j] = idx.shape[0]

U2W_non_sparse_indices = torch.zeros((UW2_torch.shape[0], 1), dtype=torch.int32).cuda()

count = 0
for i in range(UW2_torch.shape[0]):
        idx, edx, ldx = torch.where(UW2_torch[i] != 0.0)
        idx = torch.unique(idx)

        if (idx.shape[0] > 0):
            U2W_non_sparse_indices[i, :idx.shape[0]] = idx


            count +=1

out3 = correlation_3_main(UW3_torch,
                        U3W_non_sparse_indices,
                        U3W_num_nonsparse,
                        X_torch,
                        atom_types_torch, 21, 16, 1, 32, 16, 1)

print (out3.shape, outputs[2].shape)

idx =  torch.where(outputs[3][atom] - out3[atom].transpose(-1, -2).transpose(-2, -3) > 1e-7)
print (idx)

print ("test:", out3.shape)

out2 = correlation_2_contraction(UW2_torch, out3, X_torch, atom_types_torch, 21, 1, 1, 32, 8, 1)

print (out2.shape, outputs[2].shape)

idx =  torch.where(outputs[2][atom] - out2[atom].transpose(-1, -2) > 1e-7)

if (len(idx[0]) > 0):
    print (idx)
    print ("problem with cuda correlation_2")

test_2 = torch.zeros_like(outputs[2])

for atom in range (X_torch.shape[0]):

    atom_type = atom_types_torch[atom]

    for i in range (UW2_torch.shape[0]):
        for j in range (UW2_torch.shape[1]):

            #uw2: torch.Size([16, 16, 3, 128])
            #test_3: torch.Size([21, 128, 16, 16])
            #test_2: torch.Size([21, 128, 16])
            #out3: torch.Size([21, 16, 16, 128])
            #X_torch:  # (21, 16, 128)
            """         uw2_nonsparse = UW2_torch[..., i, atom_type, :] +  test_3[atom, :, :, i].transpose(-1, -2) 

            v_nonsparse = uw2_nonsparse *  X_torch[atom, i, :]

            test_2_nonsparse[atom] += v_nonsparse.transpose(-1, -2)
            """
            uw_sample =  UW2_torch[i, j, atom_type, :]

            uw2 = UW2_torch[i, j, atom_type, :] +  test_3[atom, :, i, j]

            v = uw2 *  X_torch[atom, j, :]

            test_2[atom, :, i] += v
            
idx =  torch.where(outputs[2][atom] - test_2[atom] > 1e-7)

test_1 = torch.zeros_like(outputs[1])

for atom in range (X_torch.shape[0]):

    atom_type = atom_types_torch[atom]

    for i in range (UW1_torch.shape[0]):

            uw1 = UW1_torch[i, atom_type, :] +  test_2[atom, :, i]

            v = uw1 *  X_torch[atom, i, :]

            test_1[atom, :] += v

idx =  torch.where(outputs[1][atom] - test_1[atom] > 1e-7)


if (len(idx[0]) > 0):
    print (idx)
    print ("problem with correlation_1")

out1 = correlation_1_contraction(UW1_torch, out2, X_torch, atom_types_torch, 21, 1, 1, 32, 1, 1)

idx =  torch.where(outputs[1][atom] - out1[atom] > 1e-7)

if (len(idx[0]) > 0):
    print (idx)
    print ("problem with correlation_1 cuda")

start = time()
for i in range (1000):
    out3 = correlation_3_main(UW3_torch,
                        U3W_non_sparse_indices,
                        U3W_num_nonsparse,
                        X_torch,
                        atom_types_torch, 21, 16, 1, 32, 16, 1)
end = time()

print (end - start)

start = time()
for i in range (1000):
    out2 = correlation_2_contraction(UW2_torch, out3, X_torch, atom_types_torch, 21, 1, 1, 32, 8, 1)
end = time()

print (end - start)

for i in range (1000):
    out1 = correlation_1_contraction(UW1_torch, out2, X_torch, atom_types_torch, 21, 1, 1, 64, 4, 1)
end = time()

print (end - start)

#print (outputs[2][atom])
#print (test_2[atom])




