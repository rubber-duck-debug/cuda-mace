import numpy as np
import sympy

import sympy as sym
from sympy import Array
from sympy.tensor.array.expressions import ArraySymbol
from sympy import derive_by_array
import torch
from opt_einsum import contract

torch.set_printoptions(precision=6)

def mace_v2(UW_tensors, X, Y, correlation, correlation_min=0, requires_grad = False):

    equation_main = "...ik,ekc,bci,be -> bc..."
    equation_weighting = "...k,ekc,be->bc..."
    equation_contract = "bc...i,bci->bc..."

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
nl = 3

X = ArraySymbol("X_np", (1, nl, nradial))

UW_tensors = {}
UW_tensors[3] =  ArraySymbol("UW3", (nl,nl,nl,3,nradial)) #Array(np.random.rand(nl,nl,nl,3,nradial))
UW_tensors[2] =  ArraySymbol("UW2", (nl,nl,3,nradial)) #Array(np.random.rand(nl,nl,3,nradial))
UW_tensors[1] =  ArraySymbol("UW1", (nl,3,nradial)) #Array(np.random.rand(nl,3,nradial))

UW_tensors_np = {}
UW_tensors_np[3] =  np.random.rand(nl,nl,nl,3,nradial)
UW_tensors_np[2] =  np.random.rand(nl,nl,3,nradial)
UW_tensors_np[1] = np.random.rand(nl,3,nradial)

UW_tensors_torch = {}
UW_tensors_torch[3] =  torch.from_numpy(UW_tensors_np[3])
UW_tensors_torch[2] = torch.from_numpy(UW_tensors_np[2])
UW_tensors_torch[1] =torch.from_numpy(UW_tensors_np[1])

X_np = np.random.rand(21, nl, nradial)
Y_np = np.fromfile('../../data/Y.npy').reshape(21, 3)

print (Y_np)

X_torch = torch.from_numpy(X_np.swapaxes(-1, -2))
Y_torch = torch.from_numpy(Y_np)

atom_type = at = 1
atom = 0

outputs_v2, grad_v2 = mace_v2(UW_tensors_torch,X_torch, Y_torch, 3,0, requires_grad = True)

v3 =  np.zeros((nradial, nl, nl))
v2 =  np.zeros((nradial, nl))
v1 =  np.zeros((nradial))

output3  = sym.MutableDenseNDimArray(v3, dtype=float)
output2  = sym.MutableDenseNDimArray(v2, dtype=float)
output1  = sym.MutableDenseNDimArray(v1, dtype=float)

print (output3.shape, X.shape)



#O3_dX_00 = derive_by_array(output3, X[0,0,0])
#O3_dX_01 = derive_by_array(output3, X[0,0,1])
#O3_dX_10 = derive_by_array(output3, X[0,1,0])
#O3_dX_11 = derive_by_array(output3, X[0,1,1])

O2_dX_00 = derive_by_array(output2, X[0,0,0])
O2_dX_01 = derive_by_array(output2, X[0,0,1])
#O2_dX_10 = derive_by_array(output2, X[0,1,0])

#O2_dX_11 = derive_by_array(output2, X[0,1,1])
#O2_dX_20 = derive_by_array(output2, X[0,2,0])
#O2_dX_21 = derive_by_array(output2, X[0,2,1])


#print ("O2_dx_10")
#print (O2_dX_10)
#print ("O2_dx_20")
#print (O2_dX_20)
                                      # j, k, i, 0, r
# UW_tensors[3] =  ArraySymbol("UW3", (nl,nl,nl,3,nradial))
                                      # j, k, 0, r
# UW_tensors[2] =  ArraySymbol("UW2", (nl,nl,3,nradial)) 

# X = ArraySymbol("X", (1, nl, nradial))

UW1 = UW_tensors_np[1]
UW2 = UW_tensors_np[2]
UW3 = UW_tensors_np[3]

for r in range(nradial):

    for j in range (UW_tensors[3].shape[0]):

        for k in range(UW_tensors[3].shape[1]):

            for i in range (UW_tensors[3].shape[2]):
                
                uw3 = UW_tensors[3][j, k, i, atom_type, r]
                uw3_np = UW3[j,k,i,atom_type, r]

                output3[r, j, k] += uw3 *  X[atom, i, r]
                v3[r, j, k] += uw3_np * X_np[atom, i , r]


    for j in range (UW_tensors[2].shape[0]): # 0 1 2

        uw1 = UW_tensors[1][j, atom_type, r]
        uw1_np = UW1[j, atom_type, r]
        for k in range (UW_tensors[2].shape[1]): # 0 1 2

            uw2 = UW_tensors[2][j, k, atom_type, r]
            uw2_np = UW2[j, k, atom_type, r]

            output2[r, j] += (output3[r, j, k] + uw2) *  X[atom, k, r]
            v2[r, j] += (v3[r, j, k] + uw2_np) *  X_np[atom, k, r]

        output1[r] += (output2[r,j] + uw1) * X[atom, j, r]
        v1[r] += (v2[r,j] + uw1_np) * X_np[atom, j, r]



if (2 in outputs_v2):
    print ("-- output 2 torch --")
    print (outputs_v2[2][atom])
    print ("-- output 2 np --")
    print (v2)

if (1 in outputs_v2):
    print ("-- output 1 torch --")
    print (outputs_v2[1][atom])
    print ("-- output 1 np --")
    print (v1)

print ("--grad_torch--")
print (grad_v2[atom])


deriv2 =  np.zeros((nradial, nl))

for r in range (nradial):

    for i in range (nl):

        for j in range (nl):
             
            deriv2[r, i] += UW2[j, i, at, r]

            for k in range(nl):
                deriv2[r, k] +=  UW3[j, k, i, at, r] * X_np[atom, i, r] + UW3[j, i, k, at, r] * X_np[atom, i, r]
                
deriv1 =  np.zeros((nradial, nl))

for r in range (nradial):
    for i in range (nl):

        deriv1_tmp = UW1[i, at, r]

        for j in range (nl):
             
            deriv_1_j_tmp = UW2[i, j, at, r]

            for l in range(nl):
                deriv_1_j_tmp += UW3[i,j,l,at,r] * X_np[atom, l, r]

            deriv_1_j_tmp *= X_np[atom, j, r]

            deriv_1_j_tmp2 = UW2[j, i, at, r]

            for l in range(nl):
                deriv_1_j_tmp2 += UW3[j, l, i, at, r] * X_np[atom, l, r] + UW3[j, i, l, at, r] * X_np[atom, l, r]

            deriv_1_j_tmp2 *= X_np[atom, j, r]

            deriv1_tmp += ( deriv_1_j_tmp + deriv_1_j_tmp2)

        deriv1[r, i] = deriv1_tmp


[(UW2[0, 0, 1, 0] + UW3[0, 0, 0, 1, 0]*X_np[0, 0, 0] + UW3[0, 0, 1, 1, 0]*X_np[0, 1, 0] + UW3[0, 0, 2, 1, 0]*X_np[0, 2, 0])*X_np[0, 0, 0] + 
(UW2[0, 1, 1, 0] + UW3[0, 1, 0, 1, 0]*X_np[0, 0, 0] + UW3[0, 1, 1, 1, 0]*X_np[0, 1, 0] + UW3[0, 1, 2, 1, 0]*X_np[0, 2, 0])*X_np[0, 1, 0] + 
(UW2[0, 2, 1, 0] + UW3[0, 2, 0, 1, 0]*X_np[0, 0, 0] + UW3[0, 2, 1, 1, 0]*X_np[0, 1, 0] + UW3[0, 2, 2, 1, 0]*X_np[0, 2, 0])*X_np[0, 2, 0] + 
(UW2[0, 0, 1, 0] + 2*UW3[0, 0, 0, 1, 0]*X_np[0, 0, 0] + UW3[0, 0, 1, 1, 0]*X_np[0, 1, 0] + UW3[0, 0, 2, 1, 0]*X_np[0, 2, 0] + UW3[0, 1, 0, 1, 0]*X_np[0, 1, 0] + 
UW3[0, 2, 0, 1, 0]*X_np[0, 2, 0])*X_np[0, 0, 0] + (UW2[1, 0, 1, 0] + 2*UW3[1, 0, 0, 1, 0]*X_np[0, 0, 0] + UW3[1, 0, 1, 1, 0]*X_np[0, 1, 0] + 
UW3[1, 0, 2, 1, 0]*X_np[0, 2, 0] + UW3[1, 1, 0, 1, 0]*X_np[0, 1, 0] + UW3[1, 2, 0, 1, 0]*X_np[0, 2, 0])*X_np[0, 1, 0] + (UW2[2, 0, 1, 0] + 
2*UW3[2, 0, 0, 1, 0]*X_np[0, 0, 0] + UW3[2, 0, 1, 1, 0]*X_np[0, 1, 0] + UW3[2, 0, 2, 1, 0]*X_np[0, 2, 0] + UW3[2, 1, 0, 1, 0]*X_np[0, 1, 0] + 
UW3[2, 2, 0, 1, 0]*X_np[0, 2, 0])*X_np[0, 2, 0] + UW1[0, 1, 0], 0]

print ("--manual deriv 2--")
print (deriv2)
print ("--manual deriv 1--")
print (deriv1)

basis = X[0,0,0]

O3_dX = derive_by_array(output1, basis)

print ("O3_dX")
print (O3_dX)




tmp = np.array([[2*UW2[0, 0, 1, 0] + 6*UW3[0, 0, 0, 1, 0]*X_np[0, 0, 0]], [0]])


print ("tmp")
print (tmp)
print (tmp.shape)
print (tmp.sum(axis=0))


# O2_dX_00
#   UW3[0, 0, 0, 0, 0]*X[0, 0, 0]
#   UW3[0, 0, 1, 0, 0]*X[0, 1, 0]
#   UW3[0, 0, 2, 0, 0]*X[0, 2, 0]

#   UW3[0, 0, 0, 0, 0]*X[0, 0, 0]
#   UW3[0, 1, 0, 0, 0]*X[0, 1, 0]
#   UW3[0, 2, 0, 0, 0]*X[0, 2, 0]

# O2_dx_10
#   UW3[0, 1, 0, 0, 0]*X[0, 0, 0]
#   UW3[0, 1, 1, 0, 0]*X[0, 1, 0]
#   UW3[0, 1, 2, 0, 0]*X[0, 2, 0]

#   UW3[0, 0, 1, 0, 0]*X[0, 0, 0]
#   UW3[0, 1, 1, 0, 0]*X[0, 1, 0]
#   UW3[0, 2, 1, 0, 0]*X[0, 2, 0]

# O2_dx_20
# UW3[0, 2, 0, 0, 0]*X[0, 0, 0] 
# UW3[0, 2, 1, 0, 0]*X[0, 1, 0]
# UW3[0, 2, 2, 0, 0]*X[0, 2, 0]

# UW3[0, 0, 2, 0, 0]*X[0, 0, 0]
# UW3[0, 1, 2, 0, 0]*X[0, 1, 0]
# UW3[0, 2, 2, 0, 0]*X[0, 2, 0]

'''  O2_dX_00
    0: UW2[0, 0, 0, 0] + 2*UW3[0, 0, 0, 0, 0]*X[0, 0, 0] + UW3[0, 0, 1, 0, 0]*X[0, 1, 0] + UW3[0, 0, 2, 0, 0]*X[0, 2, 0] + UW3[0, 1, 0, 0, 0]*X[0, 1, 0] + UW3[0, 2, 0, 0, 0]*X[0, 2, 0], 
    1: UW2[1, 0, 0, 0] + 2*UW3[1, 0, 0, 0, 0]*X[0, 0, 0] + UW3[1, 0, 1, 0, 0]*X[0, 1, 0] + UW3[1, 0, 2, 0, 0]*X[0, 2, 0] + UW3[1, 1, 0, 0, 0]*X[0, 1, 0] + UW3[1, 2, 0, 0, 0]*X[0, 2, 0], 
    2: UW2[2, 0, 0, 0] + 2*UW3[2, 0, 0, 0, 0]*X[0, 0, 0] + UW3[2, 0, 1, 0, 0]*X[0, 1, 0] + UW3[2, 0, 2, 0, 0]*X[0, 2, 0] + UW3[2, 1, 0, 0, 0]*X[0, 1, 0] + UW3[2, 2, 0, 0, 0]*X[0, 2, 0]
'''

''' O2_dx_10
    UW2[0, 1, 0, 0] + UW3[0, 0, 1, 0, 0]*X[0, 0, 0] + UW3[0, 1, 0, 0, 0]*X[0, 0, 0] + 2*UW3[0, 1, 1, 0, 0]*X[0, 1, 0] + UW3[0, 1, 2, 0, 0]*X[0, 2, 0] + UW3[0, 2, 1, 0, 0]*X[0, 2, 0], 
    UW2[1, 1, 0, 0] + UW3[1, 0, 1, 0, 0]*X[0, 0, 0] + UW3[1, 1, 0, 0, 0]*X[0, 0, 0] + 2*UW3[1, 1, 1, 0, 0]*X[0, 1, 0] + UW3[1, 1, 2, 0, 0]*X[0, 2, 0] + UW3[1, 2, 1, 0, 0]*X[0, 2, 0], 
    UW2[2, 1, 0, 0] + UW3[2, 0, 1, 0, 0]*X[0, 0, 0] + UW3[2, 1, 0, 0, 0]*X[0, 0, 0] + 2*UW3[2, 1, 1, 0, 0]*X[0, 1, 0] + UW3[2, 1, 2, 0, 0]*X[0, 2, 0] + UW3[2, 2, 1, 0, 0]*X[0, 2, 0]
'''

''' O2_dx_20
   UW2[0, 2, 0, 0] + UW3[0, 0, 2, 0, 0]*X[0, 0, 0] + UW3[0, 1, 2, 0, 0]*X[0, 1, 0] + UW3[0, 2, 0, 0, 0]*X[0, 0, 0] + UW3[0, 2, 1, 0, 0]*X[0, 1, 0] + 2*UW3[0, 2, 2, 0, 0]*X[0, 2, 0], 
   UW2[1, 2, 0, 0] + UW3[1, 0, 2, 0, 0]*X[0, 0, 0] + UW3[1, 1, 2, 0, 0]*X[0, 1, 0] + UW3[1, 2, 0, 0, 0]*X[0, 0, 0] + UW3[1, 2, 1, 0, 0]*X[0, 1, 0] + 2*UW3[1, 2, 2, 0, 0]*X[0, 2, 0], 
   UW2[2, 2, 0, 0] + UW3[2, 0, 2, 0, 0]*X[0, 0, 0] + UW3[2, 1, 2, 0, 0]*X[0, 1, 0] + UW3[2, 2, 0, 0, 0]*X[0, 0, 0] + UW3[2, 2, 1, 0, 0]*X[0, 1, 0] + 2*UW3[2, 2, 2, 0, 0]*X[0, 2, 0]
'''

