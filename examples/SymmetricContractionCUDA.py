import numpy as np
from time import time
from opt_einsum import contract
import torch
import logging
import traceback
from mace_ops.cuda import SymmetricContraction

nchannels = 128

X = np.fromfile('symm_contraction_data/X.npy').reshape(21, 128, 16)[:, :nchannels, :]

Y = np.fromfile('symm_contraction_data/Y.npy').reshape(21, 3)

U3 = np.fromfile('symm_contraction_data/U_3.npy').reshape(16,16,16,23)
U2 = np.fromfile('symm_contraction_data/U_2.npy').reshape(16,16, 4)
U1 = np.fromfile('symm_contraction_data/U_1.npy').reshape(16, 1)

U3 = torch.from_numpy(U3).float().cuda()
U2 = torch.from_numpy(U2).float().cuda()
U1 = torch.from_numpy(U1).float().cuda()

W3 = np.fromfile('symm_contraction_data/W_3.npy').reshape(3,23,128)[:, :, :nchannels]

W2 = np.fromfile('symm_contraction_data/W_2.npy').reshape(3,4, 128)[:, :, :nchannels]

W1 = np.fromfile('symm_contraction_data/W_1.npy').reshape(3,1, 128)[:, :, :nchannels]


W3 = torch.from_numpy(W3).float().cuda()

W2 = torch.from_numpy(W2).float().cuda()

W1 = torch.from_numpy(W1).float().cuda()
        
correlation = 3

U_tensors = {3: U3, 2:  U2, 1: U1}
W_tensors = {3: W3, 2: W2, 1: W1}

nrepeats = int((500.0) / 21)

X = torch.from_numpy(X).float().cuda().repeat(nrepeats, 1, 1)
Y = torch.from_numpy(Y).float().cuda().repeat(nrepeats, 1)

X_torch = X.transpose(-1, -2).contiguous()
atom_types = np.array([1,1,1,1,1,1,1,2,2,2,1,1,2,0,0,0,0,0,0,0,0])
atom_types_torch = torch.from_numpy(atom_types).repeat(nrepeats).int().cuda()


equation_main = "...ik,ekc,bci,be -> bc..."
equation_weighting = "...k,ekc,be->bc..."
equation_contract = "bc...i,bci->bc..."

equation_contract_weights = '...ik, ekc -> ...iec'

symm_contract = SymmetricContraction(U_tensors, W_tensors, device='cuda', dtype=torch.float32)

def mace_v1(U_tensors, W_tensors, X, Y, correlation, correlation_min=0, requires_grad = True):

    outputs = {}

    out_v1 = contract(equation_main, U_tensors[correlation],W_tensors[correlation], X, Y)

    outputs[correlation] = out_v1

    for corr in range(correlation - 1, correlation_min, -1):      

        c_tensor_v1 = contract(
            equation_weighting,
            U_tensors[corr],
            W_tensors[corr],
            Y,
        )

        c_tensor_v1 = c_tensor_v1 + out_v1
        
        #equation_contract = "bc...i,bci->bc..."
        out_v1 = contract(equation_contract, c_tensor_v1, X)

        outputs[corr] = out_v1

    if (requires_grad):
        out_v1.sum().backward()

    if (requires_grad):
        return outputs, X.grad.clone()

X_torch.requires_grad = True
out_cuda = symm_contract.forward(X_torch, atom_types_torch)

X.requires_grad=True
outputs_v1, grad_v1 = mace_v1(U_tensors, W_tensors, X, Y, 3, 0, requires_grad = True)

atom_check = 0

print (out_cuda[atom_check])
print (outputs_v1[1][atom_check])

out_cuda.sum().backward()

print (grad_v1[atom_check])
print (X_torch.grad[atom_check].transpose(-1, -2))


torch.matmul(torch.zeros(1024, 1024, device='cuda'), torch.zeros(1024, 1024, device='cuda'))


start = time()
for i in range (1000):
    outputs_v1, grad_v1 = mace_v1(U_tensors, W_tensors, X, Y, 3, 0, requires_grad = True)
    torch.cuda.synchronize()

end = time()

print ("Dense forward + backward: ", end - start)

start = time()
for i in range (1000):
    _ = symm_contract.forward(X_torch, atom_types_torch)
    
    os = _.sum()

    os.backward()
    torch.cuda.synchronize()

end = time()
print ("Sparse forward + backward:", end - start)

# U1 = U1.reshape(16)

# start = time()
# for i in range (1000):
#     out = sparse_full_symmetric_contraction(U3_nonsparse_indices,U3_num_nonsparse, U3_nonzero_elements, U2_non_sparse_indices, U2_nonsparse_elements, U1, W3, W2, W1, X_torch, atom_types_torch, 64, 16, 1)
# end = time()
# print (end - start)

# start = time()
# for i in range (1000):
#     out, grad = sparse_full_symmetric_contraction_derivative(U3_nonsparse_indices,U3_num_nonsparse, U3_nonzero_elements, U2_non_sparse_indices, U2_nonsparse_elements, U1, W3, W2, W1, X_torch, atom_types_torch, 64, 16, 1)
# end = time()
# print (end - start)

# atom_check = 0

# assert torch.allclose(grad_v1[atom_check], grad[atom_check].transpose(-1, -2), atol=1e-7)
