import numpy as np
from time import time
from opt_einsum import contract
import torch

from numba import cuda



        
X = np.fromfile('X.npy').reshape(21, 128, 16)


Y = np.fromfile('Y.npy').reshape(21, 3)


U_3 = np.fromfile('U_3.npy').reshape(16,16,16,23)
U_2 = np.fromfile('U_2.npy').reshape(16,16, 4)
U_1 = np.fromfile('U_1.npy').reshape(16,1)

U_3 = torch.from_numpy(U_3).cuda()
U_2 = torch.from_numpy(U_2).cuda()
U_1 = torch.from_numpy(U_1).cuda()

W_3 = np.fromfile('W_3.npy').reshape(3,23,128)
W_2 = np.fromfile('W_2.npy').reshape(3,4,128)
W_1 = np.fromfile('W_1.npy').reshape(3,1,128)

W_3 = torch.from_numpy(W_3).cuda()
W_2 = torch.from_numpy(W_2).cuda()
W_1 = torch.from_numpy(W_1).cuda()


equation_main = "...ik,ekc,bci,be -> bc..."
equation_weighting = "...k,ekc,be->bc..."
equation_contract = "bc...i,bci->bc..."
            
correlation = 3





# only at most 1 value on the 23 dimension

# for i in range(U_3.shape[0]):
#     for j in range(U_3.shape[1]): 
#         for k in range(U_3.shape[2]): 
#             print (i, j, k, torch.count_nonzero(U_3[i, j, k]))

indices = torch.nonzero(U_3)
indices = indices.cpu().numpy()

print (indices)

#out = contract(equation_main, U_3,W_3, X, Y)

#     torch.Size([16, 16, 16, 23]) 94208 tensor(353, device='cuda:0')
# torch.Size([3, 23, 128]) torch.Size([21, 128, 16]) torch.Size([21, 3])
# tensor([[ 0,  0,  0,  0],
#         [ 0,  1,  1,  4],
#         [ 0,  2,  2,  4],
#         ...,
#         [15, 14,  7, 16],
#         [15, 15,  0,  3],
#         [15, 15,  6, 16]], device='cuda:0')
# ---> torch.Size([21, 128, 16, 16])

# equation_main = "...ik,ekc,bci,be -> bc..."

# torch.Size([16, 16, 16, 23]), torch.Size([3, 23, 128]), torch.Size([21, 128, 16]), torch.Size([21, 3])

# "...ik,ekc -> ...iec
#                       i   k                e   k   c                            i  e   c
#  torch.Size([16, 16, 16, 23]), torch.Size([3, 23, 128]) -> torch.Size([16, 16, 16, 3, 128])

        
        
# "...iec, bci -> bce..."
#                     i   e    c                b    c    i                  b    c   e
# torch.Size([16, 16, 16, 3, 128]), torch.Size([21, 128, 16]) -> torch.Size([21, 128, 3, 16, 16])

# 'bce, be -> bc..'

# torch.Size([21, 128, 3, 16, 16]), torch.Size([21, 3]) -> torch.Size([21, 128, 16, 16])

out_1 = np.zeros((16,16,16,3,128))

out_1_sparse = np.zeros((16,16,16,3,128))

U_3_npy = U_3.cpu().numpy()
W_3_npy = W_3.cpu().numpy()


print (Y)

out_3 = np.zeros((21,128, 16, 16))

# natoms, j, k


        
for i in range(5):
    start = time()
    
    for a in range (indices.shape[0]):
        
        i, j, k, l = indices[a]
        
 
        out_3[:, :,  j,k] = U_3_npy[i,j,k,l] * W_3_npy[0, l, :] * X[..., i]
            
            #print (a.shape)
            #out_1_sparse[i,j,k, e, :] = U_3_npy[i,j,k,l] * W_3_npy[e, l, :] 
        
    end = time()

print (end - start)


out_2 = np.zeros((21,128,3, 16, 16))
out_3 = np.zeros((21,128, 16, 16))

start = time()
for e in range(W_3.shape[0]):
    for c in range(W_3.shape[2]):
           out_1[..., e, c] = np.dot(U_3_npy[..., :], W_3_npy[e, :, c])
end = time()

print (end - start)


for e in range(W_3.shape[0]):       
    for b in range(X.shape[0]):
        for c in range(X.shape[1]):
            out_2[b, c, e, ...] =  np.dot(out_1[..., e, c], X[b, c, :])



for e in range(W_3.shape[0]):       
    for b in range(X.shape[0]):
            out_3[b, ...] +=  (out_2[b, :, e, :] *  Y[b,e])


U_tensors = {3: U_3, 2:  U_2, 1: U_1.cuda()}
W_tensors = {3: W_3, 2: W_2, 1: W_1.cuda()}
X = torch.from_numpy(X).cuda()
Y = torch.from_numpy(Y).cuda()

for i in range(5):
    start = time()
    out = contract(equation_main, U_3,W_3, X, Y)
    end = time()

print ("CONTRACITON: ", end - start)

print (U_tensors[3].shape, U_tensors[3].numel(), torch.count_nonzero(U_tensors[3]))
print (W_3.shape, X.shape, Y.shape)
print ("--->", out.shape)
    
    #print (U_3.numel(), torch.count_nonzero(U_3))
    
    #print (torch.nonzero(U_tensors[3]))
    
    #print (out.numel(), torch.count_nonzero(out))
    #print (out.shape)
    
for corr in range(correlation - 1, 0, -1):      

    
    print (U_tensors[corr].shape, W_tensors[corr].shape, Y.shape, U_tensors[corr].numel(), torch.count_nonzero(U_tensors[corr]))
    
    print (torch.nonzero(U_tensors[corr]))
    # equation_weighting = "...k,ekc,be->bc..."
    
    c_tensor = contract(
        equation_weighting,
        U_tensors[corr],
        W_tensors[corr],
        Y,
    )
    
    print (c_tensor.shape)
    c_tensor = c_tensor + out
    
    out = contract(equation_contract, c_tensor, X)

end = time()

print (end - start)

#
#
#
# def forward(self, x: torch.Tensor, y: Optional[torch.Tensor]):
#         from time import time
#
#
#
#         with profiler.record_function("CONTRACTION"):
#
#             if self.element_dependent:
#
#                 out = contract(
#                     self.equation_main,
#                     self.U_tensors(self.correlation),
#                     self.weights[str(self.correlation)],
#                     x,
#                     y,
#                 )  # TODO: use optimize library and cuTENSOR  # pylint: disable=fixme
#
#
#                 print (f"correlation: {self.correlation} U:", self.U_tensors(self.correlation).shape)
#                 print (f"correlation: {self.correlation} W:", self.weights[str(self.correlation)].shape)
#                 print ("X:", x.shape)
#                 print ("Y:", y.shape)
#                 print ("out: ", out.shape)
#
#
#                 self.U_tensors(self.correlation).cpu().numpy().tofile(f'U_{self.correlation}.npy')
#                 self.weights[str(self.correlation)].cpu().detach().numpy().tofile(f'W_{self.correlation}.npy')
#                 x.cpu().detach().numpy().tofile(f'X.npy')
#                 y.cpu().detach().numpy().tofile(f'Y.npy')
#
#                 #print ()
#
#                 #start = time()
#
#                 for corr in range(self.correlation - 1, 0, -1):
#
#                     print (f"correlation: {corr} U:", self.U_tensors(corr).shape)
#                     print (f"correlation: {corr} W:", self.weights[str(corr)].shape)
#
#                     self.U_tensors(corr).cpu().numpy().tofile(f'U_{corr}.npy')
#                     self.weights[str(corr)].cpu().detach().numpy().tofile(f'W_{corr}.npy')
#
#                     c_tensor = contract(
#                         self.equation_weighting,
#                         self.U_tensors(corr),
#                         self.weights[str(corr)],
#                         y,
#                     )
#                     c_tensor = c_tensor + out
#                     out = contract(self.equation_contract, c_tensor, x)
#
#                 #print (out.shape, out)
#                 #end = time()
#
#                 #print (end-start)
#
#             else:
#
#                 out = contract(
#                     self.equation_main,
#                     self.U_tensors(self.correlation),
#                     self.weights[str(self.correlation)],
#                     x,
#                 )  # TODO: use optimize library and cuTENSOR  # pylint: disable=fixme
#                 for corr in range(self.correlation - 1, 0, -1):
#                     c_tensor = contract(
#                         self.equation_weighting,
#                         self.U_tensors(corr),
#                         self.weights[str(corr)],
#                     )
#                     c_tensor = c_tensor + out
#                     out = contract(self.equation_contract, c_tensor, x)
#             resize_shape = torch.prod(torch.tensor(out.shape[1:]))
#             return out.view(out.shape[0], resize_shape)
#
#     def U_tensors(self, nu):
#         return self._buffers[f"U_matrix_{nu}"]