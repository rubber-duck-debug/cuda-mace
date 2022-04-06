from typing import Dict
from opt_einsum import contract
import torch


class Convolution_vect(torch.nn.Module):

    def __init__(self,
                 U_tensors: Dict[int, torch.tensor],
                 num_features: torch.tensor,
                 correlation: int,):
        super().__init__() 
        self.U_tensors = U_tensors  # [(lmax+1)**2]**correlation + [num_weights]
        self.num_features = num_features
        self.correlation = correlation
        self.equation_main = '...ik,kc,bci -> bc...'
        self.equation_weighting = '...k,kc->c...'
        self.equation_contract = 'bc...i,bci->bc...'

    def forward(self, node_feats: torch.tensor, weights: Dict[int, torch.tensor]):
        
        total_start = torch.cuda.Event(enable_timing=True)
        total_end = torch.cuda.Event(enable_timing=True)
        
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        total_start.record()
        start.record()
        
        out = torch.einsum(self.equation_main,
                               self.U_tensors[self.correlation],
                               weights[self.correlation],
                               node_feats)  # Conctract the U tensor, the weights and the representations
        end.record()
        torch.cuda.synchronize()
        
        print("correlation order: ", 3, ", UwN contraction time:", start.elapsed_time(end), "ms")
        
        sub_start = torch.cuda.Event(enable_timing=True)
        sub_end = torch.cuda.Event(enable_timing=True)
        
        for corr in range(self.correlation - 1, 0, -1):  # Loop over the correlation order and contract
                start.record()
                
                sub_start.record()
                c_tensor = torch.einsum(self.equation_weighting,
                                        U_tensors[corr],
                                        weights[corr])
                
                sub_end.record()
                torch.cuda.synchronize()
                
                print("correlation order: ", corr, "c_tensor time:", sub_start.elapsed_time(sub_end), "ms")
                
                sub_start.record()
                c_tensor = c_tensor + out
                sub_end.record()
                torch.cuda.synchronize()
                
                if (corr == 2):
                    pass
#                     #  'bc...i,bci->bc...'
#                     # torch.Size([128, 96, 48, 48])
#                     # torch.Size([128, 96, 48])
#                     
#                     U2_indices = torch.zeros(128, 96, 48, 48, device='cuda', dtype=torch.int)
#                     U2_nvals = torch.zeros(128, 96, 48, device='cuda', dtype=torch.int)
#                     
# #                     print(c_tensor.shape)
#                     for i in range(c_tensor.shape[0]):  # 128
#                         for j in range(c_tensor.shape[1]):  # nirreps
#                             for k in range(c_tensor.shape[2]):  # nirreps
#                                 (indexes,) = torch.where(c_tensor[i, j, k,:] != 0)
#                                 
#                                 if (indexes.shape[0] > 0):
#                                     U2_indices[i, j, k,:indexes.shape[0]] = indexes
#                                     U2_nvals[i, j, k] = indexes.shape[0]
#                                 # if (indexes.shape[0] > 0):
#                                 #    print (i, j, k, indexes.shape, indexes)
#                 
#                     print (U2_indices)
#                     print (sparseU2w_indexes)
#                     print (U2_nvals)
#                     print (sparseU2w_nvals)
                                    
                print("correlation order: ", corr, "c_tensor addition time:", sub_start.elapsed_time(sub_end), "ms")
                
                # print("c_tensor sparsity:", (torch.count_nonzero(c_tensor) / c_tensor.numel()).item() * 100.0, "%")
                
                sub_start.record()
                out = torch.einsum(self.equation_contract, c_tensor, node_feats)
                sub_end.record()
                torch.cuda.synchronize()

                print("correlation order: ", corr, "output time:", sub_start.elapsed_time(sub_end), "ms")
                
                end.record()
                torch.cuda.synchronize()
        
        total_end.record()
        torch.cuda.synchronize()
        print("total time: ", total_start.elapsed_time(total_end), "ms")
        
        return out


# warmup
torch.matmul(torch.rand(1024, 1024, device='cuda'), torch.rand(1024, 1024, device='cuda'))

# Define some parameters for the contraction
l_max = 3
num_irreps = ((l_max + 1) ** 2) * 3

num_features = 96
correlation = 3  # Correlation order
num_atoms = 128  # number of atoms

print ("natoms: ", num_atoms)
print ("num_irreps: ", num_irreps)
print ("correlation order: ", correlation)

U_tensors = {2: torch.randn(num_irreps, num_irreps, num_irreps, 120, device='cuda'),
            1: torch.randn(num_irreps, num_irreps, 8, device='cuda'),
            0: torch.randn(num_irreps, 4, device='cuda')}

U_tensors = torch.load("U_tensors_n_s_fp32.pt")

for key in U_tensors.keys():
    U_tensors[key] = U_tensors[key].cuda().float()
    
    print ("----", key, U_tensors[key].shape, "----")

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
stuff = torch.where(U_tensors[3] != 0)
end.record()
torch.cuda.synchronize()

Conv_vect = Convolution_vect(U_tensors, num_features, correlation)

node_feats = torch.randn(num_atoms , num_features, num_irreps, device='cuda')
weights = {3: torch.randn((1270, num_features), device='cuda', requires_grad=True),
           2:torch.randn((24, num_features), device='cuda', requires_grad=True),
           1: torch.randn((3, num_features), device='cuda', requires_grad=True) }

# sparseUIndexes = torch.zeros(num_irreps, num_irreps, num_irreps, 16, device='cuda', dtype=torch.int).fill_(-1)
# sparseU_nvals = torch.zeros(num_irreps, num_irreps, num_irreps, device='cuda', dtype=torch.int)
# sparseUValues = torch.zeros(num_irreps, num_irreps, num_irreps, 16, device='cuda')
# 
# sparseUw_indexes = torch.zeros(num_irreps, num_irreps, 6, device='cuda', dtype=torch.int).fill_(-1)
# sparseUw_nvals = torch.zeros(num_irreps, num_irreps, device='cuda', dtype=torch.int)
# 
# sparseU2w_indexes = torch.zeros(num_irreps, num_irreps, device='cuda', dtype=torch.int).fill_(-1)
# sparseU2w_nvals = torch.zeros(num_irreps, device='cuda', dtype=torch.int)
# 
# for i in range(num_irreps):
#     
#     idx2 = []
#     
#     for j in range(num_irreps):
#         count = 0
#         idxs = []
#         for k in range(num_irreps):
#             (indexes,) = torch.where(U_tensors[3][i, j, k,:] != 0)
# 
#             if (indexes.shape[0] > 0):
#                 # print (i, j, k, indexes)
#                 sparseUIndexes[i, j, k,:indexes.shape[0]] = indexes
#                 sparseU_nvals[i, j, k] = indexes.shape[0]
#                 sparseUValues[i, j, k,:indexes.shape[0]] = U_tensors[3][i, j, k, indexes]
#                 count += 1
#                 idxs.append(k)
#         
#         sparseUw_indexes[i, j,:count] = torch.tensor([idxs]).int().cuda()
#         sparseUw_nvals[i, j] = count
#         
#         if (count > 0): 
#             idx2.append(j)
#     
#     idx2 = torch.tensor(idx2).int().cuda()
#     
#     if (idx2.shape[0] > 0):
#         sparseU2w_indexes[i, 0:idx2.shape[0]] = idx2
#         sparseU2w_nvals[i] = idx2.shape[0]
# 
# print (sparseU2w_nvals.sum())  # vs 48x48
# 
# n_irreps_per_block = sparseU2w_nvals.sum() / 16
# 
# print (n_irreps_per_block)
'''
for a in range(blockIdx.x, natoms, gridDim.x):
    for i in range(blockIdx.y, nirreps, gridDim.y):
    
        for j in range(0, n_irreps_per_block): 
            irrep_j = irrepj_idxs[i][j]
            
            dot = 0.0
            for l in range (threadIdx.x, nfeatures, blockDim.x):
            
                for k in range (threadIdx.y, nirreps, blockDim.y):
                    dot += c_tensor[a][i][irrep_j][l] * node_features[a][i][k]
                    
                output[a][i][j][l] = dot

'''
x_vect = Conv_vect(node_feats, weights)
