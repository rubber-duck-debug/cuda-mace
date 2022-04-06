from typing import Dict

import torch
from tensor_contraction.cuda import  tensor_contraction

# Define some parameters for the contraction
l_max = 3
num_irreps = ((l_max + 1) ** 2) * 3
num_features = 96
num_atoms = 120

print ("num_irreps: ", num_irreps)
#  nirreps, nirreps, nirreps, 96
U_tensors = torch.load("U_tensors_n_s_fp32.pt")

for key in U_tensors.keys():
    U_tensors[key] = U_tensors[key].cuda().float()

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

sparseUIndexes = torch.zeros(num_irreps, num_irreps, num_irreps, 16, device='cuda', dtype=torch.int).fill_(-1)
sparseU_nvals = torch.zeros(num_irreps, num_irreps, num_irreps, device='cuda', dtype=torch.int)
sparseUValues = torch.zeros(num_irreps, num_irreps, num_irreps, 16, device='cuda')

sparseUw_indexes = torch.zeros(num_irreps, num_irreps, 6, device='cuda', dtype=torch.int).fill_(-1)
sparseUw_nvals = torch.zeros(num_irreps, num_irreps, device='cuda', dtype=torch.int)

sparseU2w_indexes = torch.zeros(num_irreps, num_irreps, device='cuda', dtype=torch.int).fill_(-1)
sparseU2w_nvals = torch.zeros(num_irreps, device='cuda', dtype=torch.int)

for i in range(num_irreps):
    
    idx2 = []
    
    for j in range(num_irreps):
        count = 0
        idxs = []
        for k in range(num_irreps):
            (indexes,) = torch.where(U_tensors[3][i, j, k,:] != 0)

            if (indexes.shape[0] > 0):
                # print (i, j, k, indexes)
                sparseUIndexes[i, j, k,:indexes.shape[0]] = indexes
                sparseU_nvals[i, j, k] = indexes.shape[0]
                sparseUValues[i, j, k,:indexes.shape[0]] = U_tensors[3][i, j, k, indexes]
                count += 1
                idxs.append(k)
        
        sparseUw_indexes[i, j,:count] = torch.tensor([idxs]).int().cuda()
        sparseUw_nvals[i, j] = count
        
        if (count > 0): 
            idx2.append(j)
    
    idx2 = torch.tensor(idx2).int().cuda()
    
    if (idx2.shape[0] > 0):
        sparseU2w_indexes[i, 0:idx2.shape[0]] = idx2
        sparseU2w_nvals[i] = idx2.shape[0]
        
t1 = torch.zeros(int(sparseU2w_nvals.shape[0] / 2), 2, device='cuda', dtype=torch.int)
t2 = torch.zeros(int(sparseU2w_nvals.shape[0] / 2), 2, device='cuda', dtype=torch.int)

for i in range (int(sparseU2w_nvals.shape[0] / 2)):
    t1[i][0] = i
    t1[i][1] = sparseU2w_nvals.shape[0] - (i + 1)
    t2[i][0] = sparseU2w_nvals[i]
    t2[i][1] = sparseU2w_nvals[-(i + 1)]
    
    # print (sparseU2w_nvals[i] + sparseU2w_nvals[-(i + 1)])
print (t1)
print (t2)
       
print (sparseU2w_indexes)
print (sparseU2w_nvals)

total_nonsparse_elements = sparseU2w_nvals.sum()
nwork_per_block = torch.ceil((total_nonsparse_elements / 16)).int().item()  # 75

block_work_ids_i = torch.zeros(16, nwork_per_block, device='cuda', dtype=torch.int).fill_(-1)
block_work_ids_j = torch.zeros(16, nwork_per_block, device='cuda', dtype=torch.int).fill_(-1)
block_nwork = torch.zeros(16, device='cuda', dtype=torch.int).fill_(-1)

nelem = 0
count_i = 0  # indexes the i in num_irreps (:, j)
count_j = 0  # indexes the j in num_irreps (i, :)

for i in range(0, 16):
    nwork = 0
    
    for j in range(nwork_per_block):
        block_work_ids_i[i, j] = count_i
        block_work_ids_j[i, j] = sparseU2w_indexes[count_i, count_j]
        
        # print (count_i, count_j, block_work_ids_j[i, j].item(), sparseU2w_nvals[count_i].item())
        
        count_j += 1
        nelem += 1
        nwork += 1
        
        if (count_j == sparseU2w_nvals[count_i]):
            count_i += 1
            count_j = 0
            
        if (nelem == total_nonsparse_elements.item() - 1):
            break
            
    block_nwork [i] = nwork
    nwork = 0

print (nwork_per_block)
print (block_nwork)
print (block_work_ids_i)
print (block_work_ids_j)

weights = {3: torch.randn((1270, num_features), device='cuda', requires_grad=True),
           2:torch.randn((24, num_features), device='cuda', requires_grad=True),
           1: torch.randn((3, num_features), device='cuda', requires_grad=True) }

node_feats = torch.randn(num_atoms , num_irreps, num_features, device='cuda')

torch.matmul(torch.rand(1024, 1024, device='cuda'), torch.rand(1024, 1024, device='cuda'))

print (torch.count_nonzero(U_tensors[3]) / U_tensors[3].numel() * 100.0)
start.record()
Uw = torch.einsum('...ik,kc ->...ic', U_tensors[3], weights[3])
end.record()
torch.cuda.synchronize()
print("Uw einsum", start.elapsed_time(end), "ms") 

start.record()
Uw_cuda = tensor_contraction.get_Uw3_sparse_contraction(sparseUValues, sparseUIndexes, sparseU_nvals, sparseUw_indexes, sparseUw_nvals, weights[3])
end.record()
torch.cuda.synchronize()

print("Uw sparse", start.elapsed_time(end), "ms") 
print ("einsum - sparse error:", torch.linalg.norm(Uw_cuda - Uw))

start.record()
UwN = torch.einsum('...ic,bic ->bc...', Uw_cuda, node_feats)
end.record()
torch.cuda.synchronize()

print("UwN einsum", start.elapsed_time(end), "ms, UwN: ", UwN.shape) 

start.record()
UwN_cuda = tensor_contraction.get_UwN3_sparse(Uw_cuda, sparseUw_indexes, sparseUw_nvals, node_feats, 16, 16, 16, 96, 2)
end.record()
torch.cuda.synchronize()

print("UwN sparse", start.elapsed_time(end), "ms, UwN:", UwN_cuda.shape) 

print ("einsum - sparse error:", torch.linalg.norm(UwN_cuda.transpose(-1, -3).transpose(-1, -2).contiguous() - UwN))

start.record() 
c_tensor = torch.einsum('...k,kc->c...',
                                        U_tensors[2],
                                        weights[2])

''' could potentially combine this into a single step'''
end.record()
torch.cuda.synchronize()
print("Uw2 einsum part 1", start.elapsed_time(end), "ms") 
print (torch.count_nonzero(c_tensor), c_tensor.numel())

start.record() 
c_tensor = c_tensor + UwN
end.record()
torch.cuda.synchronize()
print("Uw2 tensor addition", start.elapsed_time(end), "ms") 
''' ------------------------------------------------ end'''

print ("c_tensor shape: ", c_tensor.shape, "node_feats shape:", node_feats.shape)

# torch.Size([120, 96, 48, 48]) torch.Size([120, 48, 96])

start.record() 
out = torch.einsum('bc...i,bci->bc...', c_tensor, node_feats.transpose(-1, -2).contiguous())
end.record()
torch.cuda.synchronize()

print (out.shape)

print("U2W2N einsum", start.elapsed_time(end), "ms, output: ", out.shape) 

start.record() 
out_cuda = tensor_contraction.get_UwN2_dense_contraction(c_tensor, node_feats.transpose(-1, -2).contiguous(), 32, 32, 4,
       8)
end.record()
torch.cuda.synchronize()

print (out)
print (out_cuda)
print("U2W2N cuda", start.elapsed_time(end), "ms, output: ", out_cuda.shape) 
print ("tensorcore - einsum norm diff: ", torch.linalg.norm(out - out_cuda))
