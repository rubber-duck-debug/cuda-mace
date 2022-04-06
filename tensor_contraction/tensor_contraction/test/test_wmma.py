from typing import Dict

import torch
from tensor_contraction.cuda import  tests

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

U_tensors_dense = torch.load("U_tensors_n_s_fp32.pt")

for key in U_tensors_dense.keys():
    U_tensors_dense[key] = U_tensors_dense[key].cuda().float()

U_indices = torch.load("StructU_indices.pt")
U_nvals = torch.load("StructU_nvals.pt")
U_vals = torch.load("StructU_vals.pt")

U_tensor = torch.zeros(48, 48, 48, 16, device='cuda')
U_tensor[:,:,:,:12] = U_vals  # torch.randn(48, 48, 48, 16, device='cuda')
weights_cuda = torch.randn(16, 96, device='cuda')
weights_torch = torch.randn(1270, 96, device='cuda')

# 48, 48, 96, 48
node_feats = torch.randn(120 , 96, 48, device='cuda')

# warmup
start.record()
torch.matmul(torch.rand(1024, 1024, device='cuda'), torch.rand(1024, 1024, device='cuda'))
end.record()
torch.cuda.synchronize()

print (U_tensor.shape)
print (weights_cuda.shape)

start.record()
output_torch_uw = torch.einsum('...ik, kc -> ...ic', U_tensors_dense[3], weights_torch)
end.record()
torch.cuda.synchronize()
print (output_torch_uw.shape)
print("torch einsum", start.elapsed_time(end), "ms") 

start.record()
output_torch = torch.einsum('...ik, kc, bci -> bc...', U_tensors_dense[3], weights_torch, node_feats)
end.record()
torch.cuda.synchronize()

print (output_torch.shape)
print("torch einsum", start.elapsed_time(end), "ms") 

start.record()
output_cuda = tests.get_wmma_dense(U_tensor, weights_cuda)
end.record()
torch.cuda.synchronize()
print("cuda wmma dense", start.elapsed_time(end), "ms") 

start.record()
uwn_cuda = tests.get_wmma_UwN_dense(output_torch_uw.transpose(-1, -2).contiguous(), node_feats)
end.record()
torch.cuda.synchronize()
print("cuda wmma uwn", start.elapsed_time(end), "ms") 

Uw_indices = torch.zeros(48, 48, 48, device='cuda', dtype=int)

U_indices_nonzero = []
for i in range(U_nvals.shape[0]):
    for j in range(U_nvals.shape[1]):
        for k in range(U_nvals.shape[2]):
            nonzero_idx, = torch.where(U_vals[i, j, k] != 0)
            
            if (len(nonzero_idx) > 0):
                U_indices_nonzero.append([i, j, k])
                Uw_indices[i, j, k] = 1

U_indices_nonzero = torch.tensor(U_indices_nonzero)

print (output_torch_uw.shape)
print (node_feats.shape)
print (U_indices_nonzero.shape)

start.record()
uwn_cuda = tests.get_UwN_sparse(output_torch_uw, U_indices_nonzero.cuda().int(), node_feats.transpose(-1, -2).contiguous())
end.record()
torch.cuda.synchronize()
print("cuda uwn sparse", start.elapsed_time(end), "ms") 

