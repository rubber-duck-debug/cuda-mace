from typing import Dict

import torch
from tensor_contraction.cuda import sparse_stuff

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

# Define some parameters for the contraction
l_max = 3
num_irreps = ((l_max + 1) ** 2) * 3

print ("num_irreps: ", num_irreps)

num_atoms = 120
num_features = 96

node_feats = torch.randn(num_atoms , num_features, num_irreps, device='cuda')

U_tensor_full = torch.randn(num_irreps, num_irreps, num_irreps, 1200, device='cuda')
weight_full = torch.randn(1200, num_features, device='cuda')

U_tensor = torch.randn(num_irreps, num_irreps, num_irreps, 12, device='cuda')
weights = torch.randn(12, num_features, device='cuda')

indexes = torch.arange(1028, 1040, device='cuda').repeat(48, 48, 48, 1).int()
nvals = torch.zeros((num_irreps, num_irreps, num_irreps), device='cuda').fill_(12).int()

# warmup
torch.matmul(torch.rand(1024, 1024, device='cuda'), torch.rand(1024, 1024, device='cuda'))

start.record()
test1 = torch.einsum('...ik,kc ->...ic',
                               U_tensor_full,
                               weight_full)

end.record()
torch.cuda.synchronize()
print("einsum 1 fp32", start.elapsed_time(end), "ms") 

print (test1.shape)
print (node_feats.shape)

start.record()
test1 = torch.einsum('...ic, bci-> bc...',
                               test1,
                               node_feats)

end.record()
torch.cuda.synchronize()
print("einsum 2 fp32", start.elapsed_time(end), "ms") 

print (test1.shape)

start.record()
test1 = torch.einsum('...ik,kc,bci -> bc...',
                               U_tensor_full, weight_full,
                               node_feats)

end.record()
torch.cuda.synchronize()
print("einsum full fp32", start.elapsed_time(end), "ms") 

start.record()
test2 = sparse_stuff.get_sparse_uw_contraction_fp32(nvals, indexes, U_tensor, weights)
end.record()
torch.cuda.synchronize()

print("sparse fp32", start.elapsed_time(end), "ms") 

# print (test2)
