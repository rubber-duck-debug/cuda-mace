import torch
from tensor_contraction.cuda import  tensor_contraction

# Define some parameters for the contraction
l_max = 3
num_irreps = ((l_max + 1) ** 2) * 3

print ("num_irreps: ", num_irreps)

U_tensors = torch.load("U_tensors_n_s_fp32.pt")

for key in U_tensors.keys():
    U_tensors[key] = U_tensors[key].cuda().float()
    
    print ("----", key, U_tensors[key].shape, "----")

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

num_features = 96e
correlation = 3  # Correlation order
num_atoms = 128  # number of atoms

node_feats = torch.randn(num_atoms , num_irreps, num_features, device='cuda')

weights = {3: torch.randn((1270, num_features), device='cuda', requires_grad=True),
           2:torch.randn((24, num_features), device='cuda', requires_grad=True),
           1: torch.randn((3, num_features), device='cuda', requires_grad=True) }

# warmup
torch.matmul(torch.rand(1024, 1024, device='cuda'), torch.rand(1024, 1024, device='cuda'))

start.record()
Uw = torch.einsum('...ik,kc ->...ic', U_tensors[3], weights[3])
end.record()
torch.cuda.synchronize()
print("Uw einsum", start.elapsed_time(end), "ms") 

start.record()
UwN = torch.einsum('...ik,kc,bic ->bc...', U_tensors[3], weights[3], node_feats)
end.record()
torch.cuda.synchronize()
print("full einsum", start.elapsed_time(end), "ms") 

start.record()
UwN = torch.einsum('...ic,bic ->bc...', Uw, node_feats)
end.record()
torch.cuda.synchronize()

print("UwN einsum", start.elapsed_time(end), "ms") 

print (UwN.shape)
print ("UwN Shape:", Uw.shape)

avg_time = 0.0

avg_nsparse = 1

for i in range (20):
    
    nvals = torch.tensor([avg_nsparse]).repeat(48, 48).int().cuda()
    U_indexes = torch.randint(high=48, size=(avg_nsparse,)).repeat(48, 48, 1).int().cuda()
    U_indexes = U_indexes.transpose(-1, -2).contiguous()

    start.record()
    test = tensor_contraction.get_UwN3_sparse(Uw, U_indexes, nvals, node_feats, 16, 32, 16, 32)
    end.record()
    torch.cuda.synchronize()
    avg_time += start.elapsed_time(end)

print("sparse_time", avg_time / 20 , "ms") 

