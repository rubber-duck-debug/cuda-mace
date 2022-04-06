import torch

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

num_features = 80
correlation = 3  # Correlation order
num_atoms = 128  # number of atoms

node_feats = torch.randn(num_atoms , num_features, num_irreps, device='cuda')

weights = {3: torch.randn((1270, num_features), device='cuda', requires_grad=True),
           2:torch.randn((24, num_features), device='cuda', requires_grad=True),
           1: torch.randn((3, num_features), device='cuda', requires_grad=True) }

# warmup
torch.matmul(torch.rand(1024, 1024, device='cuda'), torch.rand(1024, 1024, device='cuda'))

start.record()
test1 = torch.einsum('...ik,kc ->...ic', U_tensors[3], weights[3])
end.record()
torch.cuda.synchronize()
print("einsum", start.elapsed_time(end), "ms") 

print (test1)
print ("Uw Shape:", test1.shape)

Uw_nonzero_idx = []

for i in range(test1.shape[0]):
    for j in range(test1.shape[1]):
        for k in range(test1.shape[2]):
            nonzero_idx, = torch.where(test1[i, j, k] != 0)
            
            if (len(nonzero_idx) > 0):
                Uw_nonzero_idx.append([i, j, k])

U_indices = torch.load("StructU_indices.pt")
U_nvals = torch.load("StructU_nvals.pt")
U_vals = torch.load("StructU_vals.pt")

print (U_vals.shape)
print (U_indices.shape)

U_indices_nonzero = []
for i in range(U_nvals.shape[0]):
    for j in range(U_nvals.shape[1]):
        for k in range(U_nvals.shape[2]):
            nonzero_idx, = torch.where(U_nvals[i, j, k] != 0)
            
            if (len(nonzero_idx) > 0):
                U_indices_nonzero.append([i, j, k])

U_indices_nonzero = torch.tensor(U_indices_nonzero)
Uw_nonzero_idx = torch.tensor(Uw_nonzero_idx)

print (Uw_nonzero_idx, Uw_nonzero_idx.shape)
print (U_indices_nonzero, U_indices_nonzero.shape)

print (torch.linalg.norm(Uw_nonzero_idx.float() - U_indices_nonzero.float()))

weights = torch.zeros(48, 48, 12, 96, device='cuda')

#---- 3 torch.Size([48, 48, 48, 1270]) ----
# weights: 48,48,48,12,96
for i in range(48):
    for j in range(48):
        # print (U_nvals[i, j])
        U_indices[i, j]  # 48,12
        
