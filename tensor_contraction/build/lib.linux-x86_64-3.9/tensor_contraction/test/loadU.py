from typing import Dict

import torch
from tensor_contraction.cuda import sparse_stuff


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
        
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        print (self.U_tensors[self.correlation].shape, weights[self.correlation].shape, node_feats.shape)
        
        print ("non-zero:", torch.count_nonzero(self.U_tensors[self.correlation][23][2]))
        
        start.record()
        test_fp32 = sparse_stuff.get_uw_contraction_fp32(self.U_tensors[self.correlation], weights[self.correlation])
        end.record()
        torch.cuda.synchronize()
        print("contract U, weights custom FP32: ", start.elapsed_time(end), "ms")
        
        start.record()
        test_fp64 = sparse_stuff.get_uw_contraction_fp64(self.U_tensors[self.correlation].double(), weights[self.correlation].double())
        end.record()
        torch.cuda.synchronize()
        print("contract U, weights custom FP64: ", start.elapsed_time(end), "ms")
        
        start.record()
        test1 = torch.einsum('...ik,kc ->...ic',
                               self.U_tensors[self.correlation],
                               weights[self.correlation])
        
        end.record()
        torch.cuda.synchronize()
        print("contract U, weights einsum fp32: ", start.elapsed_time(end), "ms")
        
        start.record()
        test1_fp64 = torch.einsum('...ik,kc ->...ic',
                               self.U_tensors[self.correlation].double(),
                               weights[self.correlation].double())
        
        end.record()
        torch.cuda.synchronize()
        print("contract U, weights einsum FP64: ", start.elapsed_time(end), "ms")
        
        print (torch.linalg.norm(test_fp32 - test1_fp64))
        
        start.record()
        out = torch.einsum(self.equation_main,
                               self.U_tensors[self.correlation],
                               weights[self.correlation],
                               node_feats)  # Conctract the U tensor, the weights and the representations
        end.record()
        torch.cuda.synchronize()
        
        print("contract U, weights, rep: ", start.elapsed_time(end), "ms")
            
        print ("out shape:", out.shape)
        
        start.record()
        
        for corr in range(self.correlation - 1, 0, -1):  # Loop over the correlation order and contract
                c_tensor = torch.einsum(self.equation_weighting,
                                        U_tensors[corr],
                                        weights[corr])
                c_tensor = c_tensor + out
                out = torch.einsum(self.equation_contract, c_tensor, node_feats)
                
        end.record()
        torch.cuda.synchronize()
        
        print("correlation order contract: ", start.elapsed_time(end), "ms")
        
        return out


# warmup
torch.matmul(torch.rand(1024, 1024, device='cuda'), torch.rand(1024, 1024, device='cuda'))

# Define some parameters for the contraction
l_max = 3
num_irreps = ((l_max + 1) ** 2) * 3

print ("num_irreps: ", num_irreps)

U_tensors = torch.load("U_tensors_n_s_fp32.pt")

for key in U_tensors.keys():
    U_tensors[key] = U_tensors[key].cuda().float()
    
    print ("----", key, U_tensors[key].shape, "----")
    
    sparse_U = U_tensors[key].to_sparse()
    
    indexes = sparse_U.indices()

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
stuff = torch.where(U_tensors[3] != 0)
end.record()
torch.cuda.synchronize()

num_features = 80
correlation = 3  # Correlation order
num_atoms = 128  # number of atoms

Conv_vect = Convolution_vect(U_tensors, num_features, correlation)

node_feats = torch.randn(num_atoms , num_features, num_irreps, device='cuda')

weights = {3: torch.randn((1270, num_features), device='cuda', requires_grad=True),
           2:torch.randn((24, num_features), device='cuda', requires_grad=True),
           1: torch.randn((3, num_features), device='cuda', requires_grad=True) }

x_vect = Conv_vect(node_feats, weights)

start.record()
test1 = torch.einsum('...ik,kc ->...ic', U_tensors[3], weights[3])
end.record()
torch.cuda.synchronize()
print("einsum", start.elapsed_time(end), "ms") 

U_indices = torch.load("StructU_indices.pt")
U_nvals = torch.load("StructU_nvals.pt")
U_vals = torch.load("StructU_vals.pt")

start.record()
test2 = sparse_stuff.get_sparse_uw_contraction_fp32(U_nvals.int(), U_indices.int(), U_tensors[3], weights[3])
end.record()
torch.cuda.synchronize()

print("sparse fp32", start.elapsed_time(end), "ms") 

start.record()
test3 = sparse_stuff.get_wmma_uw_contraction_fp32(U_vals, U_indices.int(), U_nvals.int(), weights[3])
end.record()
torch.cuda.synchronize()

print (test3)
print("hmm", start.elapsed_time(end), "ms") 

