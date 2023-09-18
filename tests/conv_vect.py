from typing import Dict

import torch

class Convolution_vect(torch.nn.Module):
    def __init__(self,
                 U_tensors: Dict[int,torch.tensor],
                 num_features: torch.tensor,
                 correlation: int,):
        super().__init__() 
        self.U_tensors = U_tensors   #[(lmax+1)**2]**correlation + [num_weights]
        self.num_features = num_features
        self.correlation = correlation
        self.equation_main = '...ik,kc,bci -> bc...'
        self.equation_weighting = '...k,kc->c...'
        self.equation_contract = 'bc...i,bci->bc...'
    def forward(self,
                node_feats: torch.tensor,
                weights: Dict[int,torch.tensor]):
        out = torch.einsum(self.equation_main,
                               self.U_tensors[self.correlation],
                               weights[self.correlation],
                               node_feats) #Conctract the U tensor, the weights and the representations
        for corr in range(self.correlation-1,-1,-1): #Loop over the correlation order and contract
                c_tensor = torch.einsum(self.equation_weighting,
                                        U_tensors[corr],
                                        weights[corr])
                c_tensor  = c_tensor + out
                out = torch.einsum(self.equation_contract,c_tensor,node_feats)
        return out

#Define some parameters for the contraction
l_max = 3
num_irreps = (l_max + 1)**2
U_tensors = {2 : torch.randn(num_irreps,num_irreps,num_irreps,120,device='cuda'),1: torch.randn(num_irreps,num_irreps,8,device='cuda'), 0: torch.randn(num_irreps,4,device='cuda')}
num_features = 32
correlation = 2 #Correlation order minus one
num_atoms = 135 #number of atoms

Conv_vect = Convolution_vect(U_tensors,num_features,correlation)



node_feats = torch.randn(num_atoms ,num_features,num_irreps,device='cuda')
weights = {2 : torch.randn((120,num_features),device='cuda',requires_grad=True), 1:torch.randn((8,num_features),device='cuda',requires_grad=True), 0: torch.randn((4,num_features),device='cuda',requires_grad=True) }
x_vect = Conv_vect(node_feats,weights)
 

x_vect = Conv_vect(node_feats,weights)

print (x_vect.shape)