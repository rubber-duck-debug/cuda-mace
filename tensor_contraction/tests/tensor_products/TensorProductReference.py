import torch
import numpy as np
from torch.utils import cpp_extension
from e3nn import o3
from torch.profiler import profile, record_function, ProfilerActivity
from typing import Tuple, List

class TensorProductReference(torch.nn.Module):

  def __init__(self, irreps1, irreps2, target_irreps, nchannels, n_l_channels, weights=None, device="cpu"):
    super().__init__()
    self.irreps1 = irreps1
    self.irreps2 = irreps2
    self.target_irreps = target_irreps

    self.dim_out = 0
    self.device = device
    
    self.mu_dict = {}
    self.cg_sparse_dict = {}
    self.offsets_dict = {}
    
    offset_sph_harm = 0

    for l in range (9):
      self.offsets_dict[l] = offset_sph_harm
      offset_sph_harm += (2 * l) + 1
    
    self.n_l_channels = n_l_channels
    self.nchannels = nchannels
    
    if (weights == None):
        self.weights = torch.randn(nchannels, n_l_channels).cuda()
    else: 
        self.weights = weights
    
    l_channel = 0
    
    self.weight_indices = {}
    
    for i, (mul, ir_in) in enumerate(self.irreps1):
        for j, (_, ir_edge) in enumerate(self.irreps2):
            for ir_out in ir_in * ir_edge:  # | l1 - l2 | <= l <= l1 + l2
                if ir_out in target_irreps:
                    l1 = ir_in.l
                    l2 = ir_edge.l
                    l3 = ir_out.l
                    
                    cg = o3.wigner_3j(l1,l2,l3).to(self.device)
                    x,y,z = cg.shape[0],  cg.shape[1],  cg.shape[2]
                    
                    mu1, mu2, mu3 = cg.nonzero(as_tuple=True)
                    cg_sparse = cg[(mu1, mu2, mu3)]

                    sorted_indices = mu3.argsort()
                    
                    mu1 = mu1[sorted_indices]
                    mu2 = mu2[sorted_indices]
                    mu3 = mu3[sorted_indices]
                    cg_sparse = cg_sparse[sorted_indices]
                    
                    self.mu_dict[l1, l2, l3] = mu1, mu2, mu3
                    self.cg_sparse_dict[l1, l2, l3] = cg_sparse
        
                    self.dim_out += z
                    
                    weight_idxs = torch.ones_like(mu1)

                    self.weight_indices[l1, l2, l3] = l_channel
                    
                    l_channel +=1
                  

    assert l_channel == n_l_channels, "n_l_channels must equal the number of representation products"
    
    
  def forward(self, x, y):

    all_outputs = []
    
    for i in range(x.shape[0]): # loop over edges

        outputs = []

        for j, (mul, ir_in) in enumerate(self.irreps1):
            for k, (_, ir_edge) in enumerate(self.irreps2):
                for ir_out in ir_in * ir_edge:  # | l1 - l2 | <= l <= l1 + l2
                    if ir_out in self.target_irreps:
                                                
                        l1 = ir_in.l
                        l2 = ir_edge.l
                        l3 = ir_out.l
                        
                        mu1, mu2, mu3 = self.mu_dict[l1, l2, l3]
                        cg_sparse = self.cg_sparse_dict[l1, l2, l3]
                        
                        cg_iteration = x[i, :, self.offsets_dict[l1] + mu1] * cg_sparse * y[i, :, self.offsets_dict[l2] + mu2]
                    
                        output = torch.zeros(x.shape[1], (2 * l3 + 1), device=self.device)
                    
                        output.index_add_(1, mu3, cg_iteration)
                        
                        outputs.append(output)
                        
        output_i = torch.cat(outputs, dim=-1)
        
        all_outputs.append(output_i)
        
    return torch.stack(all_outputs)

    
  def weighted_forward(self, x, y):

    output = torch.zeros(x.shape[0], x.shape[1], device=self.device)
    
    for i in range(x.shape[0]): # loop over edges

        for j, (mul, ir_in) in enumerate(self.irreps1):
            for k, (_, ir_edge) in enumerate(self.irreps2):
                for ir_out in ir_in * ir_edge:  # | l1 - l2 | <= l <= l1 + l2
                    if ir_out in self.target_irreps:
                                                
                        l1 = ir_in.l
                        l2 = ir_edge.l
                        l3 = ir_out.l
                        
                        mu1, mu2, mu3 = self.mu_dict[l1, l2, l3]
                        cg_sparse = self.cg_sparse_dict[l1, l2, l3]
                        
                        cg_iteration = x[i, :, self.offsets_dict[l1] + mu1] * cg_sparse * y[i, :, self.offsets_dict[l2] + mu2]
                        
                        output[i] += self.weights[:, self.weight_indices[l1, l2, l3]] * torch.sum(cg_iteration, dim=-1)

    return output

if __name__ == "__main__":

    nchannels=256

    irreps1, irreps2, target_irreps = o3.Irreps(str(nchannels) +"x0e + " + str(nchannels)+"x1o"), o3.Irreps("1x0e + 1x1o + 1x2e + 1x3o"), \
                                      o3.Irreps(str(nchannels)+"x0e + " + str(nchannels)+"x1o +" + str(nchannels)+"x2e + " + str(nchannels)+"x3o")
    
    tp_custom = TensorProductReference(irreps1,irreps2,target_irreps,nchannels, device="cuda")
    
    l1 = 1
    l2 = 3
    
    n_edges = 2

    X1 = torch.randn(n_edges, nchannels, (l1 + 1)**2)
    X2 = torch.randn(n_edges, 1, (l2 + 1)**2)
    
    X1 = X1.to("cuda")
    X2 = X2.to("cuda")
    
    out = tp_custom.forward(X1, X2)
    
    print (out)
    print (out.shape)
