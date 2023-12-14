from mace_ops import cuda
import torch

class InvariantMessagePassingTP(torch.nn.Module):

    def __init__(self):

        super().__init__()
    
    def forward(
            self, 
            node_feats: torch.Tensor, # [nedges, nfeats]
            edge_attrs: torch.Tensor, # [nedges, 16]
            tp_weights: torch.Tensor, # [nedges, 4, nfeats]
            receiver_list: torch.Tensor, #[nedges] -> must be monotonically increasing
            nnodes
            ):

        return torch.ops.invariant_tp.forward(node_feats, edge_attrs, tp_weights, receiver_list, nnodes) # outputs [nnodes, 16, nfeats]