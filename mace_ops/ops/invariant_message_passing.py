from mace_ops import cuda
import torch

class InvariantMessagePassingTP(torch.nn.Module):

    def __init__(self):

        super().__init__()
    
    def calculate_first_occurences(
            self, 
            reciever_list: torch.Tensor,
            nnodes: int,
            sorted_idx : torch.Tensor = torch.Tensor()
            ) -> torch.Tensor:
        
            '''
            For every node, computes the first occurence of that node index in receiver_list. Simple method for computing per-node neighbourlist begin/end indices.
            
            Output has shape [nnodes]. output[0] contains the 0th's node first edge index in the edge list. output[1] contains the first occurence of the 1st node edge,
            so edge-boundaries can be computed as start = [node_id], end = [node_id + 1] -1.
            
            supports nodes without neighbours.
            '''
            return torch.ops.invariant_tp.calculate_first_occurences(reciever_list, nnodes, 64, sorted_idx)
        
    def forward(
            self, 
            node_feats: torch.Tensor, # [nnodes, nfeats]
            edge_attrs: torch.Tensor, # [nedges, 16]
            tp_weights: torch.Tensor, # [nedges, 4, nfeats]
            sender_list: torch.Tensor, # [nedges] -> 
            receiver_list: torch.Tensor, #[nedges] -> must be monotonically increasing
            first_occurences: torch.Tensor  #[nnodes] -> monotonically increasing by construction
            ) -> torch.Tensor:

        return torch.ops.invariant_tp.forward(node_feats, edge_attrs, tp_weights, sender_list, receiver_list, first_occurences) # outputs [nnodes, 16, nfeats]