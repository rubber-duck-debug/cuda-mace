import torch

class InvariantMessagePassingTP(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.cuda_obj = torch.classes.inv_message_passing.InvariantMessagePassingTP()

    def forward(
        self,
        node_feats: torch.Tensor,  # [nnodes, nfeats]
        edge_attrs: torch.Tensor,  # [nedges, 16]
        tp_weights: torch.Tensor,  # [nedges, 4, nfeats]
        # [nedges] -> must be monotonically increasing
        sender_list: torch.Tensor,
        # [nedges] -> must be monotonically increasing
        receiver_list: torch.Tensor,
        nnodes: int,
    ):
        return self.cuda_obj.forward(
            node_feats, edge_attrs, tp_weights, sender_list, receiver_list, nnodes
        )  # outputs [nnodes, 16, nfeats]
