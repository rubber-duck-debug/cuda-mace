from math import prod
import torch
from e3nn import o3
from mace_ops.ops.linear import ElementalLinear 

    
class LinearElementRef(torch.nn.Module):

    def __init__(self, irreps_in, irreps_out, e3nn_instructions, e3nn_weights, num_elements):

        super().__init__()
        self.num_elements = num_elements
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out

        self.e3nn_instructions = e3nn_instructions
        self.e3nn_weights = e3nn_weights

        self.out_lmax = int(irreps_out.lmax)
        self.out_dim = int(irreps_out.dim / (self.out_lmax + 1) ** 2)

        self.instructions = []

        flat_weight_index = 0
        for ins in e3nn_instructions:
            path_nweight = prod(ins.path_shape)
            mul_ir_out = irreps_out[ins.i_out]
            # extract the weights for the current path
            w = e3nn_weights.narrow(-1, flat_weight_index, path_nweight)
            w = w.reshape([-1] + list(ins.path_shape))
            # 0 | 1 2 3 | 4 5 6
            start = ins.i_in ** 2
            end = start + (2 * ins.i_in + 1)

            print (ins, w.shape)
            self.instructions.append((start, end, w, ins.path_weight))

            flat_weight_index += path_nweight

    def forward(self, x, y):
        # x : [batch,  num_l, num_channels]
        # y : [batch, num_elements]
        output = torch.zeros(x.shape[0], (self.out_lmax + 1) ** 2, self.out_dim,
                             device='cuda', dtype=torch.float32)
        for elem in range(self.num_elements):
            idx_elem = y[:, elem] == 1
            x_elem = x[idx_elem, :]
            if x_elem.shape[0] == 0:
                continue
            for i, instruction in enumerate(self.instructions):
                start_l_idx, end_l_idx, weights, path_weight = instruction
                output[idx_elem, start_l_idx:end_l_idx, :] += path_weight * \
                    torch.matmul(x_elem[:, start_l_idx:end_l_idx, :], weights[elem, :,:])

        return output
    
from math import sqrt
nnodes = 1000
max_l = 3
n_channels = 96
nelements = 10
# get name of the class model.interactions[0]  
node_feats_irreps = o3.Irreps("96x0e + 96x1o + 96x2e + 96x3o")
node_attrs_irreps = o3.Irreps("10x0e")
hidden_irreps = o3.Irreps("96x0e + 96x1o + 96x2e + 96x3o")    
skip_tp = o3.FullyConnectedTensorProduct(
            node_feats_irreps, node_attrs_irreps, hidden_irreps
        ).to("cuda")
linear = o3.Linear(node_feats_irreps, node_feats_irreps).to("cuda")
x = torch.randn(nnodes, (max_l+1)**2, n_channels,
                device='cuda', dtype=torch.float32, requires_grad=True)
one_hot_embedding = torch.randn(nnodes, nelements, device='cuda', dtype=torch.float)
one_hot_embedding[:] = 0.0
one_hot_embedding[0:10, 0] = 1
one_hot_embedding[10:90,1] = 1
one_hot_embedding[90:,2] = 1
ws = skip_tp.weight.data.reshape([4,96,10,96]).permute(2,0,1,3)
ws = ws.flatten(1) / sqrt(10)
linear_element_ref = LinearElementRef(node_feats_irreps, node_feats_irreps, linear.instructions, ws, nelements).to("cuda")
linear_element_cuda = ElementalLinear(node_feats_irreps, node_feats_irreps, linear.instructions, ws, nelements)

out_ref = linear_element_ref(x, one_hot_embedding)
out_cuda = linear_element_cuda(x, one_hot_embedding)

assert torch.allclose(out_ref, out_cuda, atol=1e-5)
    
