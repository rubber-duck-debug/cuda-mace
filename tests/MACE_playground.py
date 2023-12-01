import numpy as np
import torch
import torch.nn.functional
from e3nn import o3
from matplotlib import pyplot as plt
from typing import Tuple
from mace import data, modules, tools
from mace.tools import torch_geometric

z_table = tools.AtomicNumberTable([1, 8])
atomic_energies = np.array([-1.0, -3.0], dtype=float)
cutoff = 4

def get_edge_vectors_and_lengths(
    positions: torch.Tensor,  # [n_nodes, 3]
    edge_index: torch.Tensor,  # [2, n_edges]
    shifts: torch.Tensor,  # [n_edges, 3]
    normalize: bool = False,
    eps: float = 1e-9,
) -> Tuple[torch.Tensor, torch.Tensor]:
    sender = edge_index[0]
    receiver = edge_index[1]
    vectors = positions[receiver] - positions[sender] + shifts  # [n_edges, 3]
    lengths = torch.linalg.norm(vectors, dim=-1, keepdim=True)  # [n_edges, 1]
    if normalize:
        vectors_normed = vectors / (lengths + eps)
        return vectors_normed, lengths

    return vectors, lengths


model_config = dict(
        num_elements=2,  # number of chemical elements
        atomic_energies=atomic_energies,  # atomic energies used for normalisation
        avg_num_neighbors=8,  # avg number of neighbours of the atoms, used for internal normalisation of messages
        atomic_numbers=z_table.zs,  # atomic numbers, used to specify chemical element embeddings of the model
        r_max=cutoff,  # cutoff
        num_bessel=8,  # number of radial features
        num_polynomial_cutoff=6,  # smoothness of the radial cutoff
        max_ell=3,  # expansion order of spherical harmonic adge attributes
        num_interactions=2,  # number of layers, typically 2
        interaction_cls_first=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],  # interation block of first layer
        interaction_cls=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],  # interaction block of subsequent layers
        hidden_irreps=o3.Irreps("32x0e + 32x1o"),  # 32: number of embedding channels, 0e, 1o is specifying which equivariant messages to use. Here up to L_max=1
        correlation=3,  # correlation order of the messages (body order - 1)
        MLP_irreps=o3.Irreps("16x0e"),  # number of hidden dimensions of last layer readout MLP
        gate=torch.nn.functional.silu,  # nonlinearity used in last layer readout MLP
    )
model = modules.MACE(**model_config)

print(model)

config = data.Configuration(
    atomic_numbers=np.array([8, 1, 1]),
    positions=np.array(
        [
            [0.0, -2.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    ),
    forces=np.array(
        [
            [0.0, -1.3, 0.0],
            [1.0, 0.2, 0.0],
            [0.0, 1.1, 0.3],
        ]
    ),
    energy=-1.5,
)

atomic_data = data.AtomicData.from_config(config, z_table=z_table, cutoff=float(model.r_max))
data_loader = torch_geometric.dataloader.DataLoader(
        dataset=[atomic_data],
        batch_size=1,
        shuffle=True,
        drop_last=False,
    )
batch = next(iter(data_loader))
print("The data is stored in batches. Each batch is a single graph, potentially made up of several disjointed sub-graphs corresponding to different chemical structures. ")
print(batch)
print("\nbatch.edge_index contains which atoms are connected within the cutoff. It is the adjacency matrix in sparse format.\n")
print(batch.edge_index)


# create random set of points on the unit sphere and plot them
n = 200
points = torch.randn(n, 3)
points = points / points.norm(dim=-1, keepdim=True)

l_max = model.spherical_harmonics.lmax
print("l_max =",l_max)

edge_attrs = model.spherical_harmonics(points)
print("shape:", edge_attrs.shape)
print("number of edges:", edge_attrs.shape[0])
print("number of features: (2l + 1)^2=", edge_attrs.shape[1])

vectors, lengths = get_edge_vectors_and_lengths(
            positions=batch["positions"],
            edge_index=batch["edge_index"],
            shifts=batch["shifts"],
        )
edge_attrs = model.spherical_harmonics(vectors)
print(f"The edge attributes have shape (num_edges, num_spherical_harmonics)\n", edge_attrs.shape)

dists = torch.tensor(np.linspace(0.1, 5.5, 100), dtype=torch.get_default_dtype()).unsqueeze(-1)

radials = model.radial_embedding(dists)

edge_feats = model.radial_embedding(lengths)
print("The edge features have shape (num_edges, num_radials)")
print(edge_feats.shape)

atomic_numbers = [8, 1, 1]  # the atomic numbers of the structure evaluated
indices = tools.utils.atomic_numbers_to_indices(atomic_numbers, z_table=z_table)
node_attrs = tools.torch_tools.to_one_hot(
            torch.tensor(indices, dtype=torch.long).unsqueeze(-1),
            num_classes=len(z_table),
        )
print(node_attrs)  # node attributes are the one hot encoding of the chemical  elements of each node

print("Weights are internally flattened and have a shape",
      model.node_embedding.linear.__dict__['_parameters']['weight'].shape)

print("\nThis corresponds to (num_chemical_elements, num_channels) learnable embeddings for each chemical element with shape:",
      model.node_embedding.linear.__dict__['_parameters']['weight'].reshape((2, 32)).shape)

# In MACE we create the initial node features using this block:
node_feats = model.node_embedding(node_attrs)

# chemical elements are embedded into 32 channels of the model. These 32 numbers are the initial node features.
print("The node embedding block returns (num_atoms, num_channels) shaped tensor:", node_feats.shape)

node_feats = model.interactions[0].linear_up(node_feats)

print ("node_feats after linear:", node_feats.shape)

tp_weights = model.interactions[0].conv_tp_weights(edge_feats)
print("tp weights:", tp_weights.shape)

sender, receiver = batch["edge_index"] # use the graph to get the sender and receiver indices

print ("sender", sender)
print ("receiver", receiver)

#batch.edge_index contains which atoms are connected within the cutoff. It is the adjacency matrix in sparse format.

#sender = edge_index[0]
#receiver = edge_index[1]
#vectors = positions[receiver] - positions[sender] + shifts 
    
#tensor([[0, 0, 1, 1, 2, 2], -> sender
#        [1, 2, 0, 2, 0, 1]]) -. receiver

mji = model.interactions[0].conv_tp(
            node_feats[sender], edge_attrs, tp_weights
        )
print("The first dimension is the number of edges, highlighted by the ij in the variable name", mji.shape)
print(f"The second dimension is num_channels * num_paths dimensional, in this case: {mji.shape[-1]} = 32 * {mji.shape[-1] // 32}", )

from mace.tools.scatter import scatter_sum
message = scatter_sum(
            src=mji, index=receiver, dim=0, dim_size=node_feats.shape[0]
        )
print("The messages have first dimension corresponding to the nodes i:", message.shape)