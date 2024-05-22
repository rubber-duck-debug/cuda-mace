from typing import Any, Callable, Dict, List, Optional, Type, Union, Tuple

import ase
import numpy as np
import torch
from mace.tools import torch_geometric
from mace import data, tools
from e3nn import o3, nn
from mace import modules
from time import time
import inspect

from mace.modules.blocks import SphericalHarmonics

from mace.tools.scatter import scatter_sum

from mace.modules.blocks import (
    AtomicEnergiesBlock,
    EquivariantProductBasisBlock,
    InteractionBlock,
    LinearDipoleReadoutBlock,
    LinearNodeEmbeddingBlock,
    LinearReadoutBlock,
    NonLinearDipoleReadoutBlock,
    NonLinearReadoutBlock,
    RadialEmbeddingBlock,
    ScaleShiftBlock,
)
from mace.modules.utils import (
    compute_fixed_charge_dipole,
    compute_forces,
    get_edge_vectors_and_lengths,
    get_outputs,
    get_symmetric_displacement,
)

from mace.modules.irreps_tools import (
    linear_out_irreps,
    reshape_irreps,
    tp_out_irreps_with_instructions,
)

from mace_ops.ops.invariant_message_passing import InvariantMessagePassingTP
from mace_ops.ops.linear import Linear, ElementalLinear
from mace_ops.ops.symmetric_contraction import SymmetricContraction as CUDAContraction


def compute_forces_virials(
    energy: torch.Tensor,
    positions: torch.Tensor,
    displacement: torch.Tensor,
    cell: torch.Tensor,
    training: bool = True,
    compute_stress: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(energy)]
    forces, virials = torch.autograd.grad(
        outputs=[energy],  # [n_graphs, ]
        inputs=[positions, displacement],  # [n_nodes, 3]
        grad_outputs=grad_outputs,
        retain_graph=training,  # Make sure the graph is not destroyed during training
        create_graph=training,  # Create graph for second derivative
        allow_unused=True,
    )
    stress = torch.zeros_like(displacement)
    if compute_stress and virials is not None:
        cell = cell.view(-1, 3, 3)
        volume = torch.einsum(
            "zi,zi->z",
            cell[:, 0, :],
            torch.cross(cell[:, 1, :], cell[:, 2, :], dim=1),
        ).unsqueeze(-1)
        stress = virials / volume.view(-1, 1, 1)
    if forces is None:
        forces = torch.zeros_like(positions)
    if virials is None:
        virials = torch.zeros((1, 3, 3))

    return -1 * forces, -1 * virials, stress

# nb_iters = 30
#     warmup_iters = 20

#     for i in range (nb_iters):
#         start = time()
#         if i == warmup_iters: torch.cuda.cudart().cudaProfilerStart()
#         if i >= warmup_iters: torch.cuda.nvtx.range_push("iteration{}".format(i))
#         if i >= warmup_iters: torch.cuda.nvtx.range_push("forward")
#         cuda_out = tp.forward(
#             node_feats_cuda,
#             edge_attrs_cuda,
#             radial_feats_cuda,
#             sender,
#             receiver,
#             nnodes,
#         )
#         if i >= warmup_iters: torch.cuda.nvtx.range_pop()

#         if (args.grad):
#             os = cuda_out.sum()
            
#             if i >= warmup_iters: torch.cuda.nvtx.range_push("backward")
#             os.backward()
#             if i >= warmup_iters: torch.cuda.nvtx.range_pop()
            
#         if i >= warmup_iters: torch.cuda.nvtx.range_pop()     
#         end = time()
        
#         print ((end - start) * 1000)       
#     torch.cuda.cudart().cudaProfilerStop()

def optimize_cuda_mace(model) -> None:
    """
    Optimize the MACE model for CUDA inference.
    """
    for param in model.parameters():
        param.requires_grad = False
    dtype = get_model_dtype(model)
    n_layers = int(model.num_interactions)
    sh_irreps = o3.Irreps.spherical_harmonics(3)
    # spherical_harmonics = SphericalHarmonics(
    #     sh_irreps=sh_irreps,
    #     normalize=True,
    #     normalization="component",
    #     backend="opt",
    # )
    
    #spherical_harmonics = torch.classes.spherical_harmonics.SphericalHarmonics()
    
    #model.spherical_harmonics = spherical_harmonics
    num_elements = model.node_embedding.linear.irreps_in.num_irreps
    for i in range(n_layers):
        model.interactions[i].linear_up = linear_matmul(model.interactions[i].linear_up)
        model.interactions[i].linear = linear_to_cuda(model.interactions[i].linear)
        model.interactions[i].tp = InvariantMessagePassingTP()
        if "Residual" in model.interactions[i].__class__.__name__:
            bound_method = invariant_residual_interaction_forward.__get__(
                model.interactions[i], model.interactions[i].__class__
            )
            setattr(model.interactions[i], "forward", bound_method)
        else:
            model.interactions[i].skip_tp = element_linear_to_cuda(
                model.interactions[i].skip_tp
            )
            bound_method = invariant_interaction_forward.__get__(
                model.interactions[i], model.interactions[i].__class__
            )
            setattr(model.interactions[i], "forward", bound_method)

        symm_contract = model.products[i].symmetric_contractions
        all_weights = {}
        for j in range(len(symm_contract.contractions)):
            all_weights[str(j)] = {}
            all_weights[str(j)][3] = (
                symm_contract.contractions[j].weights_max.detach().clone().type(dtype)
            )
            all_weights[str(j)][2] = (
                symm_contract.contractions[j].weights[0].detach().clone().type(dtype)
            )
            all_weights[str(j)][1] = (
                symm_contract.contractions[j].weights[1].detach().clone().type(dtype)
            )
        irreps_in = o3.Irreps(model.products[i].symmetric_contractions.irreps_in)
        coupling_irreps = o3.Irreps([irrep.ir for irrep in irreps_in])
        irreps_out = o3.Irreps(model.products[i].symmetric_contractions.irreps_out)
        symmetric_contractions = CUDAContraction(
            coupling_irreps,
            irreps_out,
            all_weights,
            nthreadX=32,
            nthreadY=4,
            nthreadZ=1,
            dtype=dtype,
        )
        model.products[i].symmetric_contractions = SymmetricContractionWrapper(
            symmetric_contractions
        )
        model.products[i].linear = linear_matmul(model.products[i].linear)
    return model


class SymmetricContractionWrapper(torch.nn.Module):
    def __init__(self, symmetric_contractions):
        super().__init__()
        self.symmetric_contractions = symmetric_contractions

    def forward(self, x, y):
        y = y.argmax(dim=-1).int()
        out = self.symmetric_contractions(x, y).squeeze()
        return out


def get_model_dtype(model: torch.nn.Module) -> torch.dtype:
    """Get the dtype of the model"""
    model_dtype = next(model.parameters()).dtype
    return model_dtype


class linear_matmul(torch.nn.Module):
    def __init__(self, linear_e3nn):
        super().__init__()
        num_channels_in = linear_e3nn.__dict__["irreps_in"].num_irreps
        num_channels_out = linear_e3nn.__dict__["irreps_out"].num_irreps
        self.weights = (
            linear_e3nn.weight.data.reshape(num_channels_in, num_channels_out)
            / num_channels_in**0.5
        )

    def forward(self, x):
        return torch.matmul(x, self.weights)


def linear_to_cuda(linear):
    return Linear(
        linear.__dict__["irreps_in"],
        linear.__dict__["irreps_out"],
        linear.instructions,
        linear.weight,
    )


def element_linear_to_cuda(skip_tp):
    #print("elementlinear", skip_tp)
    num_elements = skip_tp.__dict__["irreps_in2"].dim
    n_channels = skip_tp.__dict__["irreps_in1"][0].dim
    lmax = skip_tp.__dict__["irreps_in1"].lmax
    ws = skip_tp.weight.data.reshape(
        [lmax + 1, n_channels, num_elements, n_channels]
    ).permute(2, 0, 1, 3)
    ws = ws.flatten(1) / sqrt(num_elements)
    linear_instructions = o3.Linear(
        skip_tp.__dict__["irreps_in1"], skip_tp.__dict__["irreps_out"]
    )
    return ElementalLinear(
        skip_tp.__dict__["irreps_in1"],
        skip_tp.__dict__["irreps_out"],
        linear_instructions.instructions,
        ws,
        num_elements,
    )


def invariant_residual_interaction_forward(
    self,
    node_attrs: torch.Tensor,
    node_feats: torch.Tensor,
    edge_attrs: torch.Tensor,
    edge_feats: torch.Tensor,
    edge_index: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    sender = edge_index[0]
    receiver = edge_index[1]
    num_nodes = torch.tensor(node_feats.shape[0])

    torch.cuda.nvtx.range_push("RInteraction::skip tp")
    sc = self.skip_tp(node_feats, node_attrs)
    torch.cuda.nvtx.range_pop()
        
    
    torch.cuda.nvtx.range_push("RInteraction::linear up")
    node_feats = self.linear_up(node_feats)
    torch.cuda.nvtx.range_pop()
        
    #torch.cuda.nvtx.range_push("RInteraction::tp weights")
    #tp_weights = self.conv_tp_weights(edge_feats)
    #torch.cuda.nvtx.range_pop()
    
    torch.cuda.nvtx.range_push("RInteraction::inv tp")
    message = self.tp.forward(
        node_feats,
        edge_attrs,
        edge_feats.view(edge_feats.shape[0], -1, node_feats.shape[-1]),
        sender.int(),
        receiver.int(),
        num_nodes,
    )
    torch.cuda.nvtx.range_pop()

    
    torch.cuda.nvtx.range_push("RInteraction::message linear")
    message = self.linear(message) / self.avg_num_neighbors
    torch.cuda.nvtx.range_pop()
    
    return (
        message,
        sc,
    )  # [n_nodes, channels, (lmax + 1)**2]


def invariant_interaction_forward(
    self,
    node_attrs: torch.Tensor,
    node_feats: torch.Tensor,
    edge_attrs: torch.Tensor,
    edge_feats: torch.Tensor,
    edge_index: torch.Tensor,
) -> Tuple[torch.Tensor, None]:
    sender = edge_index[0]
    receiver = edge_index[1]
    num_nodes = torch.tensor(node_feats.shape[0])
    
    if (self.profile):
        torch.cuda.nvtx.range_push("Interaction::linear up")
    node_feats = self.linear_up(node_feats)
    if (self.profile):
        torch.cuda.nvtx.range_pop()
    
    #if (self.profile):
    #    torch.cuda.nvtx.range_push("Interaction::tp weights")
    #tp_weights = self.conv_tp_weights(edge_feats)
    #if (self.profile):
    #    torch.cuda.nvtx.range_pop()
    
    if (self.profile):
        torch.cuda.nvtx.range_push("Interaction::inv tp")
    message = self.tp.forward(
        node_feats,
        edge_attrs,
        edge_feats.view(edge_feats.shape[0], -1, node_feats.shape[-1]),
        sender.int(),
        receiver.int(),
        num_nodes,
    )
    if (self.profile):
        torch.cuda.nvtx.range_pop()

    if (self.profile):
        torch.cuda.nvtx.range_push("Interaction::message linear")
    message = self.linear(message) / self.avg_num_neighbors
    if (self.profile):
        torch.cuda.nvtx.range_pop()

    if (self.profile):
        torch.cuda.nvtx.range_push("Interaction::skip tp")
    message = self.skip_tp(message, node_attrs)
    if (self.profile):
        torch.cuda.nvtx.range_pop()

    return (
        message,
        None,
    )  # [n_nodes, channels, (lmax + 1)**2]


class RealAgnosticResidualInteractionBlock(InteractionBlock):
    def _setup(self) -> None:
        # First linear
        self.linear_up = o3.Linear(
            self.node_feats_irreps,
            self.node_feats_irreps,
            internal_weights=True,
            shared_weights=True,
        )
        # TensorProduct
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            self.target_irreps,
        )
        self.conv_tp = o3.TensorProduct(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
        )

        # Convolution weights
        input_dim = self.edge_feats_irreps.num_irreps
        self.conv_tp_weights = nn.FullyConnectedNet(
            [input_dim] + self.radial_MLP + [self.conv_tp.weight_numel],
            torch.nn.functional.silu,
        )

        # Linear
        irreps_mid = irreps_mid.simplify()
        self.irreps_out = self.target_irreps
        self.linear = o3.Linear(
            irreps_mid, self.irreps_out, internal_weights=True, shared_weights=True
        )

        # Selector TensorProduct
        self.skip_tp = o3.FullyConnectedTensorProduct(
            self.node_feats_irreps, self.node_attrs_irreps, self.hidden_irreps
        )
        self.reshape = reshape_irreps(self.irreps_out)

    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sender = edge_index[0]
        receiver = edge_index[1]
        num_nodes = node_feats.shape[0]
        sc = self.skip_tp(node_feats, node_attrs)
        node_feats = self.linear_up(node_feats)
        tp_weights = self.conv_tp_weights(edge_feats)
        mji = self.conv_tp(
            node_feats[sender], edge_attrs, tp_weights
        )  # [n_edges, irreps]
        message = scatter_sum(
            src=mji, index=receiver, dim=0, dim_size=num_nodes
        )  # [n_nodes, irreps]
        message = self.linear(message) / self.avg_num_neighbors
        return (
            self.reshape(message),
            sc,
        )  # [n_nodes, channels, (lmax + 1)**2]


class MACE(torch.nn.Module):
    def __init__(
        self,
        r_max: float,
        num_bessel: int,
        num_polynomial_cutoff: int,
        max_ell: int,
        interaction_cls: Type[InteractionBlock],
        interaction_cls_first: Type[InteractionBlock],
        num_interactions: int,
        num_elements: int,
        hidden_irreps: o3.Irreps,
        MLP_irreps: o3.Irreps,
        atomic_energies: np.ndarray,
        avg_num_neighbors: float,
        atomic_numbers: List[int],
        correlation: Union[int, List[int]],
        gate: Optional[Callable],
        radial_MLP: Optional[List[int]] = None,
        radial_type: Optional[str] = "bessel",
        profile=False
    ):
        super().__init__()
        self.register_buffer(
            "atomic_numbers", torch.tensor(atomic_numbers, dtype=torch.int64)
        )
        self.register_buffer(
            "r_max", torch.tensor(r_max, dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            "num_interactions", torch.tensor(num_interactions, dtype=torch.int64)
        )
        if isinstance(correlation, int):
            correlation = [correlation] * num_interactions
        # Embedding
        node_attr_irreps = o3.Irreps([(num_elements, (0, 1))])
        node_feats_irreps = o3.Irreps([(hidden_irreps.count(o3.Irrep(0, 1)), (0, 1))])
        
        #print ("LinearNodeEmbeddingBlock:", node_attr_irreps, "->", node_feats_irreps)
        self.node_embedding = LinearNodeEmbeddingBlock(
            irreps_in=node_attr_irreps, irreps_out=node_feats_irreps
        )
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
            radial_type=radial_type,
        )
        edge_feats_irreps = o3.Irreps(f"{self.radial_embedding.out_dim}x0e")

        sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        num_features = hidden_irreps.count(o3.Irrep(0, 1))
        interaction_irreps = (sh_irreps * num_features).sort()[0].simplify()
        #self.spherical_harmonics = o3.SphericalHarmonics(
        #    sh_irreps, normalize=True, normalization="component"
        #)
        self.spherical_harmonics = torch.classes.spherical_harmonics.SphericalHarmonics()
        if radial_MLP is None:
            radial_MLP = [64, 64, 64]
        # Interactions and readout
        self.atomic_energies_fn = AtomicEnergiesBlock(atomic_energies)

        inter = interaction_cls_first(
            node_attrs_irreps=node_attr_irreps,
            node_feats_irreps=node_feats_irreps,
            edge_attrs_irreps=sh_irreps,
            edge_feats_irreps=edge_feats_irreps,
            target_irreps=interaction_irreps,
            hidden_irreps=hidden_irreps,
            avg_num_neighbors=avg_num_neighbors,
            radial_MLP=radial_MLP,
        )
        self.interactions = torch.nn.ModuleList([inter])

        # Use the appropriate self connection at the first layer for proper E0
        use_sc_first = False
        if "Residual" in str(interaction_cls_first):
            use_sc_first = True

        node_feats_irreps_out = inter.target_irreps
        prod = EquivariantProductBasisBlock(
            node_feats_irreps=node_feats_irreps_out,
            target_irreps=hidden_irreps,
            correlation=correlation[0],
            num_elements=num_elements,
            use_sc=use_sc_first,
        )
        self.products = torch.nn.ModuleList([prod])

        self.readouts = torch.nn.ModuleList()
        self.readouts.append(LinearReadoutBlock(hidden_irreps))

        for i in range(num_interactions - 1):
            if i == num_interactions - 2:
                hidden_irreps_out = str(
                    hidden_irreps[0]
                )  # Select only scalars for last layer
            else:
                hidden_irreps_out = hidden_irreps
            inter = interaction_cls(
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=hidden_irreps,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=interaction_irreps,
                hidden_irreps=hidden_irreps_out,
                avg_num_neighbors=avg_num_neighbors,
                radial_MLP=radial_MLP,
            )
            self.interactions.append(inter)
            prod = EquivariantProductBasisBlock(
                node_feats_irreps=interaction_irreps,
                target_irreps=hidden_irreps_out,
                correlation=correlation[i + 1],
                num_elements=num_elements,
                use_sc=True,
            )
            self.products.append(prod)
            if i == num_interactions - 2:
                self.readouts.append(
                    NonLinearReadoutBlock(hidden_irreps_out, MLP_irreps, gate)
                )
            else:
                self.readouts.append(LinearReadoutBlock(hidden_irreps))
        
        r,h = np.linspace(1e-12, self.r_max.item() + 1.0, 128, retstep=True)
        r = torch.tensor(r, dtype=torch.float32)
        bessel_j = self.radial_embedding(r.unsqueeze(-1)) 
        
        self.edge_splines  = []
        for i, interaction in enumerate(self.interactions):
            R = interaction.conv_tp_weights(bessel_j)
            self.edge_splines.append(torch.classes.cubic_spline.CubicSpline(r.cuda(), R.cuda()))

        self.profile = profile

    def forward(
        self,
        data: Dict[str, torch.Tensor],
    ) -> Dict[str, Optional[torch.Tensor]]:
        # Setup
        num_graphs = data["ptr"].numel() - 1

        if (self.profile):
            torch.cuda.nvtx.range_push("MACE::atomic_energies")
        # Atomic energies
        node_e0 = self.atomic_energies_fn(data["node_attrs"])
        # node_e0 = self.atomic_energies_fn(data["node_attrs"])
        e0 = scatter_sum( src=node_e0, index=data["batch"], dim=-1, dim_size=num_graphs) # [n_graphs,]
            
        if (self.profile):
            torch.cuda.nvtx.range_pop()
        # Embeddings
        
        if (self.profile):
            torch.cuda.nvtx.range_push("MACE::embeddings")
            
        #print ("node attrs:", data["node_attrs"].shape)
        node_feats = self.node_embedding (data["node_attrs"])
        if (self.profile):
            torch.cuda.nvtx.range_pop()
            
        if (self.profile):
            torch.cuda.nvtx.range_push("MACE::edge vectors")
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data["positions"],
            edge_index=data["edge_index"],
            shifts=data["shifts"],
        )
        if (self.profile):
            torch.cuda.nvtx.range_pop()

        if (self.profile):
            torch.cuda.nvtx.range_push("MACE::sph")
        edge_attrs = self.spherical_harmonics.forward(vectors)
        if (self.profile):
            torch.cuda.nvtx.range_pop()
            
        # Interactions
        energies = [e0]
        node_energies_list = [node_e0]
        node_feats_list = []
        for j, (interaction, product, readout, edge_spline) in enumerate(zip(
            self.interactions, self.products, self.readouts, self.edge_splines)
        ):
            if (self.profile):
                torch.cuda.nvtx.range_push("MACE::edge spline")
            edge_feats = edge_spline.forward(lengths.squeeze(-1))
            if (self.profile):
                torch.cuda.nvtx.range_pop()
            
            if (self.profile):
                torch.cuda.nvtx.range_push("MACE::interaction: {}".format(j))
            node_feats, sc = interaction(
                node_attrs=data["node_attrs"],
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data["edge_index"],
            )
            if (self.profile):
                torch.cuda.nvtx.range_pop()

            if (self.profile):
                torch.cuda.nvtx.range_push("MACE::product: {}".format(j))
            node_feats = product(node_feats=node_feats, sc=sc, node_attrs=data["node_attrs"]
            )
            if (self.profile):
                torch.cuda.nvtx.range_pop()

            if (self.profile):
                torch.cuda.nvtx.range_push("MACE::readout: {}".format(j))
                
            node_feats_list.append(node_feats)
            node_energies = readout(node_feats).squeeze(-1)  # [n_nodes, ]
            energy = scatter_sum(
                src=node_energies,
                index=data["batch"],
                dim=-1,
                dim_size=num_graphs,
            )  # [n_graphs,]
            energies.append(energy)
            node_energies_list.append(node_energies)
            if (self.profile):
                torch.cuda.nvtx.range_pop()

        if (self.profile):
                torch.cuda.nvtx.range_push("MACE:: node sum")
                
        # Concatenate node features
        node_feats_out = torch.cat(node_feats_list, dim=-1)
        
        # Sum over energy contributions
        contributions = torch.stack(energies, dim=-1)
        total_energy = torch.sum(contributions, dim=-1)  # [n_graphs, ]
        node_energy_contributions = torch.stack(node_energies_list, dim=-1)
        node_energy = torch.sum(node_energy_contributions, dim=-1
        )  # [n_nodes, ]

        if (self.profile):
            torch.cuda.nvtx.range_pop()
        # Outputs
        
        return total_energy

def run_test():
    from ase import build
    from mace import modules

    size = 8
    cutoff = 4.0

    # build very large diamond structure
    atoms = build.bulk("C", "diamond", a=3.567, cubic=True)
    atoms_list = [atoms.repeat((size, size, size))]
    print("Number of atoms", len(atoms_list[0]))

    configs = [data.config_from_atoms(atoms) for atoms in atoms_list]

    z_table = tools.AtomicNumberTable(
        [int(z) for z in np.unique(configs[0].atomic_numbers)]
    )

    nnodes = configs[0].atomic_numbers.shape[0]

    data_loader = torch_geometric.dataloader.DataLoader(
        dataset=[
            data.AtomicData.from_config(config, z_table=z_table, cutoff=cutoff)
            for config in configs
        ],
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )
    atomic_energies = np.array([1.0], dtype=float)

    batch = next(iter(data_loader)).to("cuda")
    
    print ('edge_index: ', batch['edge_index'].shape)

    model_config = dict(
        r_max=cutoff,
        num_bessel=8,
        num_polynomial_cutoff=6,
        max_ell=3,
        interaction_cls=RealAgnosticResidualInteractionBlock,
        interaction_cls_first=RealAgnosticResidualInteractionBlock,
        num_interactions=2,
        num_elements=1,
        hidden_irreps=o3.Irreps("96x0e"),
        MLP_irreps=o3.Irreps("16x0e"),
        gate=torch.nn.functional.silu,
        atomic_energies=atomic_energies,
        avg_num_neighbors=45,
        atomic_numbers=z_table.zs,
        correlation=3,
        radial_type="bessel",
        profile=True
    )
    model = MACE(**model_config).to("cuda")
    model = optimize_cuda_mace(model)

    #model = torch.jit.script(model)
    
    warmup = 10
    for i in range(warmup):
        out = model(batch)
    torch.cuda.synchronize()

    batch['positions'].requires_grad_(True)
    batch["node_attrs"].requires_grad_(True)
    
    warmup = 1000
    start = time()
    for i in range(warmup):
        out = model(batch)
        os = out.sum()
        os.backward()
        torch.cuda.synchronize()
    torch.cuda.synchronize()
    end = time()
    print (end - start)
    
    batch['positions'].requires_grad_(True)
    batch["node_attrs"].requires_grad_(True)
    
    torch.cuda.cudart().cudaProfilerStart()
    for i in range (5):
        torch.cuda.nvtx.range_push("MACE::forward {}".format(i))
        out = model(batch)
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
        
        torch.cuda.nvtx.range_push("MACE::backward {}".format(i))
        os = out.sum()
        os.backward()
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
    torch.cuda.cudart().cudaProfilerStop()



def run_benchmark():
    from ase import build
    from mace import modules
    
    sizes = (
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9
        )
    times = []
    natoms = []
    for size in sizes:
        cutoff = 4.0

        # build very large diamond structure
        atoms = build.bulk("C", "diamond", a=3.567, cubic=True)
        atoms_list = [atoms.repeat((size, size, size))]
        print("Number of atoms", len(atoms_list[0]))

        natoms.append(len(atoms_list[0]))
        
        configs = [data.config_from_atoms(atoms) for atoms in atoms_list]

        z_table = tools.AtomicNumberTable(
            [int(z) for z in np.unique(configs[0].atomic_numbers)]
        )

        nnodes = configs[0].atomic_numbers.shape[0]

        data_loader = torch_geometric.dataloader.DataLoader(
            dataset=[
                data.AtomicData.from_config(config, z_table=z_table, cutoff=cutoff)
                for config in configs
            ],
            batch_size=1,
            shuffle=False,
            drop_last=False,
        )
        atomic_energies = np.array([1.0], dtype=float)

        batch = next(iter(data_loader)).to("cuda")
        
        print ('edge_index: ', batch['edge_index'].shape)

        model_config = dict(
            r_max=cutoff,
            num_bessel=8,
            num_polynomial_cutoff=6,
            max_ell=3,
            interaction_cls=RealAgnosticResidualInteractionBlock,
            interaction_cls_first=RealAgnosticResidualInteractionBlock,
            num_interactions=2,
            num_elements=1,
            hidden_irreps=o3.Irreps("96x0e"),
            MLP_irreps=o3.Irreps("16x0e"),
            gate=torch.nn.functional.silu,
            atomic_energies=atomic_energies,
            avg_num_neighbors=45,
            atomic_numbers=z_table.zs,
            correlation=3,
            radial_type="bessel",
            profile=True
        )
        model = MACE(**model_config).to("cuda")
        model = optimize_cuda_mace(model)

        #model = torch.jit.script(model)
        
        warmup = 10
        for i in range(warmup):
            out = model(batch)
        torch.cuda.synchronize()

        batch['positions'].requires_grad_(True)
        batch["node_attrs"].requires_grad_(True)
        
        warmup = 1000
        start = time()
        for i in range(warmup):
            out = model(batch)
            os = out.sum()
            os.backward()
            torch.cuda.synchronize()
        torch.cuda.synchronize()
        end = time()
        print (end - start)
        
        print (times.append(end - start))

    print(natoms)
    print([f"{t:.2f}" for t in times])
    formatted_times = [f"{((1000.0 / t) * 86400) / 1000000.0:.2f}" for t in times]
    print(formatted_times)
    
def main(args=None):
    run_benchmark()
    #run_test()


if __name__ == "__main__":
    main()
