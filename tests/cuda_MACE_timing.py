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


timings = {}
class_occurence = {}
run_timeit = False


def get_name(obj):
    if hasattr(obj, "__name__"):
        return obj.__name__
    elif hasattr(obj, "__class__") and hasattr(obj.__class__, "__name__"):
        return obj.__class__.__name__
    else:
        raise ValueError("Object has no name attribute.")


def timeit(fn, *args, **kwargs):
    global run_timeit

    if not run_timeit:
        return fn(*args, **kwargs)
    else:
        start = time()
        out = fn(*args, **kwargs)
        torch.cuda.synchronize()
        end = time()
        if hasattr(fn, "__class__"):
            if not fn.__class__ in class_occurence:
                timings[fn.__class__.__name__] = {}
                class_occurence[fn.__class__] = 0
            else:
                class_occurence[fn.__class__] += 1

            fn_name = get_name(fn)
            timings[fn.__class__.__name__][
                fn_name + str(class_occurence[fn.__class__])
            ] = (end - start)
        return out


def optimize_cuda_mace(model) -> None:
    """
    Optimize the MACE model for CUDA inference.
    """
    for param in model.parameters():
        param.requires_grad = False
    dtype = get_model_dtype(model)
    n_layers = int(model.num_interactions)
    sh_irreps = o3.Irreps.spherical_harmonics(3)
    spherical_harmonics = SphericalHarmonics(
        sh_irreps=sh_irreps,
        normalize=True,
        normalization="component",
        backend="opt",
    )
    model.spherical_harmonics = spherical_harmonics
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
    global run_timeit
    sender = edge_index[0]
    receiver = edge_index[1]
    num_nodes = torch.tensor(node_feats.shape[0])
    #if run_timeit:
        #print (node_feats.shape, node_attrs.shape)
    start = time()
    sc = self.skip_tp(node_feats, node_attrs)
    torch.cuda.synchronize()
    end = time()
    if run_timeit:
        print("skip_tp", (end - start) * 1000, "ms")
    
    start = time()
    node_feats = self.linear_up(node_feats)
    torch.cuda.synchronize()
    end = time()
    if run_timeit:
        print("linear_up", (end - start) * 1000, "ms")
    
    #print (edge_feats.shape)
    start = time()
    tp_weights = self.conv_tp_weights(edge_feats)
    torch.cuda.synchronize()
    end = time()

    if run_timeit:
        print("conv_tp_weights", (end - start) * 1000, "ms")

    #tp_weights = torch.randn(tp_weights.shape[0], 4, 96, device='cuda', dtype=torch.float32, requires_grad=True)
    #torch.cuda.synchronize()
    torch.cuda.synchronize()
    
    start = time()
    message = self.tp.forward(
        node_feats,
        edge_attrs,
        tp_weights.view(tp_weights.shape[0], -1, node_feats.shape[-1]),
        sender.int(),
        receiver.int(),
        num_nodes,
    )
    torch.cuda.synchronize()
    end = time()

    if run_timeit:
        print("tp: ", (end - start) * 1000, "ms")

    start = time()
    message = self.linear(message) / self.avg_num_neighbors
    torch.cuda.synchronize()
    end = time()

    if run_timeit:
        print("message", (end - start) * 1000, "ms")
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
    node_feats = self.linear_up(node_feats)
    tp_weights = self.conv_tp_weights(edge_feats)
    message = self.tp.forward(
        node_feats,
        edge_attrs,
        tp_weights.view(tp_weights.shape[0], -1, node_feats.shape[-1]),
        sender.int(),
        receiver.int(),
        num_nodes,
    )

    message = self.linear(message) / self.avg_num_neighbors
    message = self.skip_tp(message, node_attrs)
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
        
        print ([input_dim] + self.radial_MLP + [self.conv_tp.weight_numel])

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
        self.spherical_harmonics = o3.SphericalHarmonics(
            sh_irreps, normalize=True, normalization="component"
        )
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

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        training: bool = False,
        compute_force: bool = True,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_displacement: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:
        # Setup
        data["node_attrs"].requires_grad_(True)
        data["positions"].requires_grad_(True)
        num_graphs = data["ptr"].numel() - 1
        displacement = torch.empty(
            (num_graphs, 3, 3),
            dtype=data["positions"].dtype,
            device=data["positions"].device,
        )
        if compute_virials or compute_stress or compute_displacement:
            (
                data["positions"],
                data["shifts"],
                displacement,
            ) = timeit(
                get_symmetric_displacement,
                positions=data["positions"],
                unit_shifts=data["unit_shifts"],
                cell=data["cell"],
                edge_index=data["edge_index"],
                num_graphs=num_graphs,
                batch=data["batch"],
            )

        # Atomic energies
        node_e0 = timeit(self.atomic_energies_fn, data["node_attrs"])
        # node_e0 = self.atomic_energies_fn(data["node_attrs"])
        e0 = timeit(
            scatter_sum, src=node_e0, index=data["batch"], dim=-1, dim_size=num_graphs
        )  # [n_graphs,]

        # Embeddings
        node_feats = timeit(self.node_embedding, data["node_attrs"])
        vectors, lengths = timeit(
            get_edge_vectors_and_lengths,
            positions=data["positions"],
            edge_index=data["edge_index"],
            shifts=data["shifts"],
        )

        edge_attrs = timeit(self.spherical_harmonics, vectors)
        edge_feats = timeit(self.radial_embedding, lengths)

        # Interactions
        energies = [e0]
        node_energies_list = [node_e0]
        node_feats_list = []
        for interaction, product, readout in zip(
            self.interactions, self.products, self.readouts
        ):
            node_feats, sc = timeit(
                interaction,
                node_attrs=data["node_attrs"],
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data["edge_index"],
            )

            node_feats = timeit(
                product, node_feats=node_feats, sc=sc, node_attrs=data["node_attrs"]
            )

            node_feats_list.append(node_feats)
            node_energies = timeit(readout, node_feats).squeeze(-1)  # [n_nodes, ]
            energy = timeit(
                scatter_sum,
                src=node_energies,
                index=data["batch"],
                dim=-1,
                dim_size=num_graphs,
            )  # [n_graphs,]
            energies.append(energy)
            node_energies_list.append(node_energies)

        # Concatenate node features
        node_feats_out = torch.cat(node_feats_list, dim=-1)

        # Sum over energy contributions
        contributions = timeit(torch.stack, energies, dim=-1)
        total_energy = timeit(torch.sum, contributions, dim=-1)  # [n_graphs, ]
        node_energy_contributions = timeit(torch.stack, node_energies_list, dim=-1)
        node_energy = timeit(
            torch.sum, node_energy_contributions, dim=-1
        )  # [n_nodes, ]

        # Outputs
        forces, virials, stress = timeit(
            get_outputs,
            energy=total_energy,
            positions=data["positions"],
            displacement=displacement,
            cell=data["cell"],
            training=training,
            compute_force=compute_force,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
        )

        return {
            "energy": total_energy,
            "node_energy": node_energy,
            "contributions": contributions,
            "forces": forces,
            "virials": virials,
            "stress": stress,
            "displacement": displacement,
            "node_feats": node_feats_out,
        }


def print_recursive(dictionary, indent=0):
    for key, value in dictionary.items():
        print("\t" * indent + f"{key}:", end=" ")
        if isinstance(value, dict):
            print()
            print_recursive(value, indent + 1)
        else:
            print(value * 1000, "ms")


def recursive_sum(dictionary):
    total = 0.0
    for value in dictionary.values():
        if isinstance(value, dict):
            total += recursive_sum(value)
        else:
            total += value

    return total


def run_test():
    global run_timeit
    from ase import build
    from mace import modules

    size = 5
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
        avg_num_neighbors=70,
        atomic_numbers=z_table.zs,
        correlation=3,
        radial_type="bessel",
    )
    model = MACE(**model_config).to("cuda")
    model = optimize_cuda_mace(model)

    #model = torch.jit.script(model)
    
    warmup = 50
    for i in range(warmup):
        out = model(batch)
    torch.cuda.synchronize()

    run_timeit = True
    start = time()
    out = model(batch, compute_force=False)
    torch.cuda.synchronize()
    end = time()

    print((end - start) * 1000, "ms")

    print_recursive(timings)
    print(recursive_sum(timings) * 1000, "ms")


def main(args=None):
    run_test()


if __name__ == "__main__":
    main()
