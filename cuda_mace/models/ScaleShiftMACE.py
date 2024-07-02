from typing import Any, Callable, Dict, List, Optional, Type, Union, Tuple
from copy import deepcopy
import ase
import numpy as np
import torch
from mace.tools import torch_geometric
from mace import data, tools
from e3nn import o3, nn
from mace import modules
from time import time
import inspect
from math import sqrt

from mace.modules.blocks import SphericalHarmonics

from mace.tools.scatter import scatter_sum

from mace.modules.utils import (
    get_edge_vectors_and_lengths,
    get_outputs,
    get_symmetric_displacement,
)
from cuda_mace.ops.invariant_message_passing import InvariantMessagePassingTP
from cuda_mace.ops.linear import Linear, ElementalLinear
from cuda_mace.ops.symmetric_contraction import SymmetricContraction as CUDAContraction


class SymmetricContractionWrapper(torch.nn.Module):
    def __init__(self, symmetric_contractions):
        super().__init__()
        self.symmetric_contractions = symmetric_contractions

    def forward(self, x, y):
        y = y.argmax(dim=-1).int()
        out = self.symmetric_contractions(x, y).squeeze()
        return out


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
    # print("elementlinear", skip_tp)
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


class InvariantInteraction(torch.nn.Module):

    def __init__(self, mace_model, profile=False):
        super().__init__()
        self.linear_up = linear_matmul(mace_model.interactions[0].linear_up.float())
        self.linear = linear_to_cuda(mace_model.interactions[0].linear.float())
        self.tp = InvariantMessagePassingTP()
        self.skip_tp = element_linear_to_cuda(
            mace_model.interactions[0].skip_tp.float())
        self.avg_num_neighbors = mace_model.interactions[0].avg_num_neighbors
        self.profile = profile

    def forward(
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
            torch.cuda.nvtx.range_push("InvariantInteraction::linear up")
        node_feats = self.linear_up(node_feats)

        if (self.profile):
            torch.cuda.nvtx.range_pop()

        if (self.profile):
            torch.cuda.nvtx.range_push("InvariantInteraction::inv tp")

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
            torch.cuda.nvtx.range_push("InvariantInteraction::message linear")

        message = self.linear(message) / self.avg_num_neighbors

        if (self.profile):
            torch.cuda.nvtx.range_pop()

        if (self.profile):
            torch.cuda.nvtx.range_push("InvariantInteraction::skip tp")

        message = self.skip_tp(message, node_attrs)

        if (self.profile):
            torch.cuda.nvtx.range_pop()

        return (
            message,
            None,
        )


class InvariantResidualInteraction(torch.nn.Module):

    def __init__(self, mace_model, profile=False):
        super().__init__()

        self.linear_up = linear_matmul(mace_model.interactions[1].linear_up.float())
        self.linear = linear_to_cuda(mace_model.interactions[1].linear.float())
        self.tp = InvariantMessagePassingTP()
        self.skip_tp = mace_model.interactions[1].skip_tp.float()
        self.avg_num_neighbors = mace_model.interactions[1].avg_num_neighbors
        self.profile = profile

    def forward(
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
            torch.cuda.nvtx.range_push("InvariantInteraction::skip tp")
        sc = self.skip_tp(node_feats, node_attrs)
        if (self.profile):
            torch.cuda.nvtx.range_pop()

        if (self.profile):
            torch.cuda.nvtx.range_push("InvariantInteraction::linear up")
        node_feats = self.linear_up(node_feats)

        if (self.profile):
            torch.cuda.nvtx.range_pop()

        if (self.profile):
            torch.cuda.nvtx.range_push("InvariantInteraction::inv tp")

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
            torch.cuda.nvtx.range_push("InvariantInteraction::message linear")

        message = self.linear(message) / self.avg_num_neighbors

        if (self.profile):
            torch.cuda.nvtx.range_pop()

        return (
            message,
            sc,
        )


class OptimizedScaleShiftInvariantMACE(torch.nn.Module):
    def __init__(
        self,
        mace_model: torch.nn.Module,
        profile=False
    ):
        super().__init__()
        self.register_buffer(
            "atomic_numbers", deepcopy(mace_model.atomic_numbers)
        )
        self.register_buffer(
            "r_max", deepcopy(mace_model.r_max)
        )
        self.register_buffer(
            "num_interactions", deepcopy(mace_model.num_interactions)
        )

        self.node_embedding = deepcopy(mace_model.node_embedding)
        self.radial_embedding = deepcopy(mace_model.radial_embedding)

        self.spherical_harmonics = torch.classes.spherical_harmonics.SphericalHarmonics()

        # Interactions and readout
        self.atomic_energies_fn = deepcopy(mace_model.atomic_energies_fn)

        self.interactions = torch.nn.ModuleList([InvariantInteraction(
            mace_model, profile), InvariantResidualInteraction(mace_model, profile)])
        self.products = deepcopy(mace_model.products)
        
        self.readouts = []
        
        for i in range (len(mace_model.readouts)):
            self.readouts.append(mace_model.readouts[i].double())

        self.readouts = torch.nn.ModuleList(self.readouts)
        
        for i in range(mace_model.num_interactions):
            symm_contract = mace_model.products[i].symmetric_contractions
            all_weights = {}
            for j in range(len(symm_contract.contractions)):
                all_weights[str(j)] = {}
                all_weights[str(j)][3] = (
                    symm_contract.contractions[j].weights_max.detach(
                    ).clone().type(torch.float32)
                )
                all_weights[str(j)][2] = (
                    symm_contract.contractions[j].weights[0].detach(
                    ).clone().type(torch.float32)
                )
                all_weights[str(j)][1] = (
                    symm_contract.contractions[j].weights[1].detach(
                    ).clone().type(torch.float32)
                )

            irreps_in = o3.Irreps(
                mace_model.products[i].symmetric_contractions.irreps_in)
            coupling_irreps = o3.Irreps([irrep.ir for irrep in irreps_in])
            irreps_out = o3.Irreps(
                mace_model.products[i].symmetric_contractions.irreps_out)

            symmetric_contractions = CUDAContraction(
                coupling_irreps,
                irreps_out,
                all_weights,
                nthreadX=32,
                nthreadY=4,
                nthreadZ=1,
                dtype=torch.float32,
            )
            self.products[i].symmetric_contractions = SymmetricContractionWrapper(
                symmetric_contractions
            )
            self.products[i].linear = linear_matmul(
                deepcopy(mace_model.products[i].linear.float()))

        r, h = np.linspace(1e-12, self.r_max.item() + 1.0, 4096, retstep=True)
        r = torch.tensor(r, dtype=torch.float64).to("cuda")
        bessel_j = self.radial_embedding(r.unsqueeze(-1))
    
        self.edge_splines = []
        for i, interaction in enumerate(mace_model.interactions):
            R = interaction.conv_tp_weights(bessel_j)
            spline = torch.classes.cubic_spline.CubicSpline(
                r.cuda(), R.cuda(), h, self.r_max.item())
            self.edge_splines.append(spline)

        self.scale_shift = deepcopy(mace_model.scale_shift.double())
        
        self.profile = profile
        self.orig_model = mace_model

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
        if (self.profile):
            torch.cuda.nvtx.range_push("MACE::forward")

        data["positions"].requires_grad_(True)
        data["node_attrs"].requires_grad_(True)
        num_graphs = data["ptr"].numel() - 1
        displacement = torch.zeros(
            (num_graphs, 3, 3),
            dtype=data["positions"].dtype,
            device=data["positions"].device,
        )
        if compute_virials or compute_stress or compute_displacement:
            (
                data["positions"],
                data["shifts"],
                displacement,
            ) = get_symmetric_displacement(
                positions=data["positions"],
                unit_shifts=data["unit_shifts"],
                cell=data["cell"],
                edge_index=data["edge_index"],
                num_graphs=num_graphs,
                batch=data["batch"],
            )

        if (self.profile):
            torch.cuda.nvtx.range_push("MACE::atomic_energies")
        # Atomic energies
        node_e0 = self.atomic_energies_fn(data["node_attrs"].double())
        # node_e0 = self.atomic_energies_fn(data["node_attrs"])
        e0 = scatter_sum(src=node_e0, index=data["batch"], dim=-1, dim_size=num_graphs)  # [n_graphs,]

        if (self.profile):
            torch.cuda.nvtx.range_pop()
        # Embeddings

        if (self.profile):
            torch.cuda.nvtx.range_push("MACE::embeddings")
        # print ("node attrs:", data["node_attrs"].shape)
        node_feats = self.node_embedding(data["node_attrs"].double())

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

        node_es_list  = []
        node_feats_list = []
        for j, (interaction, product, readout, edge_spline) in enumerate(zip(
            self.interactions, self.products, self.readouts, self.edge_splines)
        ):
            if (self.profile):
                torch.cuda.nvtx.range_push("MACE::edge spline")

            edge_feats = edge_spline.forward(lengths.squeeze(-1).double()).float()

            if (self.profile):
                torch.cuda.nvtx.range_pop()

            if (self.profile):
                torch.cuda.nvtx.range_push("MACE::interaction: {}".format(j))

            node_feats, sc = interaction(
                node_attrs=data["node_attrs"],
                node_feats=node_feats.float(),
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data["edge_index"],
            )

            if (self.profile):
                torch.cuda.nvtx.range_pop()

            if (self.profile):
                torch.cuda.nvtx.range_push("MACE::product: {}".format(j))

            if (sc is not None):
                print(sc.shape)
            node_feats = product(node_feats=node_feats, sc=sc, node_attrs=data["node_attrs"]
                                 )

            if (self.profile):
                torch.cuda.nvtx.range_pop()

            if (self.profile):
                torch.cuda.nvtx.range_push("MACE::readout: {}".format(j))

            node_feats_list.append(node_feats)
            node_energies = readout(node_feats.double()).squeeze(-1)  # [n_nodes, ]
            
            node_es_list.append(node_energies)
        
        node_inter_es = torch.sum(torch.stack(node_es_list, dim=0), dim=0)  # [n_nodes, ]
        node_inter_es = self.scale_shift(node_inter_es)
        inter_e = scatter_sum(
            src=node_inter_es.double(), index=data["batch"], dim=-1, dim_size=num_graphs
        )  # [n_graphs,]
        
        
        # Outputs
        total_energy = e0 + inter_e

        node_energy = node_e0 + node_inter_es

        forces, virials, stress = get_outputs(
            energy=inter_e,
            positions=data["positions"],
            displacement=displacement,
            cell=data["cell"],
            training=training,
            compute_force=compute_force,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
        )

        output = {
            "energy": total_energy,
            "node_energy": node_energy,
            "interaction_energy": inter_e,
            "forces": forces,
            "virials": virials,
            "stress": stress,
            "displacement": displacement,
        }

        if (self.profile):
            torch.cuda.nvtx.range_pop()

        return output