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

import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity

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

import math

from mace_ops.ops.invariant_message_passing import InvariantMessagePassingTP

def reference(X, Y, radial, sender_list, receiver_list, nnodes):
    output = torch.zeros(nnodes, Y.shape[1], X.shape[1], device=X.device, dtype=X.dtype)

    X_sender = X[sender_list]
    
    for i in range(Y.shape[1]):
        out = X_sender * Y[:, i][:, None] * radial[:, int(math.sqrt(i)), :]

        output[:, i, :].index_add_(0, receiver_list, out)

    return output

def accuracy():
    global run_timeit
    from ase import build
    from mace import modules

    size = 2
    cutoff = 2.0

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

    batch = next(iter(data_loader)).to("cuda")
    #add small pertubation to remove symmetries
    batch['positions'] = batch['positions'] + 0.1*torch.randn(batch['positions'].shape, dtype=batch['positions'].dtype, device=batch['positions'].device)

    vectors, lengths = get_edge_vectors_and_lengths(
    positions=batch["positions"],
    edge_index=batch["edge_index"],
    shifts=batch["shifts"])

    sh_irreps = o3.Irreps.spherical_harmonics(3)
    spherical_harmonics = SphericalHarmonics(
        sh_irreps=sh_irreps,
        normalize=True,
        normalization="component",
        backend="opt",
    )

    radial_embedding = RadialEmbeddingBlock(
            r_max=4.5,
            num_bessel=8,
            num_polynomial_cutoff=6,
            radial_type="bessel",
    ).to('cuda')


    num_elements=1
    hidden_irreps=o3.Irreps("96x0e")
    node_attr_irreps = o3.Irreps([(num_elements, (0, 1))])
    node_feats_irreps = o3.Irreps([(hidden_irreps.count(o3.Irrep(0, 1)), (0, 1))])
    node_embedding = LinearNodeEmbeddingBlock(
            irreps_in=node_attr_irreps, irreps_out=node_feats_irreps
        ).to('cuda')

    edge_feats_irreps = o3.Irreps(f"{radial_embedding.out_dim}x0e")
    input_dim = edge_feats_irreps.num_irreps
    radial_MLP = [64, 64, 64]
    num_features = hidden_irreps.count(o3.Irrep(0, 1))

    interaction_irreps = (sh_irreps * num_features).sort()[0].simplify()

    irreps_mid, instructions = tp_out_irreps_with_instructions(
            node_feats_irreps,
            sh_irreps,
            interaction_irreps,
        )

    conv_tp = o3.TensorProduct(
            node_feats_irreps,
            sh_irreps,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
        )

    conv_tp_weights = nn.FullyConnectedNet(
            [input_dim] + radial_MLP + [conv_tp.weight_numel],
            torch.nn.functional.silu,
    ).to('cuda')

    edge_attrs = spherical_harmonics(vectors)
    edge_feats = radial_embedding(lengths)
    node_feats = node_embedding(batch["node_attrs"])
    radial_feats = conv_tp_weights(edge_feats)
    #radial_feats = radial_feats.view(radial_feats.shape[0], -1, node_feats.shape[-1])
    edge_index = batch['edge_index']

    sender = edge_index[0]
    receiver = edge_index[1]
    
        
    print (sender)
    print (receiver)
    sort_sender_idx = torch.argsort(sender)
    sorted_sender  = sender[sort_sender_idx]
    sorted_receiver = receiver[sort_sender_idx]
    
    for i in range (sender.shape[0]):
        print (i, sender[i].item(), receiver[i].item(), sorted_sender[i].item(), sorted_receiver[i].item())

    edge_attrs_cuda = edge_attrs.clone().detach().requires_grad_(True)
    node_feats_cuda = node_feats.clone().detach().requires_grad_(True)
    radial_feats_cuda =  radial_feats.view(radial_feats.shape[0], -1, node_feats.shape[-1]).clone().detach().requires_grad_(True)

    edge_attrs_ref = edge_attrs.clone().detach().requires_grad_(True)
    node_feats_ref = node_feats.clone().detach().requires_grad_(True)
    radial_feats_ref =  radial_feats.view(radial_feats.shape[0], -1, node_feats.shape[-1]).clone().detach().requires_grad_(True)

    edge_attrs_e3nn = edge_attrs.clone().detach().requires_grad_(True)
    node_feats_e3nn = node_feats.clone().detach().requires_grad_(True)
    radial_feats_e3nn = radial_feats.clone().detach().requires_grad_(True)

    print (edge_feats.shape)
    print (node_feats.shape)
    print (radial_feats.shape)

    ref_out = reference(node_feats_ref, edge_attrs_ref, radial_feats_ref, sender, receiver, nnodes)

    (ref_out * 2.0).sum().backward()

    print (ref_out.shape)
    
    tp =  InvariantMessagePassingTP()

    cuda_out = tp.forward(
        node_feats_cuda,
        edge_attrs_cuda.transpose(-1, -2),
        radial_feats_cuda,
        sender.int(),
        receiver.int(),
        nnodes,
    )

    print (torch.abs(ref_out - cuda_out).max())

    (cuda_out * 2.0).sum().backward()
    
    print ("--Ref MAEs--")
    print (torch.abs(node_feats_ref.grad - node_feats_cuda.grad).max())
    print (torch.abs(edge_attrs_ref.grad - edge_attrs_cuda.grad).max())
    print (torch.abs(radial_feats_ref.grad - radial_feats_cuda.grad).max())

    mji = conv_tp(node_feats_e3nn[sender], edge_attrs_e3nn, radial_feats_e3nn)

    message = scatter_sum(src=mji, index=receiver.long(), dim=0, dim_size=node_feats.shape[0])
    
    (message * 2.0).sum().backward()

    print ("--E3NN MAEs--")

    print (torch.abs(node_feats_e3nn.grad - node_feats_cuda.grad).max())
    print (torch.abs(edge_attrs_e3nn.grad - edge_attrs_cuda.grad).max())
    print (torch.abs(radial_feats_e3nn.grad.view(radial_feats.shape[0], -1, node_feats.shape[-1]) - radial_feats_cuda.grad).max())



def benchmark(args):
    from ase import build

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

    batch = next(iter(data_loader)).to("cuda")
    #add small pertubation to remove symmetries
    batch['positions'] = batch['positions'] + 0.1*torch.randn(batch['positions'].shape, dtype=batch['positions'].dtype, device=batch['positions'].device)

    vectors, lengths = get_edge_vectors_and_lengths(
    positions=batch["positions"],
    edge_index=batch["edge_index"],
    shifts=batch["shifts"])

    sh_irreps = o3.Irreps.spherical_harmonics(3)
    spherical_harmonics = SphericalHarmonics(
        sh_irreps=sh_irreps,
        normalize=True,
        normalization="component",
        backend="opt",
    )

    radial_embedding = RadialEmbeddingBlock(
            r_max=4.5,
            num_bessel=8,
            num_polynomial_cutoff=6,
            radial_type="bessel",
    ).to('cuda')


    num_elements=1
    hidden_irreps=o3.Irreps("96x0e")
    node_attr_irreps = o3.Irreps([(num_elements, (0, 1))])
    node_feats_irreps = o3.Irreps([(hidden_irreps.count(o3.Irrep(0, 1)), (0, 1))])
    node_embedding = LinearNodeEmbeddingBlock(
            irreps_in=node_attr_irreps, irreps_out=node_feats_irreps
        ).to('cuda')

    edge_feats_irreps = o3.Irreps(f"{radial_embedding.out_dim}x0e")
    input_dim = edge_feats_irreps.num_irreps
    radial_MLP = [64, 64, 64]
    num_features = hidden_irreps.count(o3.Irrep(0, 1))

    interaction_irreps = (sh_irreps * num_features).sort()[0].simplify()

    irreps_mid, instructions = tp_out_irreps_with_instructions(
            node_feats_irreps,
            sh_irreps,
            interaction_irreps,
        )

    conv_tp = o3.TensorProduct(
            node_feats_irreps,
            sh_irreps,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
        )

    conv_tp_weights = nn.FullyConnectedNet(
            [input_dim] + radial_MLP + [conv_tp.weight_numel],
            torch.nn.functional.silu,
    ).to('cuda')

    edge_attrs = spherical_harmonics(vectors)
    edge_feats = radial_embedding(lengths)
    node_feats = node_embedding(batch["node_attrs"])
    radial_feats = conv_tp_weights(edge_feats)
    #radial_feats = radial_feats.view(radial_feats.shape[0], -1, node_feats.shape[-1])
    edge_index = batch['edge_index']

    sender = edge_index[0]
    receiver = edge_index[1]
    
    if (args.grad):
        edge_attrs_cuda = edge_attrs.clone().detach().requires_grad_(True)
        node_feats_cuda = node_feats.clone().detach().requires_grad_(True)
        radial_feats_cuda =  radial_feats.reshape(radial_feats.shape[0], -1, node_feats.shape[-1]).clone().detach().contiguous().requires_grad_(True)
    else:
        edge_attrs_cuda = edge_attrs.clone().detach().requires_grad_(False)
        node_feats_cuda = node_feats.clone().detach().requires_grad_(False)
        radial_feats_cuda =  radial_feats.reshape(radial_feats.shape[0], -1, node_feats.shape[-1]).clone().detach().contiguous().requires_grad_(False)

    tp =  InvariantMessagePassingTP()

    sender = sender.int()
    receiver = receiver.int()
    
    ## torch::Tensor sorted_sender_idx = torch::argsort(sender_list).to(torch::kInt32);
    ##torch::Tensor first_occurences_node = calculate_first_occurences_gpu_with_sort(sender_list, X.size(0), 64, sorted_sender_idx);

    nb_iters = 50
    warmup_iters = 30

    for i in range (nb_iters):
        start = time()
        if i == warmup_iters: torch.cuda.cudart().cudaProfilerStart()
        if i >= warmup_iters: torch.cuda.nvtx.range_push("iteration{}".format(i))
        if i >= warmup_iters: torch.cuda.nvtx.range_push("forward")
        cuda_out = tp.forward(
            node_feats_cuda,
            edge_attrs_cuda,
            radial_feats_cuda,
            sender,
            receiver,
            nnodes,
        )
        if i >= warmup_iters: torch.cuda.nvtx.range_pop()

        if (args.grad):
            os = cuda_out.sum()
            
            if i >= warmup_iters: torch.cuda.nvtx.range_push("backward")
            os.backward()
            if i >= warmup_iters: torch.cuda.nvtx.range_pop()
            
        if i >= warmup_iters: torch.cuda.nvtx.range_pop()     
        end = time()
        
        print ((end - start) * 1000)       
    torch.cuda.cudart().cudaProfilerStop()

def build_parser():
    """
    Create a parser for the command line tool.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Test Inv TP"
    )
    
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="test speed",
        default=False,
    )
    parser.add_argument(
        "--accuracy",
        action="store_true",
        help="test accuracy",
        default=False,
    )
    
    parser.add_argument(
        "--grad",
        action="store_true",
        help="test gradients",
        default=False,
    )
    
    return parser

def main(args=None):
    parser = build_parser()
    args = parser.parse_args(args)
    
    print (args)
    
    if (args.accuracy):
        accuracy()
    elif (args.benchmark):
        benchmark(args)


if __name__ == "__main__":
    main()
