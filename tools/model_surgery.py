from typing import Dict, List, Optional, Type
import torch
from mace.tools import torch_geometric
from mace import data, tools
from time import time

from copy import deepcopy
import nvtx

from mace.modules.utils import (
    get_edge_vectors_and_lengths,
)

from cuda_mace.models import OptimizedInvariantMACE


def build_parser():
    """
    Create a parser for the command line tool.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Optimize a MACE model for CUDA inference."
    )

    parser.add_argument("--model", type=str, help="Path to the MACE model.")

    parser.add_argument(
        "--output",
        type=str,
        default="optimized_model.model",
        help="Path to the output file.",
    )

    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Benchmark the optimized model.",
        default=False,
    )
    parser.add_argument(
        "--accuracy",
        action="store_true",
        help="Benchmark the optimized model.",
        default=False,
    )

    parser.add_argument(
        "--size",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--compare",
        type=int,
        default=0,
    )
    return parser


def accuracy(
    model,
    model_opt,
    size=4,
) -> None:
    from ase import build
    # build very large diamond structure
    atoms = build.bulk("C", "diamond", a=3.567, cubic=True)
    atoms_list = [atoms.repeat((size, size, size))]
    print("Number of atoms", len(atoms_list[0]))

    configs = [data.config_from_atoms(atoms) for atoms in atoms_list]

    z_table = tools.AtomicNumberTable([int(z) for z in model.atomic_numbers])

    data_loader = torch_geometric.dataloader.DataLoader(
        dataset=[
            data.AtomicData.from_config(
                config, z_table=z_table, cutoff=model.r_max.item()
            )
            for config in configs
        ],
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )
    batch = next(iter(data_loader)).to("cuda")
    print("num edges", batch.edge_index.shape)

    for batch in data_loader:
        batch = batch.to("cuda")
        batch['positions'] = batch['positions'] + 0.05*torch.randn(
            batch['positions'].shape, dtype=batch['positions'].dtype, device=batch['positions'].device)

        print("num nodes", batch.num_nodes)
        batch2 = batch.clone()

        # make a copy of the model
        batch['positions'].requires_grad_(True)
        batch["node_attrs"].requires_grad_(True)
        output_opt = model_opt(
            batch.to_dict(), training=False, compute_force=True)

        model = model.double()
        model.atomic_energies_fn.atomic_energies = (
            model.atomic_energies_fn.atomic_energies.double()
        )
        batch2.positions = batch2.positions.double()
        batch2.node_attrs = batch2.node_attrs.double()
        batch2.shifts = batch2.shifts.double()

        batch2['positions'].requires_grad_(True)
        batch2["node_attrs"].requires_grad_(True)

        output_org = model(batch2.to_dict(),
                           training=False, compute_force=True)

        model_f32 = model.float()
        output_f32 = model_f32(batch.to_dict(),
                               training=False, compute_force=True)

        print("energy org", output_org["energy"])
        print("energy opt", output_opt["energy"])
        print("energy f32", output_f32["energy"])

        abs_error = torch.abs(output_org["energy"] - output_opt["energy"])

        print("---F64:F32_Opt absolute energy error (mean, max)---")
        print("%.5f %.5f" % (abs_error.mean().item(), abs_error.max().item()))

        abs_error = torch.abs(output_org["forces"] - output_opt["forces"])

        print("---F64:F32_Opt absolute force error (mean, max)---")
        print("%.5f %.5f" % (abs_error.mean().item(), abs_error.max().item()))


def benchmark(
    model, model_opt, size=4, compare=False, dtype=torch.float64
) -> None:
    from ase import build

    # build very large diamond structure
    atoms = build.bulk("C", "diamond", a=3.567, cubic=True)
    atoms_list = [atoms.repeat((size, size, size))]
    print("Number of atoms", len(atoms_list[0]))

    configs = [data.config_from_atoms(atoms) for atoms in atoms_list]

    z_table = tools.AtomicNumberTable([int(z) for z in model.atomic_numbers])

    data_loader = torch_geometric.dataloader.DataLoader(
        dataset=[
            data.AtomicData.from_config(
                config, z_table=z_table, cutoff=model.r_max.item()
            )
            for config in configs
        ],
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )
    batch = next(iter(data_loader)).to("cuda")

    print("num nodes", batch.num_nodes)
    print("num edges", batch.edge_index.shape)

    batch = batch.to("cuda")
    batch["positions"] = batch["positions"] + 0.05 * torch.randn(
        batch["positions"].shape,
        dtype=batch["positions"].dtype,
        device=batch["positions"].device,
    )

    batch["positions"].requires_grad_(True)
    batch["node_attrs"].requires_grad_(True)

    batch.positions = batch.positions.to(torch.float32)
    batch.node_attrs = batch.node_attrs.to(torch.float32)
    batch.shifts = batch.shifts.to(torch.float32)

    batch2 = batch.clone()
    batch2.positions = batch2.positions.to(dtype)
    batch2.node_attrs = batch2.node_attrs.to(dtype)
    batch2.shifts = batch2.shifts.to(dtype)

    batch2["positions"].requires_grad_(True)
    batch2["node_attrs"].requires_grad_(True)
    model = model.to(dtype)

    warmup = 30
    niter = 20
    # warmup

    if compare:
        for i in range(warmup):
            output_org = model(
                batch2.to_dict(), training=False, compute_force=True)
        torch.cuda.synchronize()

        start = time()
        for i in range(niter):
            output_org = model(
                batch2.to_dict(), training=False, compute_force=True)
        torch.cuda.synchronize()
        print("original model time: %.2e (ms)" %
              ((time() - start) * (1000 / niter)))

    for i in range(warmup):
        output_opt = model_opt(
            batch.to_dict(), training=False, compute_force=True)
    torch.cuda.synchronize()

    start = time()
    for i in range(warmup):
        output_opt = model_opt(
            batch.to_dict(), training=False, compute_force=True)
    torch.cuda.synchronize()

    rng = nvtx.start_range(message="model_opt", color="blue")
    nbench = 1
    start = time()
    for i in range(nbench):
        output_opt = model_opt(
            batch.to_dict(), training=False, compute_force=True)
    torch.cuda.synchronize()
    print(
        "optimized model time:  %.2e (ms)" % (
            (time() - start) * (1000 / nbench))
    )
    nvtx.end_range(rng)


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    model = torch.load(args.model).to("cuda")
    model = model.to(torch.float64)

    opt_model = OptimizedInvariantMACE(deepcopy(model))

    print("--Original model")
    print(model)
    print("--Optimized model")
    print(opt_model)

    if (args.accuracy):
        print("--Analyzing errors")
        accuracy(model, opt_model, args.size)

    if (args.benchmark):
        print("--Benchmarking")
        benchmark(model, opt_model, args.size, args.compare)

    print(f"--Saving optimized model to: {args.output}")
    torch.save(opt_model, args.output)
