import os
from mace.tools import torch_geometric
from mace import data, tools
from e3nn import o3
from ase import build
import argparse
import torch
torch.serialization.add_safe_globals([slice])


def build_parser():
    """
    Create a parser for the command line tool.
    """
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float64"],
        default="float32",
        help="Data type for torch tensor: float32 or float64",
    )

    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Benchmark the optimized op.",
        default=False,
    )

    parser.add_argument(
        "--accuracy",
        action="store_true",
        help="accuracy of the optimized op.",
        default=False,
    )

    parser.add_argument(
        "--grad",
        action="store_true",
        help="compute grads",
        default=False,
    )

    return parser


def get_torch_dtype(dtype_str):
    if dtype_str == "float32":
        return torch.float32
    elif dtype_str == "float64":
        return torch.float64
    else:
        raise ValueError(f"Unsupported data type: {dtype_str}")


def load_water():
    from ase.io import read

    current_dir = os.path.dirname(os.path.abspath(__file__))
    atoms = read(current_dir + "/../data/prepared_system.pdb")
    atomic_numbers = [1, 8]
    cutoff = 4.5
    configs = [data.config_from_atoms(atoms)]

    z_table = tools.AtomicNumberTable([int(z) for z in atomic_numbers])

    data_loader = torch_geometric.dataloader.DataLoader(
        dataset=[
            data.AtomicData.from_config(config, z_table=z_table, cutoff=cutoff)
            for config in configs
        ],
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )
    # purturb system a bit to break symmetries
    batch = next(iter(data_loader)).to("cuda")

    return batch


# MACE
def create_system(size=5, cutoff=4.5):
    from ase import build

    # build diamond structure
    atoms = build.bulk("C", "diamond", a=3.567, cubic=True)
    atoms_list = [atoms.repeat((size, size, size))]
    atomic_numbers = [6]

    configs = [data.config_from_atoms(atoms) for atoms in atoms_list]

    z_table = tools.AtomicNumberTable([int(z) for z in atomic_numbers])

    data_loader = torch_geometric.dataloader.DataLoader(
        dataset=[
            data.AtomicData.from_config(config, z_table=z_table, cutoff=cutoff)
            for config in configs
        ],
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )
    # purturb system a bit to break symmetries
    batch = next(iter(data_loader)).to("cuda")
    batch["positions"] = batch["positions"] + 0.05 * torch.randn(
        batch["positions"].shape,
        dtype=batch["positions"].dtype,
        device=batch["positions"].device,
    )

    return batch


class shape_irreps(torch.nn.Module):
    # code the reverse of reshape_irreps
    def __init__(self, irreps: o3.Irreps) -> None:
        super().__init__()
        self.irreps = irreps

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        # the reverse of reshape_irreps
        ix = 0
        out = []
        batch, _, _ = tensor.shape
        for mul, ir in self.irreps:
            d = ir.dim
            field = tensor[:, :, ix: ix + d]
            field = field.reshape(batch, mul * d)
            ix = ix + d
            out.append(field)
        return torch.cat(out, dim=-1)


class reshape_irreps(torch.nn.Module):
    def __init__(self, irreps: o3.Irreps) -> None:
        super().__init__()
        self.irreps = irreps

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        ix = 0
        out = []
        batch, _ = tensor.shape
        # mul = multiplicity, i.e nchannels, ir.dim dimension of IR (1, 3, 5...)
        for mul, ir in self.irreps:
            d = ir.dim
            field = tensor[:, ix: ix + mul * d]  # [batch, sample, mul * repr]
            ix += mul * d
            field = field.reshape(batch, mul, d)
            out.append(field)
        return torch.cat(out, dim=-1)


if __name__ == "__main__":

    print(load_water())
