from test_utils import build_parser, get_torch_dtype, shape_irreps, reshape_irreps
import torch
from e3nn import o3
from mace.tools.scatter import scatter_sum

from mace.modules.irreps_tools import (
    tp_out_irreps_with_instructions,
)

from ase import build
from mace import data, tools
from mace.tools import torch_geometric
from time import time

from cuda_mace.ops.invariant_message_passing import InvariantMessagePassingTP

import sys
import os
from test_utils import create_system

import warnings

warnings.filterwarnings(
    "ignore",
    message="The TorchScript type system*",
    category=UserWarning,
    module="torch.jit._check",
)


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def gradcheck() -> None:

    class tp_wrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.tp = InvariantMessagePassingTP()

        def forward(self, *args):
            return (self.tp(*args) ** 2.0).sum()

    tp_cuda = tp_wrapper().to(torch.float64)

    nchannels = 160
    cutoff = 2.5
    size = 2

    batch = create_system(size, cutoff)

    nnodes = batch.num_nodes
    nedges = batch.edge_index.shape[1]
    sender = batch.edge_index[1].int()
    receiver = batch.edge_index[0].int()

    irreps1, irreps2, target_irreps = (
        o3.Irreps(f"{args.nchannels}x0e"),
        o3.Irreps(f"1x0e+1x1o+1x2e+1x3o"),
        o3.Irreps(
            f"{args.nchannels}x0e+{args.nchannels}x1o+{args.nchannels}x2e+{args.nchannels}x3o"
        ),
    )

    node_feats = torch.rand(
        (nnodes, nchannels),
        dtype=torch.float64,
        device="cuda",
        requires_grad=True,
    )
    edge_attrs = torch.rand(
        ((irreps2.lmax + 1) ** 2, nedges),
        dtype=torch.float64,
        device="cuda",
        requires_grad=True,
    )
    tp_weights = torch.rand(
        (nedges, 4, nchannels),
        dtype=torch.float64,
        device="cuda",
        requires_grad=True,
    )

    val = torch.autograd.gradcheck(
        tp_cuda.forward,
        (node_feats[sender], edge_attrs, tp_weights, sender, receiver, nnodes),
        eps=1e-5,
        raise_exception=True,
        fast_mode=False,
        atol=1e-5,
        rtol=1e-7,
    )

    print("torch.autograd.gradcheck passed!")


def accuracy(
    tp_cuda, nchannels=128, dtype=torch.float64, size=5, with_grad=False, cutoff=4.5
) -> None:

    batch = create_system(size, cutoff=cutoff)

    nnodes = batch.num_nodes
    nedges = batch.edge_index.shape[1]
    sender = batch.edge_index[1]
    receiver = batch.edge_index[0]

    irreps1, irreps2, target_irreps = (
        o3.Irreps(f"{args.nchannels}x0e"),
        o3.Irreps(f"1x0e+1x1o+1x2e+1x3o"),
        o3.Irreps(
            f"{args.nchannels}x0e+{args.nchannels}x1o+{args.nchannels}x2e+{args.nchannels}x3o"
        ),
    )

    node_feats = torch.rand(
        (nnodes, nchannels),
        dtype=dtype,
        device="cuda",
        requires_grad=with_grad,
    )
    edge_attrs = torch.rand(
        ((irreps2.lmax + 1) ** 2, nedges),
        dtype=dtype,
        device="cuda",
        requires_grad=with_grad,
    )
    tp_weights = torch.rand(
        (nedges, 4, nchannels),
        dtype=dtype,
        device="cuda",
        requires_grad=with_grad,
    )

    node_feats_e3nn = node_feats.clone().detach().to(dtype).requires_grad_(with_grad)
    edge_attrs_e3nn = edge_attrs.clone().detach().to(dtype).requires_grad_(with_grad)
    tp_weights_e3nn = tp_weights.clone().detach().to(dtype).requires_grad_(with_grad)

    node_feats_ref = node_feats.clone().detach().requires_grad_(with_grad)
    edge_attrs_ref = edge_attrs.clone().detach().requires_grad_(with_grad)
    tp_weights_ref = tp_weights.clone().detach().requires_grad_(with_grad)

    out_cuda = tp_cuda.forward(
        node_feats[sender],
        edge_attrs,
        tp_weights,
        sender.int(),
        receiver.int(),
        node_feats.shape[0],
    )

    # TensorProduct
    irreps_mid, instructions = tp_out_irreps_with_instructions(
        irreps1,
        irreps2,
        target_irreps,
    )

    conv_tp = (
        o3.TensorProduct(
            irreps1,
            irreps2,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
        )
        .to("cuda")
        .to(dtype)
    )

    mji = conv_tp(
        node_feats_e3nn[sender],
        shape_irreps(irreps2)(edge_attrs_e3nn.transpose(-1, -2)[:, None, :]),
        tp_weights_e3nn.reshape(
            tp_weights_e3nn.shape[0],
            tp_weights_e3nn.shape[1] * tp_weights_e3nn.shape[2],
        ),
    )  # [n_edges, irreps]
    message = scatter_sum(
        src=mji, index=receiver, dim=0, dim_size=nnodes
    )  # [n_nodes, irreps]

    e3nn_out = reshape_irreps(conv_tp.irreps_out)(message)

    error_vs_e3nn = (out_cuda - e3nn_out.transpose(-1, -2)).abs()

    print(out_cuda[0])
    print(e3nn_out.transpose(-1, -2)[0])
    print("-Invariant Message Passing Tensor Product")
    print("nnodes: ", nnodes, "nedges", nedges)
    print("input1 irreps:", irreps1)
    print("input2 irreps:", irreps2)
    print("output irreps:", irreps_mid)

    print("--output errors", out_cuda.dtype)
    print(
        f"min {torch.min(error_vs_e3nn).item():.2e} max {torch.max(error_vs_e3nn).item():.2e} mean {torch.mean(error_vs_e3nn).item():.2e}"
    )

    if with_grad:
        (out_cuda**2).sum().backward()
        (e3nn_out**2).sum().backward()

        torch.cuda.synchronize()

        print("--edge_attrs grad error")
        error_vs_e3nn = (edge_attrs.grad - edge_attrs_e3nn.grad).abs()
        print(
            f"min {torch.min(error_vs_e3nn).item():.2e} max {torch.max(error_vs_e3nn).item():.2e} mean {torch.mean(error_vs_e3nn).item():.2e}"
        )

        print("--weights grad error")
        error_vs_e3nn = (tp_weights.grad - tp_weights_e3nn.grad).abs()
        print(
            f"min {torch.min(error_vs_e3nn).item():.2e} max {torch.max(error_vs_e3nn).item():.2e} mean {torch.mean(error_vs_e3nn).item():.2e}"
        )

        print("--node_feats grad error")
        error_vs_e3nn = (node_feats.grad - node_feats_e3nn.grad).abs()
        print(
            f"min {torch.min(error_vs_e3nn).item():.2e} max {torch.max(error_vs_e3nn).item():.2e} mean {torch.mean(error_vs_e3nn).item():.2e}"
        )


def benchmark(
    tp_cuda,
    nchannels=128,
    dtype=torch.float64,
    size=5,
    with_grad=False,
    compare=False,
    cutoff=4.5,
) -> None:
    batch = create_system(size, cutoff=cutoff)

    nnodes = batch.num_nodes
    nedges = batch.edge_index.shape[1]
    sender = batch.edge_index[1]
    receiver = batch.edge_index[0]

    irreps1, irreps2, target_irreps = (
        o3.Irreps(f"{args.nchannels}x0e"),
        o3.Irreps(f"1x0e+1x1o+1x2e+1x3o"),
        o3.Irreps(
            f"{args.nchannels}x0e+{args.nchannels}x1o+{args.nchannels}x2e+{args.nchannels}x3o"
        ),
    )

    node_feats = torch.randn(
        (nnodes, nchannels),
        dtype=torch.float32,
        device="cuda",
        requires_grad=with_grad,
    )
    edge_attrs = torch.randn(
        ((irreps2.lmax + 1) ** 2, nedges),
        dtype=torch.float32,
        device="cuda",
        requires_grad=with_grad,
    )
    tp_weights = torch.randn(
        (nedges, 4, nchannels),
        dtype=torch.float32,
        device="cuda",
        requires_grad=with_grad,
    )

    # TensorProduct
    irreps_mid, instructions = tp_out_irreps_with_instructions(
        irreps1,
        irreps2,
        target_irreps,
    )

    print("-Invariant Message Passing Tensor Product")
    print("nnodes: ", nnodes, "nedges", nedges)
    print("input1 irreps:", irreps1)
    print("input2 irreps:", irreps2)
    print("output irreps:", irreps_mid)

    niter = 20

    # for i in range(niter):
    #     out = tp_cuda.forward(
    #         node_feats[sender],
    #         edge_attrs,
    #         tp_weights,
    #         sender.int(),
    #         receiver.int(),
    #         node_feats.shape[0],
    #     )

    torch.cuda.synchronize()

    print(receiver)
    print(sender)

    torch.cuda.cudart().cudaProfilerStart()
    start = time()
    torch.cuda.nvtx.range_push("iter")
    for i in range(niter):
        out = tp_cuda.forward(
            node_feats[sender],
            edge_attrs,
            tp_weights,
            sender.int(),
            receiver.int(),
            node_feats.shape[0],
        )

        out.sum().backward()

    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    end = time()

    print(f"--CUDA time (ms)")
    print(f"{((end - start) * 1000 / niter):.2e} ms")
    torch.cuda.cudart().cudaProfilerStop()
    if compare:
        conv_tp = (
            o3.TensorProduct(
                irreps1,
                irreps2,
                irreps_mid,
                instructions=instructions,
                shared_weights=False,
                internal_weights=False,
            )
            .to("cuda")
            .to(dtype)
        )

        node_feats_e3nn = torch.randn(
            (nnodes, (irreps1.lmax + 1) ** 2 * nchannels),
            dtype=dtype,
            device="cuda",
            requires_grad=with_grad,
        )
        edge_attrs_e3nn = torch.randn(
            (nedges, (irreps2.lmax + 1) ** 2),
            dtype=dtype,
            device="cuda",
            requires_grad=with_grad,
        )
        tp_weights_e3nn = torch.randn(
            (nedges, 4 * nchannels),
            dtype=dtype,
            device="cuda",
            requires_grad=with_grad,
        )

        for i in range(niter):
            mji = conv_tp(
                node_feats_e3nn[receiver], edge_attrs_e3nn, tp_weights_e3nn
            )  # [n_edges, irreps]
            message = scatter_sum(
                src=mji, index=sender, dim=0, dim_size=nnodes
            )  # [n_nodes, irreps]

        torch.cuda.synchronize()
        start = time()
        for i in range(niter):
            mji = conv_tp(
                node_feats_e3nn[sender], edge_attrs_e3nn, tp_weights_e3nn
            )  # [n_edges, irreps]
            message = scatter_sum(
                src=mji, index=receiver, dim=0, dim_size=nnodes
            )  # [n_nodes, irreps]
            if with_grad:
                message.sum().backward()
        torch.cuda.synchronize()
        end = time()
        print(f"--Torch time (ms)")
        print(f"{((end - start) * 1000 / niter):.2e} ms")


if __name__ == "__main__":

    parser = build_parser()

    parser.add_argument(
        "--size",
        help="number of cells to repeat in x, y, z direction.",
        default=2,
        type=int,
    )

    parser.add_argument(
        "--nchannels",
        help="number of independent channels",
        default=128,
        type=int,
    )

    parser.add_argument(
        "--compare",
        help="compare vs e3nn",
        default=0,
        type=int,
    )

    parser.add_argument("--cutoff", help="radial cutoff", default=4.5, type=float)

    args = parser.parse_args()

    tp_cuda = InvariantMessagePassingTP()

    if args.benchmark:
        benchmark(
            tp_cuda,
            args.nchannels,
            get_torch_dtype(args.dtype),
            args.size,
            args.grad,
            args.compare,
            args.cutoff,
        )

    if args.accuracy:
        accuracy(
            tp_cuda,
            args.nchannels,
            get_torch_dtype(args.dtype),
            args.size,
            args.grad,
            args.cutoff,
        )

        gradcheck()
