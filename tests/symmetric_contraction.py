from test_utils import build_parser, get_torch_dtype, shape_irreps, reshape_irreps
import torch
from time import time
from e3nn import o3

import sys
import os

from mace.modules.symmetric_contraction import SymmetricContraction as TorchContraction
from cuda_mace.ops.symmetric_contraction import SymmetricContraction as CUDAContraction

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")))


def generate_one_hot(nnodes, nelements):
    indices = torch.randint(0, nelements, (nnodes,))
    one_hot = torch.zeros((nnodes, nelements), dtype=torch.int)
    one_hot[torch.arange(nnodes), indices] = 1
    return one_hot


def benchmark(grad=False, dtype_reference=torch.float32):
    nnodes = 1000
    nchannels = 128
    num_elements = 10
    correlation = 3

    node_feats_irreps, target_irreps = (
        o3.Irreps(
            f"{nchannels}x0e+{nchannels}x1o+{nchannels}x2e+{nchannels}x3o"),
        o3.Irreps(f"{nchannels}x0e + {nchannels}x1o"),
    )

    print("-SymmetricContraction")
    print("nnodes: ", nnodes, "nelements", num_elements)
    print("node_feats_irreps:", node_feats_irreps)
    print("target_irreps:", target_irreps)

    symm_torch = (
        TorchContraction(
            irreps_in=node_feats_irreps,
            irreps_out=target_irreps,
            correlation=correlation,
            num_elements=num_elements,
        )
        .to("cuda")
        .to(dtype_reference)
    )

    cuda_symm = CUDAContraction(
        symm_torch,
        dtype=torch.float32,
    )

    one_hot_embedding = generate_one_hot(nnodes, num_elements).cuda()

    X = torch.randn(
        (nnodes, sum([irrep.dim for _, irrep in node_feats_irreps]), nchannels),
        device="cuda",
        dtype=torch.float32,
        requires_grad=True,
    )

    X_e3nn = X.clone().detach().to(dtype_reference).requires_grad_(True)

    X3e = reshape_irreps(node_feats_irreps)(
        shape_irreps(node_feats_irreps)(X_e3nn.transpose(-1, -2))
    )

    # warmup
    for i in range(10):
        output_cuda = cuda_symm.forward(
            X, one_hot_embedding.argmax(dim=-1).int())
        if grad:
            (output_cuda**2.0).sum().backward()

    torch.cuda.synchronize()
    start = time()
    for i in range(100):
        output_cuda = cuda_symm.forward(
            X, one_hot_embedding.argmax(dim=-1).int())
        if grad:
            (output_cuda**2.0).sum().backward()
    torch.cuda.synchronize()
    print("CUDA time (ms)")
    print((time() - start) * 10)

    start = time()
    for i in range(100):
        output_torch = symm_torch.forward(
            X3e, one_hot_embedding.to(torch.float32))
        if grad:
            (output_torch**2.0).sum().backward()
    torch.cuda.synchronize()
    print("Torch time (ms)")
    print((time() - start) * 10)


def accuracy(grad=False, dtype_reference=torch.float32):
    nnodes = 256
    nchannels = 128
    num_elements = 10
    correlation = 3

    node_feats_irreps, target_irreps = (
        o3.Irreps(
            f"{nchannels}x0e+{nchannels}x1o+{nchannels}x2e+{nchannels}x3o"),
        o3.Irreps(f"{nchannels}x0e"),
    )

    print("-SymmetricContraction")
    print("nnodes: ", nnodes, "nelements", num_elements)
    print("node_feats_irreps:", node_feats_irreps)
    print("target_irreps:", target_irreps)

    symm_torch = (
        TorchContraction(
            irreps_in=node_feats_irreps,
            irreps_out=target_irreps,
            correlation=correlation,
            num_elements=num_elements,
        )
        .to("cuda")
        .to(dtype_reference)
    )

    cuda_symm = CUDAContraction(
        symm_torch,
        dtype=torch.float32,
    )

    one_hot_embedding = generate_one_hot(nnodes, num_elements).cuda()

    X = torch.randn(
        (nnodes, sum([irrep.dim for _, irrep in node_feats_irreps]), nchannels),
        device="cuda",
        dtype=torch.float32,
        requires_grad=True,
    )

    X_e3nn = X.clone().detach().to(dtype_reference).requires_grad_(True)

    X3e = reshape_irreps(node_feats_irreps)(
        shape_irreps(node_feats_irreps)(X_e3nn.transpose(-1, -2))
    )

    output_torch = symm_torch.forward(
        X3e, one_hot_embedding.to(dtype_reference))

    output_torch = reshape_irreps(target_irreps)(
        output_torch).transpose(-1, -2)

    output_cuda = cuda_symm.forward(X, one_hot_embedding.argmax(dim=-1).int())

    error_vs_e3nn = (output_cuda - output_torch).abs()

    print("--output error")
    print(
        f"min {torch.min(error_vs_e3nn).item():.2e} max {torch.max(error_vs_e3nn).item():.2e} mean {torch.mean(error_vs_e3nn).item():.2e}"
    )

    if grad:
        (output_cuda**2.0).sum().backward()
        (output_torch**2.0).sum().backward()
        torch.cuda.synchronize()

        error_vs_e3nn = (X_e3nn.grad - X.grad).abs()

        print("--grad error")
        print(
            f"min {torch.min(error_vs_e3nn).item():.2e} max {torch.max(error_vs_e3nn).item():.2e} mean {torch.mean(error_vs_e3nn).item():.2e}"
        )


if __name__ == "__main__":

    parser = build_parser()
    args = parser.parse_args()

    if args.accuracy:
        accuracy(args.grad, get_torch_dtype(args.dtype))

    if args.benchmark:
        benchmark(args.grad, get_torch_dtype(args.dtype))
