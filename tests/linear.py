import warnings
import os
import sys
from mace.modules.irreps_tools import (
    linear_out_irreps,
    tp_out_irreps_with_instructions,
)
from e3nn import o3
from time import time
from cuda_mace.ops.linear import Linear
from test_utils import build_parser, get_torch_dtype, shape_irreps, reshape_irreps
import torch
torch.serialization.add_safe_globals([slice])


warnings.filterwarnings(
    "ignore",
    message="The TorchScript type system*",
    category=UserWarning,
    module="torch.jit._check",
)
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")))


def unsimplify_irreps(irreps, nchannels):
    """Convert simplified irreps to an expanded form."""
    rstr = ""
    for i, (mul, irrep) in enumerate(irreps):
        repeat = int(mul / nchannels)
        for j in range(repeat):
            rstr += f"{nchannels}x{irrep}"
            if j < repeat - 1:
                rstr += "+"
        if i < len(irreps) - 1:
            rstr += "+"
    return o3.Irreps(rstr)


def benchmark(irreps_mid, irreps_out, nchannels, grad=False, dtype=torch.float32):
    nnodes = 1000

    print("-Linear")
    print("nnodes: ", nnodes, "nchannels:", nchannels)
    print("input irreps:", irreps_mid)
    print("output irreps:", irreps_out)

    linear = (
        o3.Linear(irreps_mid, irreps_out,
                  internal_weights=True, shared_weights=True)
        .to("cuda")
        .to(dtype)
    )

    X = torch.randn(
        (nnodes, sum([irrep.dim for _, irrep in irreps_mid]), nchannels),
        device="cuda",
        dtype=torch.float32,
        requires_grad=True,
    )
    X_e3nn = X.clone().detach().to(dtype).requires_grad_(True)
    X_e3nn = shape_irreps(irreps_mid)(X_e3nn.transpose(-1, -2))
    e3nn_out = linear(X_e3nn)
    e3nn_out = reshape_irreps(irreps_out)(e3nn_out)
    e3nn_out = e3nn_out.transpose(-1, -2)

    cuda_linear = Linear(linear)

    torch.cuda.synchronize()
    for i in range(10):
        out = cuda_linear.forward(X)
    torch.cuda.synchronize()

    start = time()
    for i in range(1000):
        out = cuda_linear.forward(X)
        if grad:
            (out**2.0).sum().backward()
    torch.cuda.synchronize()

    print("CUDA time (ms):", time() - start)

    torch.cuda.synchronize()
    for i in range(10):
        e3nn_out = linear(X_e3nn)
    torch.cuda.synchronize()

    start = time()
    for i in range(1000):
        e3nn_out = linear(X_e3nn)
        if grad:
            (e3nn_out**2.0).sum().backward()
    torch.cuda.synchronize()

    print("E3NN time (ms):", time() - start)


def accuracy(irreps_in, irreps_out, nchannels, grad=False, dtype=torch.float32):

    nnodes = 1000

    unsimplified_irrep_in = unsimplify_irreps(irreps_in, nchannels)

    linear = (
        o3.Linear(irreps_in, irreps_out,
                  internal_weights=True, shared_weights=True)
        .to("cuda")
        .to(dtype)
    )

    X = torch.randn(
        (nnodes, sum(
            [irrep.dim for _, irrep in unsimplified_irrep_in]), nchannels),
        device="cuda",
        dtype=torch.float32,
        requires_grad=True,
    )

    X_e3nn = X.clone().detach().to(dtype).requires_grad_(True)

    e3nn_out = linear(shape_irreps(unsimplified_irrep_in)
                      (X_e3nn.transpose(-1, -2)))
    e3nn_out = reshape_irreps(irreps_out)(e3nn_out)
    e3nn_out = e3nn_out.transpose(-1, -2)

    cuda_linear = Linear(linear)

    print("--Linear")
    print("nnodes: ", nnodes, "nchannels:", nchannels)
    print("input irreps:", irreps_in)
    print("unsimplified input irreps:", unsimplified_irrep_in)
    print("output irreps:", irreps_out)

    out = cuda_linear.forward(X)

    error_vs_e3nn = (out - e3nn_out).abs()
    print("-output errors")
    print(
        f"min {torch.min(error_vs_e3nn).item():.2e} max {torch.max(error_vs_e3nn).item():.2e} mean {torch.mean(error_vs_e3nn).item():.2e}"
    )

    assert torch.mean(error_vs_e3nn).item(
    ) < 5e-5, "precision check failed on output"

    if grad:
        print("--grad errors")
        (e3nn_out**2.0).sum().backward()
        (out**2.0).sum().backward()

        error_vs_e3nn = (X.grad - X_e3nn.grad).abs()
        print(
            f"min {torch.min(error_vs_e3nn).item():.2e} max {torch.max(error_vs_e3nn).item():.2e} mean {torch.mean(error_vs_e3nn).item():.2e}"
        )
        assert torch.mean(error_vs_e3nn).item(
        ) < 5e-5, "precision check failed on grad"

    print("--=--")


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    nchannels = 128

    if args.accuracy:

        accuracy(
            o3.Irreps("128x0e+128x1o+128x2e+128x3o"),
            o3.Irreps("128x0e+128x1o+128x2e+128x3o"),
            128,
            args.grad,
            get_torch_dtype(args.dtype),
        )

    if args.benchmark:
        benchmark(
            o3.Irreps("128x0e+128x1o+128x2e+128x3o"), o3.Irreps(
                "128x0e+128x1o+128x2e+128x3o"), 128, args.grad, get_torch_dtype(args.dtype)
        )
