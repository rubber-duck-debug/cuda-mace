import torch
import numpy as np
from e3nn import o3
from typing import Tuple, List
from e3nn import o3
from torch.profiler import profile, record_function, ProfilerActivity

def tp_out_irreps_with_instructions(
    irreps1: o3.Irreps, irreps2: o3.Irreps, target_irreps: o3.Irreps
) -> Tuple[o3.Irreps, List]:
    trainable = True

    # Collect possible irreps and their instructions
    irreps_out_list: List[Tuple[int, o3.Irreps]] = []
    instructions = []
    for i, (mul, ir_in) in enumerate(irreps1):
        for j, (_, ir_edge) in enumerate(irreps2):
            for ir_out in ir_in * ir_edge:  # | l1 - l2 | <= l <= l1 + l2
                if ir_out in target_irreps:
                    k = len(irreps_out_list)  # instruction index
                    irreps_out_list.append((mul, ir_out))
                    instructions.append((i, j, k, "uvu", trainable))

    # We sort the output irreps of the tensor product so that we can simplify them
    # when they are provided to the second o3.Linear
    irreps_out = o3.Irreps(irreps_out_list)
    irreps_out, permut, _ = irreps_out.sort()

    # Permute the output indexes of the instructions to match the sorted irreps:
    instructions = [
        (i_in1, i_in2, permut[i_out], mode, train)
        for i_in1, i_in2, i_out, mode, train in instructions
    ]

    instructions = sorted(instructions, key=lambda x: x[2])

    return irreps_out, instructions

if __name__ == "__main__":

    irreps1, irreps2, target_irreps = o3.Irreps("256x0e + 256x1o"), o3.Irreps("1x0e + 1x1o + 1x2e + 1x3o"), o3.Irreps("256x0e + 256x1o + 256x2e + 256x3o")
    irreps_out, instructions = tp_out_irreps_with_instructions(irreps1, irreps2, target_irreps)

    tp_torch = o3.TensorProduct(irreps1, irreps2,irreps_out, instructions).to("cuda")

    l1 = 1
    l2 = 3
    n_edges = 2000

    X1 = torch.randn(n_edges, 256, (l1 + 1)**2)
    X2 = torch.randn(n_edges, 1, (l2 + 1)**2)
    X1 = X1.to("cuda")
    X2 = X2.to("cuda")

    import torch.utils.benchmark as benchmark

    t0 = benchmark.Timer(
    stmt='tp(X1, X2)',
    globals={'X1': X1.reshape(n_edges, 1024), 'X2': X2.reshape(n_edges,16), "tp": tp_torch})

    print(t0.timeit(1000))

    out = tp_torch(X1.reshape(n_edges, 1024), X2.reshape(n_edges,16))

    out =  out.reshape(n_edges, 256, 40)

    # with profile(
    #     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    #     with_stack=True,
    # ) as prof:
    #     tp_torch( X1.reshape(n_edges, 1024),  X2.reshape(n_edges,16))

    #print(prof.key_averages(group_by_stack_n=5).table(sort_by="self_cuda_time_total", row_limit=5))