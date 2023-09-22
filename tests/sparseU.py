###########################################################################################
# Higher Order Real Clebsch Gordan (based on e3nn by Mario Geiger)
# Authors: Ilyes Batatia
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################
import torch
from mace.tools.cg import U_matrix_real
from typing import Union
import collections
from typing import List, Union

from e3nn import o3

_TP = collections.namedtuple("_TP", "op, args")
_INPUT = collections.namedtuple("_INPUT", "tensor, start, stop")


def _wigner_nj(
    irrepss: List[o3.Irreps],
    normalization: str = "component",
    filter_ir_mid=None,
    dtype=None,
):
    irrepss = [o3.Irreps(irreps) for irreps in irrepss]
    if filter_ir_mid is not None:
        filter_ir_mid = [o3.Irrep(ir) for ir in filter_ir_mid]

    if len(irrepss) == 1:
        (irreps,) = irrepss
        ret = []
        e = torch.eye(irreps.dim, dtype=dtype)
        i = 0
        for mul, ir in irreps:
            for _ in range(mul):
                sl = slice(i, i + ir.dim)
                ret += [(ir, _INPUT(0, sl.start, sl.stop), e[sl])]
                i += ir.dim
        return ret

    *irrepss_left, irreps_right = irrepss
    ret = []
    for ir_left, path_left, C_left in _wigner_nj(
        irrepss_left,
        normalization=normalization,
        filter_ir_mid=filter_ir_mid,
        dtype=dtype,
    ):
        i = 0
        for mul, ir in irreps_right:
            for ir_out in ir_left * ir:
                if filter_ir_mid is not None and ir_out not in filter_ir_mid:
                    continue
                C = o3.wigner_3j(ir_out.l, ir_left.l, ir.l, dtype=dtype)
                if normalization == "component":
                    C *= ir_out.dim**0.5
                if normalization == "norm":
                    C *= ir_left.dim**0.5 * ir.dim**0.5

                C = torch.einsum("jk,ijl->ikl", C_left.flatten(1), C)
                C = C.reshape(
                    ir_out.dim, *(irreps.dim for irreps in irrepss_left), ir.dim
                )
                for u in range(mul):
                    E = torch.zeros(
                        ir_out.dim,
                        *(irreps.dim for irreps in irrepss_left),
                        irreps_right.dim,
                        dtype=dtype,
                    )
                    sl = slice(i + u * ir.dim, i + (u + 1) * ir.dim)
                    E[..., sl] = C
                    ret += [
                        (
                            ir_out,
                            _TP(
                                op=(ir_left, ir, ir_out),
                                args=(
                                    path_left,
                                    _INPUT(len(irrepss_left), sl.start, sl.stop),
                                ),
                            ),
                            E,
                        )
                    ]
            i += mul * ir.dim
    return sorted(ret, key=lambda x: x[0])


def U_matrix_real(
    irreps_in: Union[str, o3.Irreps],
    irreps_out: Union[str, o3.Irreps],
    correlation: int,
    normalization: str = "component",
    filter_ir_mid=None,
    sparse_max=10,
    dtype=None,
):
    irreps_out = o3.Irreps(irreps_out)
    irrepss = [o3.Irreps(irreps_in)] * correlation
    if correlation == 4:
        filter_ir_mid = [
            (0, 1),
            (1, -1),
            (2, 1),
            (3, -1),
            (4, 1),
            (5, -1),
            (6, 1),
            (7, -1),
            (8, 1),
            (9, -1),
            (10, 1),
            (11, -1),
        ]
    wigners = _wigner_nj(irrepss, normalization, filter_ir_mid, dtype)
    current_ir = wigners[0][0]
    out = []
    stack = torch.tensor([])
    for ir, _, base_o3 in wigners:
        if ir in irreps_out and ir == current_ir:
            if stack.shape[-1] <= sparse_max:
                stack = torch.cat((stack, base_o3.squeeze().unsqueeze(-1)), dim=-1)
            last_ir = current_ir
        elif ir in irreps_out and ir != current_ir:
            if len(stack) != 0:
                out += [last_ir, stack]
            stack = base_o3.squeeze().unsqueeze(-1)
            current_ir, last_ir = ir, ir
        else:
            current_ir = ir
    out += [last_ir, stack]
    return out

nchannels = 96
max_ell = 3
correlation = 3
natoms = 100
dtype = torch.float32
torch.set_default_dtype(dtype)

hidden_irreps=o3.Irreps(str(nchannels) + "x0e + " + str(nchannels) + "x1o + " + str(nchannels) + "x2e")
#hidden_irreps=o3.Irreps(str(nchannels) + "x0e + " + str(nchannels) + "x1o")
#hidden_irreps=o3.Irreps(str(nchannels) + "x0e")

sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
num_features = hidden_irreps.count(o3.Irrep(0, 1))
interaction_irreps = (sh_irreps * num_features).sort()[0].simplify()

coupling_irreps = o3.Irreps([irrep.ir for irrep in interaction_irreps])

U_matrices = {}

for nu in range(1, correlation + 1):
    U_matrices[nu] = {}

    for ir_id, ir_out in enumerate(hidden_irreps):
        U_matrix = U_matrix_real(
            irreps_in=coupling_irreps,
            irreps_out=o3.Irreps(str(ir_out.ir)),
            correlation=nu,
            dtype=torch.float32,
            sparse_max=10
        )[-1]
        U_matrices[nu][ir_id] = U_matrix

        idx =  torch.where(U_matrix !=0)

        print (f"nu: {nu}, ir_id: {ir_id}, ir_out: {ir_out}, U.shape {U_matrix.shape}, num_nonzero: {len(idx[0])}")
