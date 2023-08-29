import numpy as np
from time import time
from opt_einsum import contract
import torch
import logging
import traceback
from mace_ops.cuda import SymmetricContraction
from mace.tools.cg import U_matrix_real
from typing import Dict, Optional, Union
import opt_einsum_fx
import torch
import torch.fx
from e3nn import o3
from e3nn.util.codegen import CodeGenMixin
from e3nn.util.jit import compile_mode
import math
from copy import deepcopy
try:
    from mace_ops.cuda import SymmetricContraction as CUDAContraction_
except ImportError:
    pass

BATCH_EXAMPLE = 10
ALPHABET = ["w", "x", "v", "n", "z", "r", "t", "y", "u", "o", "p", "s"]

class SymmetricContraction(CodeGenMixin, torch.nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
        correlation: Union[int, Dict[str, int]],
        irrep_normalization: str = "component",
        path_normalization: str = "element",
        internal_weights: Optional[bool] = None,
        shared_weights: Optional[bool] = None,
        cuda_optimized: Optional[bool] = False,
        num_elements: Optional[int] = None,
    ) -> None:
        super().__init__()

        if irrep_normalization is None:
            irrep_normalization = "component"

        if path_normalization is None:
            path_normalization = "element"

        assert irrep_normalization in ["component", "norm", "none"]
        assert path_normalization in ["element", "path", "none"]

        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)

        del irreps_in, irreps_out

        self.cuda_optimized = cuda_optimized

        if not isinstance(correlation, tuple):
            corr = correlation
            correlation = {}
            for irrep_out in self.irreps_out:
                correlation[irrep_out] = corr

        assert shared_weights or not internal_weights

        if internal_weights is None:
            internal_weights = True

        self.internal_weights = internal_weights
        self.shared_weights = shared_weights

        del internal_weights, shared_weights

        self.contractions = torch.nn.ModuleList()
        if cuda_optimized:
            self.contractions.append(
                CUDAContraction(
                    irreps_in=self.irreps_in,
                    irreps_out=self.irreps_out, # ILYES: pass all irreps to the CUDA code
                    correlation=correlation[self.irreps_out[-1]], # ILYES: not sure how best to handle this argument for the CUDAContraction, since we fix correlation=3...
                    internal_weights=self.internal_weights,
                    num_elements=num_elements,
                    weights=self.shared_weights,
                )
            )
        else:
            for irrep_out in self.irreps_out:
                self.contractions.append(
                    Contraction(
                        irreps_in=self.irreps_in,
                        irrep_out=o3.Irreps(str(irrep_out.ir)),
                        correlation=correlation[irrep_out],
                        internal_weights=self.internal_weights,
                        num_elements=num_elements,
                        weights=self.shared_weights,
                    )
                )

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        if (self.cuda_optimized): # ILYES: handle the x reshape here so we dont do it 4x in the CUDAContraction module for L=0,1
            
            ## ILYES:  returns [batch_size, nlout, channel_dim]
            ## output has different ordering than the one provided by the contractions, but elements are the same...
            out = self.contractions[0](x.transpose(-1, -2).contiguous(), torch.argmax(y, dim=-1)) 

            resize_shape = torch.prod(torch.tensor(out.shape[1:]))
            return out.view(out.shape[0], resize_shape) 
        else:
            outs = [contraction(x, y) for contraction in self.contractions]
            return torch.cat(outs, dim=-1)
    
class Contraction(torch.nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        irrep_out: o3.Irreps,
        correlation: int,
        internal_weights: bool = True,
        num_elements: Optional[int] = None,
        weights: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()

        self.num_features = irreps_in.count((0, 1))
        self.coupling_irreps = o3.Irreps([irrep.ir for irrep in irreps_in])
        self.correlation = correlation
        dtype = torch.get_default_dtype()
        
        for nu in range(1, correlation + 1):
            U_matrix = U_matrix_real(
                irreps_in=self.coupling_irreps,
                irreps_out=irrep_out,
                correlation=nu,
                dtype=dtype,
            )[-1]
            self.register_buffer(f"U_matrix_{nu}", U_matrix)

        # Tensor contraction equations
        self.contractions_weighting = torch.nn.ModuleList()
        self.contractions_features = torch.nn.ModuleList()

        # Create weight for product basis
        self.weights = torch.nn.ParameterList([])

        for i in range(correlation, 0, -1):
            # Shapes definying
            num_params = self.U_tensors(i).size()[-1]
            num_equivariance = 2 * irrep_out.lmax + 1
            num_ell = self.U_tensors(i).size()[-2]

            if i == correlation:
                parse_subscript_main = (
                    [ALPHABET[j] for j in range(i + min(irrep_out.lmax, 1) - 1)]
                    + ["ik,ekc,bci,be -> bc"]
                    + [ALPHABET[j] for j in range(i + min(irrep_out.lmax, 1) - 1)]
                )
                graph_module_main = torch.fx.symbolic_trace(
                    lambda x, y, w, z: torch.einsum(
                        "".join(parse_subscript_main), x, y, w, z
                    )
                )

                # Optimizing the contractions
                self.graph_opt_main = opt_einsum_fx.optimize_einsums_full(
                    model=graph_module_main,
                    example_inputs=(
                        torch.randn(
                            [num_equivariance] + [num_ell] * i + [num_params]
                        ).squeeze(0),
                        torch.randn((num_elements, num_params, self.num_features)),
                        torch.randn((BATCH_EXAMPLE, self.num_features, num_ell)),
                        torch.randn((BATCH_EXAMPLE, num_elements)),
                    ),
                )
                # Parameters for the product basis
                w = torch.nn.Parameter(
                    torch.randn((num_elements, num_params, self.num_features))
                    / num_params
                )
                self.weights_max = w
            else:
                # Generate optimized contractions equations
                parse_subscript_weighting = (
                    [ALPHABET[j] for j in range(i + min(irrep_out.lmax, 1))]
                    + ["k,ekc,be->bc"]
                    + [ALPHABET[j] for j in range(i + min(irrep_out.lmax, 1))]
                )
                parse_subscript_features = (
                    ["bc"]
                    + [ALPHABET[j] for j in range(i - 1 + min(irrep_out.lmax, 1))]
                    + ["i,bci->bc"]
                    + [ALPHABET[j] for j in range(i - 1 + min(irrep_out.lmax, 1))]
                )

                # Symbolic tracing of contractions
                graph_module_weighting = torch.fx.symbolic_trace(
                    lambda x, y, z: torch.einsum(
                        "".join(parse_subscript_weighting), x, y, z
                    )
                )
                graph_module_features = torch.fx.symbolic_trace(
                    lambda x, y: torch.einsum("".join(parse_subscript_features), x, y)
                )

                # Optimizing the contractions
                graph_opt_weighting = opt_einsum_fx.optimize_einsums_full(
                    model=graph_module_weighting,
                    example_inputs=(
                        torch.randn(
                            [num_equivariance] + [num_ell] * i + [num_params]
                        ).squeeze(0),
                        torch.randn((num_elements, num_params, self.num_features)),
                        torch.randn((BATCH_EXAMPLE, num_elements)),
                    ),
                )
                graph_opt_features = opt_einsum_fx.optimize_einsums_full(
                    model=graph_module_features,
                    example_inputs=(
                        torch.randn(
                            [BATCH_EXAMPLE, self.num_features, num_equivariance]
                            + [num_ell] * i
                        ).squeeze(2),
                        torch.randn((BATCH_EXAMPLE, self.num_features, num_ell)),
                    ),
                )
                self.contractions_weighting.append(graph_opt_weighting)
                self.contractions_features.append(graph_opt_features)
                # Parameters for the product basis
                w = torch.nn.Parameter(
                    torch.randn((num_elements, num_params, self.num_features))
                    / num_params
                )
                self.weights.append(w)
        if not internal_weights:
            self.weights = weights[:-1]
            self.weights_max = weights[-1]

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        out = self.graph_opt_main(
            self.U_tensors(self.correlation),
            self.weights_max,
            x,
            y,
        )
        for i, (weight, contract_weights, contract_features) in enumerate(
            zip(self.weights, self.contractions_weighting, self.contractions_features)
        ):
            c_tensor = contract_weights(
                self.U_tensors(self.correlation - i - 1),
                weight,
                y,
            )
            c_tensor = c_tensor + out
            out = contract_features(c_tensor, x)
        resize_shape = torch.prod(torch.tensor(out.shape[1:]))
        return out.view(out.shape[0], resize_shape)

    def U_tensors(self, nu: int):
        return dict(self.named_buffers())[f"U_matrix_{nu}"]


class CUDAContraction(torch.nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps, # ILYES: should contain all irreps...
        correlation: int,
        internal_weights: bool = True,
        num_elements: Optional[int] = None,
        weights: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()

        self.num_features = irreps_in.count((0, 1))
        self.irreps_out = irreps_out
        self.coupling_irreps = o3.Irreps([irrep.ir for irrep in irreps_in])
        self.correlation = correlation
        dtype = torch.get_default_dtype()

        self.U_matrices = {}
        # Create weight for product basis
        self.weights = {}

        for ir_out in self.irreps_out:
            self.U_matrices[str(ir_out)] = {}
            self.weights[str(ir_out)] = {}

        for nu in range(1, correlation + 1):

            for ir_out in self.irreps_out:
                U_matrix = U_matrix_real(
                    irreps_in=self.coupling_irreps,
                    irreps_out=o3.Irreps(str(ir_out.ir)),
                    correlation=nu,
                    dtype=dtype,
                )[-1]
            
                self.U_matrices[str(ir_out)][f"{nu}"] = U_matrix

        for irrep_key in self.U_matrices.keys():
            for corr_key in self.U_matrices[irrep_key].keys():
                print (irrep_key, corr_key, self.U_matrices[irrep_key][corr_key].shape)
        

        for i in range(correlation, 0, -1):

            for ir_out_ in self.irreps_out:
                ir_out = str(ir_out_)
                # Shapes definying
                num_params = self.U_matrices[ir_out][str(i)].size()[-1]

                if i == correlation:
                    # Parameters for the product basis
                    w = torch.nn.Parameter(
                        torch.randn((num_elements, num_params, self.num_features))
                        / num_params
                    )
                    self.weights[f"{i}"] = w
                else:
                    # Generate optimized contractions equations
                    w = torch.nn.Parameter(
                        torch.randn((num_elements, num_params, self.num_features))
                        / num_params
                    )
                self.weights[str(ir_out)][f"{i}"] = w

        self.symm_contract = CUDAContraction_(
            self.irreps_out, self.U_matrices, self.weights, dtype=dtype, device="cuda"
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        out = self.symm_contract.forward(
            x, y
        )
        return out

nchannels = 96
max_ell = 3
correlation = 3
natoms = 1000
dtype = torch.float32
torch.set_default_dtype(dtype)

#hidden_irreps=o3.Irreps(str(nchannels) + "x0e + " + str(nchannels) + "x1o + " + str(nchannels) + "x2e" )
hidden_irreps=o3.Irreps(str(nchannels) + "x0e + " + str(nchannels) + "x1o")
#hidden_irreps=o3.Irreps(str(nchannels) + "x0e")

sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
num_features = hidden_irreps.count(o3.Irrep(0, 1))
interaction_irreps = (sh_irreps * num_features).sort()[0].simplify()

print (interaction_irreps)
print (num_features)

cuda_optimized = False

symm_contract = SymmetricContraction(interaction_irreps, hidden_irreps, correlation, num_elements=3, cuda_optimized=cuda_optimized).to("cuda")

for param in symm_contract.parameters():
    param.requires_grad = False

X = np.fromfile('symm_contraction_data/X.npy').reshape(21, 128, 16)
Y = np.fromfile('symm_contraction_data/Y.npy').reshape(21, 3)

nrepeats = int(math.ceil(float (natoms) / X.shape[0]))

print ("n_repeats:", nrepeats)

nelements = Y.shape[-1]
X_torch = torch.from_numpy(X).cuda().repeat(nrepeats, 1, 1).type(dtype)
X_torch_copy = torch.from_numpy(X).cuda().repeat(nrepeats, 1, 1).type(dtype)
Y_torch = torch.from_numpy(Y).cuda().repeat(nrepeats, 1).type(dtype)

X_torch = X_torch[:natoms, :nchannels, :]
X_torch_copy = X_torch_copy[:natoms, :nchannels, :]
Y_torch = Y_torch[:natoms]

coupling_irreps = o3.Irreps([irrep.ir for irrep in interaction_irreps])

print (hidden_irreps.num_irreps, hidden_irreps.lmax)

all_weights = {}

for i in range(len(symm_contract.contractions)):
    all_weights[str(i)] = {}
    all_weights[str(i)][3] =  symm_contract.contractions[i].weights_max.detach().clone().type(dtype)
    all_weights[str(i)][2] =  symm_contract.contractions[i].weights[0].detach().clone().type(dtype)
    all_weights[str(i)][1] =  symm_contract.contractions[i].weights[1].detach().clone().type(dtype)

cuda_contraction = CUDAContraction_(coupling_irreps, hidden_irreps, all_weights,nthreadX = 32, nthreadY = 4, nthreadZ = 1, dtype=dtype)

torch.matmul(torch.randn(1024, 1024, device='cuda'),torch.randn(1024, 1024, device='cuda'))

X_torch.requires_grad=True

ntrials = 1000
torch.cuda.synchronize()

start = time()
for i in range (ntrials):
    output = symm_contract.forward(X_torch, Y_torch)

    os = output.sum()

    os.backward()

torch.cuda.synchronize()

end = time()

print ("forward dense:", end - start)

X_torch_copy = X_torch_copy.transpose(-1, -2).contiguous()
X_torch_copy.requires_grad = True


atom_types = torch.argmax(Y_torch, dim=-1).int()

torch.cuda.synchronize()

start = time()
for i in range (ntrials):
    out_cuda = cuda_contraction.forward(X_torch_copy, atom_types)
    os = out_cuda.sum()
    os.backward()

torch.cuda.synchronize()
end = time()
print (end - start)

print (out_cuda[0], out_cuda.shape)
print (output[0], output.shape)

#print (X_torch.grad[0] - X_torch_copy.grad[0].transpose(-1, -2))