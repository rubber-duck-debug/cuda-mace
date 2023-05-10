import os
import sys
import torch
import sysconfig
from e3nn import o3
from e3nn_jax import Instruction, Irreps
from e3nn_jax._src.core_tensor_product import _normalize_instruction_path_weights

_HERE = os.path.realpath(os.path.dirname(__file__))
EXT_SUFFIX = sysconfig.get_config_var('EXT_SUFFIX')

torch.ops.load_library(_HERE + '/tensor_product.so')
torch.ops.load_library(_HERE + '/symmetric_contraction.so')

def _tensor_product(
                X1, X2, 
                mu1, mu2, mu3,
                X1_ordering, X2_ordering, X3_ordering,
                cg_coeffs,
                output_size, nthreadx, nthready, nthreadz):
    
    return torch.ops.mace_cuda_tp.tensor_product(
                X1, X2, 
                mu1, mu2, mu3,
                X1_ordering, X2_ordering, X3_ordering,
                cg_coeffs,
                output_size, nthreadx, nthready, nthreadz)

def _symmetric_contraction(
                X: torch.Tensor, 
                atom_types: torch.Tensor, 
                U3_nonsparse_indices: torch.Tensor, 
                U3_num_nonsparse: torch.Tensor, 
                U3_nonsparse_elements: torch.Tensor, 
                U2_nonsparse_indices: torch.Tensor, 
                U2_nonsparse_elements: torch.Tensor, 
                U1: torch.Tensor, 
                W3: torch.Tensor, 
                W2: torch.Tensor, 
                W1 : torch.Tensor, 
                nthreadx: int,
                nthready: int,
                nthreadz: int ) -> torch.Tensor:
    
    return torch.ops.mace_cuda_symm_contraction.symmetric_contraction(
                X, 
                atom_types, 
                U3_nonsparse_indices, 
                U3_num_nonsparse, 
                U3_nonsparse_elements, 
                U2_nonsparse_indices, 
                U2_nonsparse_elements, 
                U1, 
                W3, 
                W2, 
                W1, 
                nthreadx,
                nthready,
                nthreadz)


class TensorProduct(torch.nn.Module):

  def __init__(self, irreps_in1, irreps_in2, target_irreps, device="cuda", dtype=torch.float64):
    super().__init__()
    
    self.irreps_in1 = o3.Irreps(irreps_in1)
    self.irreps_in2 = o3.Irreps(irreps_in2)
    self.target_irreps = o3.Irreps(target_irreps)
    
    self.device = device
    self.dtype= dtype
    
    instructions = []
    irreps_out = []
    
    for i, (mul, ir_in) in enumerate(self.irreps_in1):
            for j, (_, ir_edge) in enumerate(self.irreps_in2):
                for ir_out in ir_in * ir_edge:  # | l1 - l2 | <= l <= l1 + l2
                    if ir_out in target_irreps:
                        
                        l1 = ir_in.l
                        l2 = ir_edge.l
                        l3 = ir_out.l
            
                        instructions.append(
                            Instruction(
                                i_in1=i,
                                i_in2=j,
                                i_out=len(instructions),
                                connection_mode="uvu",
                                has_weight=False,
                                path_weight=1.0,
                                weight_std=None,
                                first_input_multiplicity=mul,
                                second_input_multiplicity=1,
                                output_multiplicity=mul,
                            )
                        )
                        irreps_out.append((mul, ir_out))
                        
    self.irreps_out = Irreps(irreps_out)

    self.instructions = _normalize_instruction_path_weights(
            instructions,
            self.irreps_in1,
            self.irreps_in2,
            self.irreps_out,
            [1.0 for _ in self.irreps_in1],
            [1.0 for _ in self.irreps_in2],
            [1.0 for _ in self.irreps_out],
            irrep_normalization="component",
            path_normalization_exponent=1.0,  # path
            gradient_normalization_exponent=1.0,  # path
        )
    
    mu_1 = []
    mu_2 = []
    mu_3 = []
    cg_coeffs = []
    
    for ins in self.instructions:
        l1 = self.irreps_in1[ins.i_in1].ir.l
        l2 = self.irreps_in2[ins.i_in2].ir.l
        l3 = self.irreps_out[ins.i_out].ir.l

        offset1 = self.irreps_in1[: ins.i_in1].dim
        offset2 = self.irreps_in2[: ins.i_in2].dim
        offset3 = self.irreps_out[: ins.i_out].dim
        
        cg = o3.wigner_3j(l1, l2, l3).to(self.device).type(dtype)

        # normalisation and weighting:
        cg = cg * ins.path_weight

        mu1, mu2, mu3 = cg.nonzero(as_tuple=True)
        
        cg_sparse = cg[(mu1, mu2, mu3)]
        
        mu1 = mu1 + offset1
        mu2 = mu2 + offset2
        mu3 = mu3 + offset3

        mu_1.append(mu1)
        mu_2.append(mu2)
        mu_3.append(mu3)

        cg_coeffs.append(cg_sparse)
    
    self.mu1 = torch.cat(mu_1).cuda().type(torch.int16)
    self.mu2 = torch.cat(mu_2).cuda().type(torch.int16)
    self.mu3 = torch.cat(mu_3).cuda().type(torch.int16)

    self.cg_coeffs = torch.cat(cg_coeffs).type(self.dtype).cuda()

    self.mu_1_sort = torch.argsort(self.mu1).type(torch.int16).cuda()
    self.mu_2_sort = torch.argsort(self.mu2).type(torch.int16).cuda()
    self.mu_3_sort = torch.argsort(self.mu3).type(torch.int16).cuda()


  def forward(self,x,y):

    assert x.dtype == self.dtype and y.dtype == self.dtype, f"x: {x.dtype} and y:{y.dtype} need to be the same as this class: {self.dtype}"
    
    nthreads = 32
    if (x.shape[-1] == 96):
        nthreads = 96
    elif (x.shape[-1] >= 64):
        nthreads = 64

    return _tensor_product(x, y, 
                          self.mu1, self.mu2, self.mu3,
                          self.mu_1_sort, self.mu_2_sort, self.mu_3_sort,
                          self.cg_coeffs,
                          self.irreps_out.dim, nthreads, 1, 1)


class SymmetricContraction(torch.nn.Module):

    def __init__(self, U_tensors, W_tensors, device="cuda", dtype=torch.float64):
        super().__init__()

        self.device=device
        self.dtype=dtype

        assert len(U_tensors.keys()) == 3, "U_tensors must contain only 3 keys."
        assert len(W_tensors.keys()) == 3, "W_tensors must contain only 3 keys."

        for key in U_tensors.keys():
            if (not U_tensors[key].is_cuda):
                U_tensors[key] = U_tensors[key].cuda()

            assert U_tensors[key].dtype == self.dtype, f"U_tensor[{key}] dtype: {U_tensors[key].dtype} is not the same as self.dtype {self.dtype}"

            if (not W_tensors[key].is_cuda):
                W_tensors[key] = W_tensors[key].cuda()

            assert W_tensors[key].dtype == self.dtype, f"W_tensor[{key}] dtype: {W_tensors[key].dtype} is not the same as self.dtype {self.dtype}"

        nl_3 = U_tensors[3].shape[0]

        self.U3_num_nonsparse = torch.zeros((nl_3, nl_3), dtype=torch.uint8).cuda()

        for i in range(nl_3):
            for j in range(nl_3):
                kdx1, ldx1  = torch.where(U_tensors[3][i, j] != 0.0)
                self.U3_num_nonsparse[i, j] = kdx1.shape[0]

        self.U3_nonsparse_indices = torch.zeros((nl_3,nl_3, 4, self.U3_num_nonsparse.max().item()), dtype=torch.uint8).cuda()
        self.U3_nonsparse_elements = torch.zeros((nl_3, nl_3, 4), dtype=torch.float).cuda()
        self.U2_nonsparse_indices = torch.zeros((nl_3,nl_3), dtype=torch.uint8).cuda()
        self.U2_nonsparse_elements = torch.zeros((nl_3, nl_3), dtype=torch.float).cuda()

        for i in range(nl_3):
            for j in range(nl_3):
                
                kdx1, ldx1  = torch.where(U_tensors[3][i, j] != 0.0)
                kdx2, ldx2  = torch.where(U_tensors[3][j, :, i, :] != 0.0)
                kdx3, ldx3  = torch.where(U_tensors[3][j, i] != 0.0)

                for k in range(kdx1.shape[0]):
                    self.U3_nonsparse_indices[i, j, 0, k] = kdx1[k]
                    self.U3_nonsparse_indices[i, j, 1, k] = ldx1[k] # ijk ldx1
                    
                    # additional derivative indices
                    self.U3_nonsparse_indices[i, j, 2, k] = ldx2[k] # ikj ldx2
                    self.U3_nonsparse_indices[i, j, 3, k] = ldx3[k] # jik ldx3

                    self.U3_nonsparse_elements[i, j, k] = U_tensors[3][i, j, kdx1[k], ldx1[k]]

        for i in range (U_tensors[2].shape[0]):

            jdx, kdx  = torch.where(U_tensors[2][i] != 0.0)

            if (jdx.shape[0] > 0):
                self.U2_nonsparse_indices[i, jdx] = kdx.type(torch.uint8)
                self.U2_nonsparse_elements[i, jdx] = U_tensors[2][i, jdx, kdx]

        self.U1 = U_tensors[1]

        self.W3 = W_tensors[3]
        self.W2 = W_tensors[2]
        self.W1 = W_tensors[1]

    def forward(self, x, atom_types):

        assert x.shape[-1] % 32 == 0, "channel dimension of x ([-1]) must be a multiple of 32."

        nthreads = 32

        if (x.shape[-1] == 96):
            nthreads = 32
        elif (x.shape[-1] >= 64):
            nthreads = 64

        return _symmetric_contraction(
                x, 
                atom_types, 
                self.U3_nonsparse_indices, 
                self.U3_num_nonsparse, 
                self.U3_nonsparse_elements, 
                self.U2_nonsparse_indices, 
                self.U2_nonsparse_elements, 
                self.U1, 
                self.W3, 
                self.W2, 
                self.W1, 
                nthreads,
                16,
                1)
