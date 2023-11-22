import os
import torch
import sysconfig

from mace.tools.cg import U_matrix_real
from e3nn import o3

from .instruction import Instruction, _normalize_instruction_path_weights
from .irreps import Irreps

_HERE = os.path.realpath(os.path.dirname(__file__))
EXT_SUFFIX = sysconfig.get_config_var('EXT_SUFFIX')

torch.ops.load_library(_HERE + '/tensor_product.so')
torch.ops.load_library(_HERE + '/symmetric_contraction.so')
torch.ops.load_library(_HERE + '/invariant_message_passing.so')
torch.ops.load_library(_HERE + '/equivariant_outer_product.so')
torch.ops.load_library(_HERE + '/linear.so')
torch.ops.load_library(_HERE + '/linear_wmma.so')
torch.ops.load_library(_HERE + '/matmul.so')

def _sum_weighted_tensor_product(
    X1,
    X2,
    weights,
    weight_index,
    receiver_list,
    num_nodes,
    avg_num_neighbours,
    nthreadx,
    nthready,
    nthreadz):

    return torch.ops.mace_cuda_tp.sum_weighted_tensor_product(
        X1,
        X2,
        weights,
        weight_index,
        receiver_list,
        num_nodes,
        avg_num_neighbours,
        nthreadx,
        nthready,
        nthreadz)

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
            X,
            atom_types,
            U3_num_nonsparse_1,
            U3_num_nonsparse_2,
            U3_num_nonsparse_3,
            U3_indices_0,
            U3_indices_1,
            U3_indices_2,
            U3_indices_3,
            U3_values_0,
            U3_values_1,
            U3_values_2,
            U3_values_3,
            U2_num_nonsparse_1,
            U2_num_nonsparse_2,
            U2_indices_1,
            U2_indices_2,
            U2_values_1,
            U2_values_2,
            U1_num_nonsparse,
            U1_index,
            W3,
            W2,
            W1,
            W3_L0_size : int,
            W2_L0_size : int,
            W1_L0_size : int,
            W3_size,
            W2_size,
            W1_size,
            U3_max_nonsparse,
            nthreadX: int,
            nthreadY: int,
            nthreadZ: int) -> torch.Tensor:
    
    return torch.ops.mace_cuda_symm_contraction.symmetric_contraction(
            X,
            atom_types,
            U3_num_nonsparse_1,
            U3_num_nonsparse_2,
            U3_num_nonsparse_3,
            U3_indices_0,
            U3_indices_1,
            U3_indices_2,
            U3_indices_3,
            U3_values_0,
            U3_values_1,
            U3_values_2,
            U3_values_3,
            U2_num_nonsparse_1,
            U2_num_nonsparse_2,
            U2_indices_1,
            U2_indices_2,
            U2_values_1,
            U2_values_2,
            U1_num_nonsparse,
            U1_index,
            W3,
            W2,
            W1,
            W3_L0_size,
            W2_L0_size,
            W1_L0_size,
            W3_size,
            W2_size,
            W1_size,
            U3_max_nonsparse,
            nthreadX,
            nthreadY,
            nthreadZ)


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

    print (self.mu1)
    print (self.mu2)
    print (self.mu3)

    self.cg_coeffs = torch.cat(cg_coeffs).type(self.dtype).cuda()

    self.mu_1_sort = torch.argsort(self.mu1).type(torch.int16).cuda()
    self.mu_2_sort = torch.argsort(self.mu2).type(torch.int16).cuda()
    self.mu_3_sort = torch.argsort(self.mu3).type(torch.int16).cuda()


  def forward(self,x,y):

    assert x.dtype == self.dtype and y.dtype == self.dtype, f"x: {x.dtype} and y:{y.dtype} need to be the same as this class: {self.dtype}"
    
    nthreads = 96

    return _tensor_product(x, y, 
                          self.mu1, self.mu2, self.mu3,
                          self.mu_1_sort, self.mu_2_sort, self.mu_3_sort,
                          self.cg_coeffs,
                          self.irreps_out.dim, nthreads, 4, 1)

class SymmetricContraction(torch.nn.Module):

    def __init__(self, irreps_in: o3.Irreps, irreps_out: o3.Irreps, W_tensors, correlation=3, nthreadX=32, nthreadY=4, nthreadZ=1, device="cuda", dtype=torch.float64):
        super().__init__()

        self.device=device
        self.dtype=dtype
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        self.correlation = correlation

        self.nthreadX = nthreadX
        self.nthreadY = nthreadY
        self.nthreadZ = nthreadZ
        
        self.nlout = (irreps_out.lmax + 1) ** 2

        assert correlation == 3, "CUDASymmetricContraction exclusively supports correlation=3"

        self.U_matrices = {}

        for nu in range(1, correlation + 1):
            self.U_matrices[nu] = {}

            for ir_id, ir_out in enumerate(irreps_out):
                U_matrix = U_matrix_real(
                    irreps_in=irreps_in,
                    irreps_out=o3.Irreps(str(ir_out.ir)),
                    correlation=nu,
                    dtype=torch.float32
                )[-1].to(self.device)
                self.U_matrices[nu][ir_id] = U_matrix

        self.W_tensors = W_tensors

        self.setup_sparse_matrices()
        self.setup_weights()

        shared_mem_required = torch.ops.mace_cuda_symm_contraction.LGT0_shared_memory_required(
            nthreadX, nthreadY, nthreadZ,  
            self.u3_max_nonsparse_tensor.max().item(), 
            16,
            self.nweights_3[-1].item(), self.nweights_2[-1].item(), self.nweights_1[-1].item(), dtype)

        if (shared_mem_required > 49152):
            print ("adjusting shared memory to fit:", shared_mem_required, "bytes")
            for device_id in range(torch.cuda.device_count()):
                if torch.ops.mace_cuda_symm_contraction.set_shared_mem_size(shared_mem_required, device_id, self.dtype):
                    print("shared memory reallocation accepted on device:", device_id)
            
    def forward(self, x, atom_types):

        assert x.shape[-1] % 32 == 0, "channel dimension of x ([-1]) must be a multiple of 32."
        assert x.shape[1] == 16, "l dimension of x ([1]) must be 16."
        
        return _symmetric_contraction(  
                    x, 
                    atom_types,
                    self.U3_num_nonsparse_1,
                    self.U3_num_nonsparse_2,
                    self.U3_num_nonsparse_3,
                    self.U3_indices_0, # L=0 specific
                    self.U3_indices_1,
                    self.U3_indices_2,
                    self.U3_indices_3,
                    self.U3_values_0, # L=0 specific
                    self.U3_values_1,
                    self.U3_values_2,
                    self.U3_values_3,
                    self.U2_num_nonsparse_1,
                    self.U2_num_nonsparse_2,
                    self.U2_indices_1,
                    self.U2_indices_2,
                    self.U2_values_1,
                    self.U2_values_2,
                    self.U1_num_values,
                    self.U1_index,
                    self.weights_3,
                    self.weights_2,
                    self.weights_1,
                    self.W3_L0_size,
                    self.W2_L0_size,
                    self.W1_L0_size,
                    self.nweights_3,
                    self.nweights_2,
                    self.nweights_1,
                    self.u3_max_nonsparse_tensor,
                    self.nthreadX,
                    self.nthreadY,
                    self.nthreadZ)
    
    def setup_sparse_matrices(self):
        lout_counter = 0
        self.U3_num_nonsparse_1 = torch.zeros((self.nlout, 16, 16), dtype=torch.int16).to(self.device)
        self.U3_num_nonsparse_2 = torch.zeros((self.nlout, 16, 16), dtype=torch.int16).to(self.device)
        self.U3_num_nonsparse_3 = torch.zeros((self.nlout, 16, 16), dtype=torch.int16).to(self.device)
        for ir_id, ir_out in enumerate(self.irreps_out):
            U_matrix = self.U_matrices[3][ir_id]
            if (len(U_matrix.shape) == 4):
                nl = U_matrix.shape[0]
                for i in range(nl):
                        for j in range(nl):
                            kdx1, ldx1  = torch.where(U_matrix[i, j] != 0.0)
                            kdx2, ldx2  = torch.where(U_matrix[j, i, :, :] != 0.0)
                            kdx3, ldx3  = torch.where(U_matrix[j, :, i, :] != 0.0)
                            if (kdx1.shape[0] > 0):
                                self.U3_num_nonsparse_1[lout_counter, i, j] = kdx1.shape[0]
                            if (kdx2.shape[0] > 0):
                                self.U3_num_nonsparse_2[lout_counter, i, j] = kdx2.shape[0]
                            if (kdx3.shape[0] > 0):   
                                self.U3_num_nonsparse_3[lout_counter, i, j] = kdx3.shape[0]
                lout_counter +=1
            else:
                nl = U_matrix.shape[1]
                for a in range(U_matrix.shape[0]):
                    for i in range(nl):
                        for j in range(nl):
                            kdx1, ldx1  = torch.where(U_matrix[a, i, j] != 0.0)
                            kdx2, ldx2  = torch.where(U_matrix[a, j, i] != 0.0)
                            kdx3, ldx3  = torch.where(U_matrix[a, j, :, i, :] != 0.0)
                            self.U3_num_nonsparse_1[lout_counter, i, j] = kdx1.shape[0]
                            self.U3_num_nonsparse_2[lout_counter, i, j] = kdx2.shape[0]
                            self.U3_num_nonsparse_3[lout_counter, i, j] = kdx3.shape[0]
                    lout_counter+=1

        self.u3_max_nonsparse = torch.max( torch.tensor([self.U3_num_nonsparse_1.max().item(), self.U3_num_nonsparse_2.max().item(), self.U3_num_nonsparse_3.max().item()])).item()

        lout_counter = 0
        self.U3_indices_0 = torch.zeros((self.u3_max_nonsparse, 16, 16), dtype=torch.int32).to(self.device)
        self.U3_values_0 = torch.zeros((self.u3_max_nonsparse, 16, 16), dtype=torch.float32).to(self.device)
        self.U3_indices_1  = torch.zeros((self.nlout, self.u3_max_nonsparse, 16, 16), dtype=torch.int16).to(self.device)
        self.U3_indices_2  = torch.zeros((self.nlout, self.u3_max_nonsparse, 16, 16), dtype=torch.int16).to(self.device)
        self.U3_indices_3  = torch.zeros((self.nlout, self.u3_max_nonsparse, 16, 16), dtype=torch.int16).to(self.device)
        self.U3_values_1 = torch.zeros((self.nlout, self.u3_max_nonsparse, 16, 16), dtype=torch.float32).to(self.device)
        self.U3_values_2 = torch.zeros((self.nlout, self.u3_max_nonsparse, 16, 16), dtype=torch.float32).to(self.device)
        self.U3_values_3 = torch.zeros((self.nlout, self.u3_max_nonsparse, 16, 16), dtype=torch.float32).to(self.device)
        for ir_id, ir_out in enumerate(self.irreps_out):
            U_matrix = self.U_matrices[3][ir_id]
            if (len(U_matrix.shape) == 4):
                nl = U_matrix.shape[0]
                for i in range(nl):
                    for j in range(nl):
                        kdx1, ldx1  = torch.where(U_matrix[i, j] != 0.0)
                        _, ldx2  = torch.where(U_matrix[j, i, :, :] != 0.0)
                        _, ldx3  = torch.where(U_matrix[j, :, i, :] != 0.0)
                        for k in range(kdx1.shape[0]):
                            compressed_output1 = kdx1[k] << 8 | ldx1[k]
                            compressed_output2 = ldx2[k] << 8 | ldx3[k]
                            compressed_output = compressed_output2 << 16 | compressed_output1
                            self.U3_indices_0[k, i, j] = compressed_output
                            self.U3_values_0[k, i, j] = U_matrix[i, j, kdx1[k], ldx1[k]]
                lout_counter +=1
            else:
                nl = U_matrix.shape[1]
                for a in range(U_matrix.shape[0]):
                    for i in range(nl):
                        for j in range(nl):
                            kdx1, ldx1  = torch.where(U_matrix[a, i, j] != 0.0)
                            kdx2, ldx2  = torch.where(U_matrix[a, j, i] != 0.0)
                            kdx3, ldx3  = torch.where(U_matrix[a, j, :, i, :] != 0.0)
                            for k in range (self.U3_num_nonsparse_1[lout_counter, i, j]):
                                compressed_index = kdx1[k] << 8 | ldx1[k]
                                self.U3_indices_1[lout_counter, k, i, j] = compressed_index
                                self.U3_values_1[lout_counter, k, i, j] = U_matrix[a, i, j, kdx1[k], ldx1[k]]
                            for k in range (self.U3_num_nonsparse_2[lout_counter, i, j]):
                                compressed_index = kdx2[k] << 8 | ldx2[k]
                                self.U3_indices_2[lout_counter, k, i, j] = compressed_index
                                self.U3_values_2[lout_counter, k, i, j] = U_matrix[a, j, i, kdx2[k], ldx2[k]]
                            for k in range (self.U3_num_nonsparse_3[lout_counter, i, j]):
                                compressed_index = kdx3[k] << 8 | ldx3[k]
                                self.U3_indices_3[lout_counter, k, i, j] = compressed_index
                                self.U3_values_3[lout_counter, k, i, j] = U_matrix[a, j, kdx3[k], i, ldx3[k]]

                    lout_counter+=1

        lout_counter = 0
        self.U2_num_nonsparse_1 = torch.zeros((self.nlout, 16, 16), dtype=torch.int16).to(self.device)
        self.U2_num_nonsparse_2 = torch.zeros((self.nlout, 16, 16), dtype=torch.int16).to(self.device)
        for ir_id, ir_out in enumerate(self.irreps_out):
            U_matrix = self.U_matrices[2][ir_id]
            if (len(U_matrix.shape) == 3):
                nl = U_matrix.shape[0]
                for i in range(nl):
                    for j in range (nl):
                            jdx,  = torch.where(U_matrix[i, j] != 0.0)
                            self.U2_num_nonsparse_1[lout_counter, i, j] = jdx.shape[0]
                            jdx,  = torch.where(U_matrix[j, i] != 0.0)
                            self.U2_num_nonsparse_2[lout_counter, i, j] = jdx.shape[0]
                lout_counter +=1
            else:
                nl = U_matrix.shape[1]
                for a in range(U_matrix.shape[0]):
                    for i in range(nl):
                        for j in range (nl):
                            jdx,  = torch.where(U_matrix[a, i, j] != 0.0)
                            self.U2_num_nonsparse_1[lout_counter, i, j] = jdx.shape[0]
                            jdx,  = torch.where(U_matrix[a, j, i] != 0.0)
                            self.U2_num_nonsparse_2[lout_counter, i, j] = jdx.shape[0]
                    lout_counter+=1

        lout_counter = 0
        self.u2_max_nonsparse = torch.max( torch.tensor([self.U2_num_nonsparse_1.max().item(), self.U2_num_nonsparse_2.max().item()])).item()
        self.U2_values_1 = torch.zeros((self.nlout, 16, 16), dtype=torch.float32).to(self.device)
        self.U2_indices_1 = torch.zeros((self.nlout, 16, 16), dtype=torch.int16).to(self.device)
        self.U2_values_2 = torch.zeros((self.nlout, 16, 16), dtype=torch.float32).to(self.device)
        self.U2_indices_2 = torch.zeros((self.nlout, 16, 16), dtype=torch.int16).to(self.device)
        for ir_id, ir_out in enumerate(self.irreps_out):
            U_matrix = self.U_matrices[2][ir_id]
            if (len(U_matrix.shape) == 3):
                nl = U_matrix.shape[0]
                for i in range(nl):
                    for j in range (nl):
                        kdx_1,  = torch.where(U_matrix[i, j] != 0.0)
                        kdx_2,  = torch.where(U_matrix[j, i] != 0.0)
                        if (self.U2_num_nonsparse_1[lout_counter, i, j] > 0):
                            self.U2_values_1[lout_counter, i, j] = U_matrix[ i, j, kdx_1]
                            self.U2_indices_1[lout_counter, i, j] = kdx_1
                        if (self.U2_num_nonsparse_2[lout_counter, i, j] > 0):
                            self.U2_values_2[lout_counter, i, j] = U_matrix[ j, i, kdx_2]
                            self.U2_indices_2[lout_counter, i, j] = kdx_2
                lout_counter +=1
            else:
                for a in range(U_matrix.shape[0]):
                    nl = U_matrix[a].shape[0]
                    for i in range(nl):
                        for j in range (nl):
                            kdx_1,  = torch.where(U_matrix[a, i, j] != 0.0)
                            kdx_2,  = torch.where(U_matrix[a, j, i] != 0.0)
                            if (self.U2_num_nonsparse_1[lout_counter, i, j] > 0):
                                self.U2_values_1[lout_counter, i, j] = U_matrix[a, i, j, kdx_1]
                                self.U2_indices_1[lout_counter, i, j] = kdx_1

                            if (self.U2_num_nonsparse_2[lout_counter, i, j] > 0):
                                self.U2_values_2[lout_counter, i, j] = U_matrix[a, j, i, kdx_2]
                                self.U2_indices_2[lout_counter, i, j] = kdx_2

                    lout_counter+=1

        lout_counter = 0
        self.U1_num_values = torch.zeros((self.nlout, 16), dtype=torch.int16).to(self.device)
        self.U1_index = torch.zeros((self.nlout, 16), dtype=torch.int16).to(self.device)
        for ir_id, ir_out in enumerate(self.irreps_out):
            U_matrix =  self.U_matrices[1][ir_id]
            if (len(U_matrix.shape) == 2):
                nl = U_matrix.shape[0]
                for i in range(nl):
                    jdx,  = torch.where(U_matrix[i] != 0.0)
                    if (jdx.shape[0] > 0):
                        self.U1_num_values[lout_counter, i] = 1
                        self.U1_index[lout_counter, i] = jdx
                lout_counter +=1
            else:
                for a in range(U_matrix.shape[0]):
                    for i in range(nl):
                        jdx,  = torch.where(U_matrix[a, i] != 0.0)
                        if (jdx.shape[0] > 0):
                            self.U1_num_values[lout_counter, i] = 1
                            self.U1_index[lout_counter, i] = jdx
                    lout_counter+=1

        self.u3_max_nonsparse_tensor = torch.zeros(self.nlout, device='cpu', dtype=int) 
        for i in range (self.U3_indices_1.shape[0]):
            self.u3_max_nonsparse_tensor[i] = torch.max( torch.tensor([self.U3_num_nonsparse_1[i].max().item(), self.U3_num_nonsparse_2[i].max().item(), self.U3_num_nonsparse_3[i].max().item()])).item()


    def setup_weights(self):
        self.weight_max_size = {}

        for c in self.W_tensors.keys():
            l = int(c)
            nl = 2 * l + 1
            for contraction in self.W_tensors[c].keys():
                if contraction not in self.weight_max_size or self.weight_max_size[contraction] < self.W_tensors[c][contraction].shape[-2]:
                    self.weight_max_size[contraction] = self.W_tensors[c][contraction].shape[-2]

        weights_dict = {}
        nweights ={}

        nelements = self.W_tensors[str(0)][3].shape[0]

        for nu in range(1, self.correlation + 1):
            weights_c = torch.zeros(self.nlout, nelements, self.weight_max_size[nu], self.W_tensors[str(0)][3].shape[-1], device=self.device, dtype=self.dtype)  
            weights_dict[str(nu)] = weights_c
            nweights[str(nu)] = torch.zeros(self.nlout, device=self.device, dtype=torch.int32)

        count = 0
        for i in range(len(self.W_tensors.keys())):
            nl_outputs = 2 * i + 1
            for j in range (nl_outputs):
                for nu in self.W_tensors[str(i)].keys():
                    nweights[str(nu)][count +j] = self.W_tensors[str(i)][nu].shape[-2]
                    weights_dict[str(nu)][count + j, :, :self.W_tensors[str(i)][nu].shape[-2], :] = self.W_tensors[str(i)][nu]
            count += nl_outputs

        self.weights_3 = weights_dict[str(3)]
        self.weights_2 = weights_dict[str(2)]
        self.weights_1 = weights_dict[str(1)]

        self.nweights_3 = nweights[str(3)]
        self.nweights_2 = nweights[str(2)]
        self.nweights_1 = nweights[str(1)]

        self.W3_L0_size = self.nweights_3[0].item()
        self.W2_L0_size = self.nweights_2[0].item()
        self.W1_L0_size = self.nweights_1[0].item()