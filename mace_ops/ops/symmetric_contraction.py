import torch
from e3nn import o3
from mace.tools.cg import U_matrix_real

class SymmetricContraction(torch.nn.Module):

    def __init__(self, irreps_in: o3.Irreps, irreps_out: o3.Irreps, W_tensors, correlation=3, nthreadX=32, nthreadY=4, nthreadZ=1, device="cuda", dtype=torch.float32):
        super().__init__()

        self.cuda_obj = torch.classes.symm_contract.SymmetricContraction()
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

        shared_mem_required = torch.ops.symm_contract.LGT0_shared_memory_required(
            nthreadX, nthreadY, nthreadZ,  
            self.u3_max_nonsparse_tensor.max().item(), 
            16,
            self.nweights_3[-1].item(), self.nweights_2[-1].item(), self.nweights_1[-1].item(), dtype)

        if (shared_mem_required > 49152):
            print ("adjusting shared memory to fit:", shared_mem_required, "bytes")
            if torch.ops.symm_contract.set_shared_mem_size(shared_mem_required, self.dtype):
                print("shared memory reallocation accepted")
            
    def forward(self, x, atom_types):

        assert x.shape[-1] % 32 == 0, "channel dimension of x ([-1]) must be a multiple of 32."
        assert x.shape[1] == 16, "l dimension of x ([1]) must be 16."
        
        return self.cuda_obj.forward(  
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