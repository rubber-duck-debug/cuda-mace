import torch
from e3nn import o3
from mace.tools.cg import U_matrix_real
from mace.modules.symmetric_contraction import SymmetricContraction

class SymmetricContraction(torch.nn.Module):

    def __init__(self, symm_contract: SymmetricContraction, device="cuda", dtype=torch.float32):
        super().__init__()

        self.cuda_obj = torch.classes.symm_contract.SymmetricContraction()
        self.device = device
        self.dtype = dtype
        self.irreps_in = o3.Irreps([irrep.ir for irrep in symm_contract.irreps_in])
        self.irreps_out = symm_contract.irreps_out
        self.correlation = 3

        self.nlout = (self.irreps_out.lmax + 1) ** 2

        assert self.correlation == 3, "CUDASymmetricContraction exclusively supports correlation=3"

        self.U_matrices = {}
        self.W_tensors = {}
        
        for nu in range(1, self.correlation + 1):
            self.U_matrices[nu] = {}

            for ir_id, ir_out in enumerate(self.irreps_out):
                U_matrix = U_matrix_real(
                    irreps_in=self.irreps_in,
                    irreps_out=o3.Irreps(str(ir_out.ir)),
                    correlation=nu,
                    dtype=torch.float32
                )[-1].to(self.device)
                self.U_matrices[nu][ir_id] = U_matrix

        for j in range(len(symm_contract.contractions)):
            self.W_tensors[str(j)] = {}
            self.W_tensors[str(j)][3] = (
                symm_contract.contractions[j]
                .weights_max.detach()
                .clone()
                .type(torch.float32)
            )

            self.W_tensors[str(j)][2] = (
                symm_contract.contractions[j]
                .weights[0]
                .detach()
                .clone()
                .type(torch.float32)
            )

            self.W_tensors[str(j)][1] = (
                symm_contract.contractions[j]
                .weights[1]
                .detach()
                .clone()
                .type(torch.float32)
            )


        self.setup_sparse_matrices()
        self.setup_weights()

    def forward(self, x, atom_types):

        assert x.shape[-1] % 32 == 0, "channel dimension of x ([-1]) must be a multiple of 32."
        assert x.shape[1] == 16, "l dimension of x ([1]) must be 16."

        return self.cuda_obj.forward(
            x,
            atom_types,
            self.u3_max_nonsparse,
            self.U3_num_nonsparse,
            self.U3_indices,
            self.U3_values,
            self.U2_num_nonsparse,
            self.U2_indices,
            self.U2_values,
            self.U1_num_values,
            self.U1_index,
            self.weights_3,
            self.weights_2,
            self.weights_1,
            self.W3_L0_size,
            self.W2_L0_size,
            self.W1_L0_size
            )

    def setup_sparse_matrices(self):
        lout_counter = 0
        U3_num_nonsparse = torch.zeros((16, 16), dtype=torch.int16).to(self.device)
      
        for ir_id, ir_out in enumerate(self.irreps_out):
            U_matrix = self.U_matrices[3][ir_id]
            if (len(U_matrix.shape) == 4):
                nl = U_matrix.shape[0]
                for i in range(nl):
                    for j in range(nl):
                        kdx1, ldx1 = torch.where(U_matrix[i, j] != 0.0)
                        if (kdx1.shape[0] > 0):
                            U3_num_nonsparse[i, j] = kdx1.shape[0]

        u3_max_nonsparse = U3_num_nonsparse.max().item()

        U3_indices = torch.zeros(
            (u3_max_nonsparse, 16, 16), dtype=torch.int32).to(self.device)
        U3_values = torch.zeros(
            (u3_max_nonsparse, 16, 16), dtype=torch.float32).to(self.device)

        for ir_id, ir_out in enumerate(self.irreps_out):
            U_matrix = self.U_matrices[3][ir_id]
            if (len(U_matrix.shape) == 4):
                nl = U_matrix.shape[0]
                for i in range(nl):
                    for j in range(nl):
                        kdx1, ldx1 = torch.where(U_matrix[i, j] != 0.0)
                        _, ldx2 = torch.where(U_matrix[j, i, :, :] != 0.0)
                        _, ldx3 = torch.where(U_matrix[j, :, i, :] != 0.0)
                        for k in range(kdx1.shape[0]):
                            compressed_output1 = kdx1[k] << 8 | ldx1[k]
                            compressed_output2 = ldx2[k] << 8 | ldx3[k]
                            compressed_output = compressed_output2 << 16 | compressed_output1
                            U3_indices[k, i, j] = compressed_output
                            U3_values[k, i, j] = U_matrix[i,
                                                            j, kdx1[k], ldx1[k]]

        U2_num_nonsparse = torch.zeros(
            (16, 16), dtype=torch.int16).to(self.device)

        for ir_id, ir_out in enumerate(self.irreps_out):
            U_matrix = self.U_matrices[2][ir_id]
            if (len(U_matrix.shape) == 3):
                nl = U_matrix.shape[0]
                for i in range(nl):
                    for j in range(nl):
                        jdx,  = torch.where(U_matrix[i, j] != 0.0)
                        U2_num_nonsparse[i, j] = jdx.shape[0]

        u2_max_nonsparse = torch.max(torch.tensor(
            [U2_num_nonsparse.max().item()])).item()

        U2_values = torch.zeros(
            (16, 16), dtype=torch.float32).to(self.device)
        U2_indices = torch.zeros(
            (16, 16), dtype=torch.int16).to(self.device)

        for ir_id, ir_out in enumerate(self.irreps_out):
            U_matrix = self.U_matrices[2][ir_id]
            if (len(U_matrix.shape) == 3):
                nl = U_matrix.shape[0]
                for i in range(nl):
                    for j in range(nl):
                        kdx_1,  = torch.where(U_matrix[i, j] != 0.0)
                        kdx_2,  = torch.where(U_matrix[j, i] != 0.0)
                        if (U2_num_nonsparse[i, j] > 0):
                            U2_values[ i,
                                        j] = U_matrix[i, j, kdx_1]
                            U2_indices[ i, j] = kdx_1

        U1_num_values = torch.zeros(
            (16), dtype=torch.int16).to(self.device)
        U1_index = torch.zeros(
            (16), dtype=torch.int16).to(self.device)
        for ir_id, ir_out in enumerate(self.irreps_out):
            U_matrix = self.U_matrices[1][ir_id]
            if (len(U_matrix.shape) == 2):
                nl = U_matrix.shape[0]
                for i in range(nl):
                    jdx,  = torch.where(U_matrix[i] != 0.0)
                    if (jdx.shape[0] > 0):
                        U1_num_values[i] = 1
                        U1_index[i] = jdx



        self.u3_max_nonsparse = u3_max_nonsparse
        self.register_buffer("U3_num_nonsparse", U3_num_nonsparse)
        self.register_buffer("U3_indices", U3_indices)
        self.register_buffer("U3_values", U3_values)
        self.register_buffer("U2_num_nonsparse", U2_num_nonsparse)
        self.register_buffer("U2_indices", U2_indices)
        self.register_buffer("U2_values", U2_values)

        self.register_buffer("U1_num_values", U1_num_values)
        self.register_buffer("U1_index", U1_index)

    def setup_weights(self):
        self.weight_max_size = {}

        for c in self.W_tensors.keys():
            l = int(c)
            nl = 2 * l + 1
            for contraction in self.W_tensors[c].keys():
                if contraction not in self.weight_max_size or self.weight_max_size[contraction] < self.W_tensors[c][contraction].shape[-2]:
                    self.weight_max_size[contraction] = self.W_tensors[c][contraction].shape[-2]

        weights_dict = {}
        nweights = {}

        nelements = self.W_tensors[str(0)][3].shape[0]

        for nu in range(1, self.correlation + 1):
            weights_c = torch.zeros(self.nlout, nelements, self.weight_max_size[nu], self.W_tensors[str(
                0)][3].shape[-1], device=self.device, dtype=self.dtype)
            weights_dict[str(nu)] = weights_c
            nweights[str(nu)] = torch.zeros(
                self.nlout, device=self.device, dtype=torch.int32)

        count = 0
        for i in range(len(self.W_tensors.keys())):
            nl_outputs = 2 * i + 1
            for j in range(nl_outputs):
                for nu in self.W_tensors[str(i)].keys():
                    nweights[str(nu)][count +
                                      j] = self.W_tensors[str(i)][nu].shape[-2]
                    weights_dict[str(nu)][count + j, :, :self.W_tensors[str(i)]
                                          [nu].shape[-2], :] = self.W_tensors[str(i)][nu]
            count += nl_outputs
        
        self.register_buffer("weights_3", weights_dict[str(3)])
        self.register_buffer("weights_2", weights_dict[str(2)])
        self.register_buffer("weights_1", weights_dict[str(1)])

        self.register_buffer("nweights_3", nweights[str(3)])
        self.register_buffer("nweights_2", nweights[str(2)])
        self.register_buffer("nweights_1", nweights[str(1)])
        
        self.W3_L0_size = self.nweights_3[0].item()
        self.W2_L0_size = self.nweights_2[0].item()
        self.W1_L0_size = self.nweights_1[0].item()