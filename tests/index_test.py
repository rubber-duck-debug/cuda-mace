
import torch
from e3nn import o3
from mace_ops.cuda.instruction import Instruction, _normalize_instruction_path_weights
from mace_ops.cuda.irreps import Irreps
import numpy as np
from time import time

from mace.modules.irreps_tools import (
    tp_out_irreps_with_instructions
)

from mace.tools.scatter import scatter_sum


class TensorProduct(torch.nn.Module):

    def __init__(self, irreps_in1, irreps_in2, target_irreps, device="cuda", dtype=torch.float64):
        super().__init__()

        self.irreps_in1 = o3.Irreps(irreps_in1)
        self.irreps_in2 = o3.Irreps(irreps_in2)
        self.target_irreps = o3.Irreps(target_irreps)

        self.device = device
        self.dtype = dtype

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

        print(self.irreps_out)

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

        self.mu_list = []
        self.cg_sparse_list = []

        for i, ins in enumerate(self.instructions):

            l1 = self.irreps_in1[ins.i_in1].ir.l
            l2 = self.irreps_in2[ins.i_in2].ir.l
            l3 = self.irreps_out[ins.i_out].ir.l

            offset1 = self.irreps_in1[: ins.i_in1].dim
            offset2 = self.irreps_in2[: ins.i_in2].dim
            offset3 = self.irreps_out[: ins.i_out].dim

            print(i, ins)
            print(l1, l2, l3)
            print(ins.i_in1, ins.i_in2, ins.i_out)

            cg = o3.wigner_3j(l1, l2, l3).to(self.device).type(dtype)

            # normalisation and weighting:
            cg = cg * ins.path_weight

            mu1, mu2, mu3 = cg.nonzero(as_tuple=True)

            cg_sparse = cg[(mu1, mu2, mu3)]

            self.mu_list += [(mu1, mu2, mu3)]
            self.cg_sparse_list += [cg_sparse]

            mu1 = mu1 + offset1
            mu2 = mu2 + offset2
            mu3 = mu3 + offset3

            mu_1.append(mu1)
            mu_2.append(mu2)
            mu_3.append(mu3)

            cg_coeffs.append(cg_sparse)

        self.mu1 = torch.cat(mu_1).cuda().type(torch.int32)
        self.mu2 = torch.cat(mu_2).cuda().type(torch.int32)
        self.mu3 = torch.cat(mu_3).cuda().type(torch.int32)

        # count the size of the output
        nmax1 = self.mu1.max().item() + 1
        nmax2 = self.mu2.max().item() + 1
        nmax3 = self.mu3.max().item() + 1

        self.nmax3 = nmax3

        print("nmax:", nmax1, nmax2, nmax3)

        self.cg_coeffs = torch.cat(cg_coeffs).type(self.dtype).cuda()

        self.mu_1_sort = torch.argsort(
            self.mu1).type(torch.int32).cuda().long()
        self.mu_2_sort = torch.argsort(
            self.mu2).type(torch.int32).cuda().long()
        self.mu_3_sort = torch.argsort(
            self.mu3).type(torch.int32).cuda().long()

    def forward(self, x, y):
        all_outputs = []

        for i in range(x.shape[0]):  # loop over edges
            outputs = []

            for ins, (mu1, mu2, mu3), cg_sparse in zip(self.instructions, self.mu_list, self.cg_sparse_list):

                mu1 = mu1.cuda()
                mu2 = mu2.cuda()
                mu3 = mu3.cuda()
                cg_sparse = cg_sparse.cuda()

                ir_in1 = self.irreps_in1[ins.i_in1].ir
                ir_in2 = self.irreps_in2[ins.i_in2].ir
                ir_out = self.irreps_out[ins.i_out].ir

                offset1 = self.irreps_in1[: ins.i_in1].dim
                offset2 = self.irreps_in2[: ins.i_in2].dim
                offset3 = self.irreps_out[: ins.i_out].dim

                cg_iteration = x[i, offset1 + mu1, :] * \
                    cg_sparse[:, None] * y[i, offset2 + mu2, :]

                output = torch.zeros(
                    ir_out.dim, x.shape[2], device=self.device, dtype=self.dtype)
                output.index_add_(0, mu3, cg_iteration)

                assert len(outputs) == ins.i_out, (len(outputs), ins.i_out)

                outputs.append(output)

            output_i = torch.cat(outputs, dim=0)

            all_outputs.append(output_i)

        return torch.stack(all_outputs)

def find_pivot_locations(numbers, num_pivots):
    total_sum = sum(numbers)
    pivot_indices = []

    # Calculate the target sum for each partition
    target_sum = total_sum / (num_pivots + 1)
    
    current_sum = 0
    current_partition_sum = 0
    pivot_index = 0
    
    for i, num in enumerate(numbers):
        current_sum += num
        current_partition_sum += num
        
        if current_partition_sum >= target_sum:
            # Adjust the pivot point if it's closer to the previous or current element
            if abs(current_partition_sum - target_sum) < abs(current_partition_sum - num - target_sum):
                pivot_index = i
                current_partition_sum = num
            else:
                current_partition_sum -= num
                
            pivot_indices.append(pivot_index)
            num_pivots -= 1
            
            if num_pivots == 0:
                break
    
    return pivot_indices


def maximum_partition(sequence, M, nr_partitions, sum_array):
    for n in range(2, len(sequence) + 1):
        for k in range(2, nr_partitions + 1):
            array = []
            for i in range(1, n + 1):
                select = max(M[i][k - 1], sum_array[n - 1] - sum_array[i - 1])
                array.append(select)
            M[n][k] = min(array)
    return M[len(sequence)][nr_partitions]

def init_matrix(sequence, nr_partitions, M, sum_array):
    for index in range(len(sequence)):
        sum_array.append(sum(sequence[: index + 1]))
    for k in range(1, nr_partitions + 1):
        M[1][k] = sequence[0]
    for n in range(1, len(sequence) + 1):
        M[n][1] = sum(sequence[:n])

def find_partitions(elements, npartitions):

    M = np.zeros((len(nelements) + 1, partitions + 1), dtype=int)
    sum_array = []
    init_matrix(nelements, partitions, M, sum_array)
    # call the main function
    range_sum_max = maximum_partition(nelements, M, partitions, sum_array)
    print("Sum of the maximum range:", range_sum_max)
    # split the sequence by using maximum sum of one range
    current_sum = 0

    all_partitions = []
    sub_partition = []
    for index in range(len(nelements)):
        if (current_sum + nelements[index]) > range_sum_max:
            print("| ", end="")
            current_sum = 0
            all_partitions.append(sub_partition)
            sub_partition = []
        current_sum += nelements[index]
        sub_partition.append(nelements[index])
        print(nelements[index], end=" ")
    all_partitions.append(sub_partition)
    print()

    return all_partitions



if __name__ == "__main__":

    torch.set_printoptions(edgeitems=3)

    dtype = torch.float32

    benchmark = False

    nedges = 30000
    nnodes = 1000
    nfeatures = 96

    irreps1, irreps2, target_irreps = (
        o3.Irreps(f"0e + 1o"),
        o3.Irreps("0e + 1o + 2e + 3o"),
        o3.Irreps(f"0e + 1o + 2e + 3o"),
    )

    print("nnodes: ", nnodes)
    print("nfeatures: ", nfeatures)
    print("irreps1: ", irreps1, irreps1.dim)
    print("irreps2: ", irreps2, irreps2.dim)
    print("target_irreps: ", target_irreps, target_irreps.dim)
    print("dtype: ", dtype)

    indices = torch.sort(torch.randint(nnodes, (nedges,), device='cuda'))[0]

    indices_cuda = indices.cuda().int()

    neighbour_cuda = torch.ops.mace_ops_equivariant_tp.calculate_neighbours(
        indices_cuda, nnodes, 64)

    X = torch.randn(nedges, irreps1.dim, nfeatures,
                    requires_grad=True, device='cuda', dtype=dtype)
    Y = torch.randn(nedges, irreps2.dim, requires_grad=True,
                    device='cuda', dtype=dtype)

    tp_cuda = TensorProduct(
        irreps1, irreps2, target_irreps, device="cuda", dtype=dtype)

    mu1 = tp_cuda.mu1[tp_cuda.mu_3_sort]
    mu2 = tp_cuda.mu2[tp_cuda.mu_3_sort]
    mu3 = tp_cuda.mu3[tp_cuda.mu_3_sort]
    cg_coeffs = tp_cuda.cg_coeffs[tp_cuda.mu_3_sort]

    # print (mu1)
    # print (mu2)
    # print (mu3)

    last_val = 0
    last_idx = 0

    nelements = []
    for i in range(len(mu3)):
        if (mu3[i] != last_val):
            nelements.append(i - last_idx)
            last_idx = i
            last_val = mu3[i]
    
    nelements.append(len(mu3)-last_idx)

    print (nelements, np.sum(nelements), len(mu3))
    print (mu3)
    nwarps = 4

    avg_elements_per_warp= len(mu3) / nwarps

    local_sum = 0.0

    starts = [0]
    for i in range(len(nelements)):

        if (local_sum > avg_elements_per_warp):
            
            idx_start = np.sum(nelements[:i])

            print (i, idx_start, mu3[idx_start -1].item(), mu3[idx_start].item(), mu3[idx_start+1].item())

            starts.append(idx_start)

            local_sum = 0.0

        local_sum += nelements[i]

    print (starts)



    indices_start = torch.tensor([0, 22, 44, 65]).int().cuda()
    nwork = torch.tensor([22, 22, 21, 21]).int().cuda()

    partitions = 4
    
    partitions = find_partitions(nelements, partitions)

    print (partitions)
    
    indices = []
    nwork = []
    idx_start = 0
    for i, partition in enumerate(partitions):
        
        indices.append(idx_start)
        
        print (idx_start, mu3[idx_start-1].item(), mu3[idx_start].item(), mu3[idx_start+1].item())
        
        sum_partition = np.sum(partition)

        nwork.append(sum_partition)
        idx_start += np.sum(partition)

    print (indices)
    print (nwork)

    indices_start = torch.tensor(indices).int().cuda()
    nwork = torch.tensor(nwork).int().cuda()

    start = time()
    for i in range(1000):
        out = torch.ops.mace_ops_equivariant_tp.equivariant_outer_product_forward(
            X,
            Y,
            indices_cuda,
            neighbour_cuda,
            mu1,
            mu2,
            mu3,
            cg_coeffs,
            indices_start,
            nwork,
            tp_cuda.nmax3,
            nnodes,
            32, 4, 1)
    torch.cuda.synchronize()
    end = time()
    print(end - start)

    

