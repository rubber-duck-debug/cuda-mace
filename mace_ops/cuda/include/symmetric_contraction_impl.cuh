#ifndef SYMMETRIC_CONTRACTION_IMPL_CUH
#define SYMMETRIC_CONTRACTION_IMPL_CUH

#include <torch/script.h>
#include <vector>

using namespace torch;
using namespace std;

std::vector<torch::Tensor> symmetric_contraction_L0_forwards_gpu(
    torch::Tensor X,
    torch::Tensor atom_types,
    torch::Tensor U3_num_nonzero,
    torch::Tensor U3_indices,
    torch::Tensor U3_values,
    torch::Tensor U2_num_nonzero,
    torch::Tensor U2_indices,
    torch::Tensor U2_values,
    torch::Tensor U1_num_nonzero,
    torch::Tensor U1_indices,
    torch::Tensor W3,
    torch::Tensor W2,
    torch::Tensor W1,
    const int w3_size,
    const int w2_size,
    const int w1_size,
    torch::Tensor u3_n_nonsparse,
    const int64_t nthreadX,
    const int64_t nthreadY,
    const int64_t nthreadZ);

std::vector<torch::Tensor> symmetric_contraction_LGT0_forwards_gpu(
    torch::Tensor X,
    torch::Tensor atom_types,
    torch::Tensor U3_num_nonzero_1,
    torch::Tensor U3_num_nonzero_2,
    torch::Tensor U3_num_nonzero_3,
    torch::Tensor U3_indices_0,
    torch::Tensor U3_indices_1,
    torch::Tensor U3_indices_2,
    torch::Tensor U3_indices_3,
    torch::Tensor U3_values_0,
    torch::Tensor U3_values_1,
    torch::Tensor U3_values_2,
    torch::Tensor U3_values_3,
    torch::Tensor U2_num_nonzero_1,
    torch::Tensor U2_num_nonzero_2,
    torch::Tensor U2_indices_1,
    torch::Tensor U2_indices_2,
    torch::Tensor U2_values_1,
    torch::Tensor U2_values_2,
    torch::Tensor U1_num_nonzero,
    torch::Tensor U1_index,
    torch::Tensor W3,
    torch::Tensor W2,
    torch::Tensor W1,
    const int64_t W3_l0_size,
    const int64_t W2_l0_size,
    const int64_t W1_l0_size,
    torch::Tensor W3_size,          // nlout
    torch::Tensor W2_size,          // nlout
    torch::Tensor W1_size,          // nlout
    torch::Tensor U3_max_nonsparse, // nlout
    const int64_t nthreadX,
    const int64_t nthreadY,
    const int64_t nthreadZ);

torch::Tensor symm_contraction_backward(
    torch::Tensor gradX,
    torch::Tensor grad_input,
    int nthreadX,
    int nthreadY,
    int nthreadZ);

int64_t curr_shared_mem();

int64_t LGT0_shared_memory_required(int64_t nthreadsX, int64_t nthreadsY, int64_t nthreadsZ, int64_t u3_maxn_nonsparse, int64_t nl, int64_t W3_size, int64_t W2_size, int64_t W1_size, torch::ScalarType scalar_type);

bool set_shared_mem_size(int64_t amount, torch::ScalarType scalar_type);

#endif