#ifndef TORCH_UTILS_CUH
#define TORCH_UTILS_CUH

#include <torch/script.h>

using namespace torch;

torch::Tensor calculate_first_occurences_gpu(torch::Tensor receiver_list, int64_t natoms, int64_t nthreadx);

torch::Tensor calculate_first_occurences_gpu_with_sort(torch::Tensor receiver_list, int64_t natoms, int64_t nthreadx, torch::Tensor sort_indices);

#endif // TORCH_UTILS_CUH