#ifndef LINEAR_WMMA_IMPL_CUH
#define LINEAR_WMMA_IMPL_CUH

#include <torch/script.h>

using namespace torch;

torch::Tensor linear_wmma(torch::Tensor X, torch::Tensor W);

torch::Tensor elemental_linear_wmma(torch::Tensor X, torch::Tensor W,
                                    torch::Tensor elemental_embedding);

#endif // LINEAR_WMMA_IMPL_CUH