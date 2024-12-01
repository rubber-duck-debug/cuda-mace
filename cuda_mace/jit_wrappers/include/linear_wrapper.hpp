#ifndef LINEAR_WRAPPER_HPP
#define LINEAR_WRAPPER_HPP

#include <torch/script.h>

using namespace torch;

torch::Tensor jit_elemental_linear(torch::Tensor X, torch::Tensor W,
                                    torch::Tensor elemental_embedding);

torch::Tensor jit_linear(torch::Tensor X, torch::Tensor W);

#endif