#ifndef SPHERICAL_HARMONICS_IMPL_CUH
#define SPHERICAL_HARMONICS_IMPL_CUH

#include <torch/script.h>
#include <vector>

using namespace std;
using namespace torch;

std::vector<torch::Tensor> spherical_harmonics(torch::Tensor xyz);

torch::Tensor spherical_harmonics_backward(torch::Tensor sph_deriv,
                                           torch::Tensor grad_output);

#endif