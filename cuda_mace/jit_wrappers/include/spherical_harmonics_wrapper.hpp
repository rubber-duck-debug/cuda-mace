#ifndef SPHERICAL_HARMONICS_WRAPPER_HPP
#define SPHERICAL_HARMONICS_WRAPPER_HPP

#include <torch/script.h>
#include <vector>

using namespace std;
using namespace torch;

std::vector<torch::Tensor>
jit_spherical_harmonics(torch::Tensor rij);

torch::Tensor jit_spherical_harmonics_backward(torch::Tensor dsph, torch::Tensor grad_outputs);

#endif // INVARIANT_MESSAGE_PASSING_WRAPPER_HPP