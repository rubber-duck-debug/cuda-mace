#ifndef CUBIC_SPLINE_IMPL_CUH
#define CUBIC_SPLINE_IMPL_CUH

#include <torch/script.h>
#include <vector>

using namespace std;
using namespace torch;

std::vector<torch::Tensor> evaluate_spline(
    torch::Tensor r,
    torch::Tensor coeffs,
    torch::Tensor r_width);

torch::Tensor generate_coefficients(
    torch::Tensor r,
    torch::Tensor R,
    torch::Tensor r_width);

torch::Tensor backward_spline(torch::Tensor grad_output, torch::Tensor R_deriv);

#endif