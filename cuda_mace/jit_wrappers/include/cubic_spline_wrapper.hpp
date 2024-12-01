#ifndef CUBIC_SPLINE_WRAPPER_HPP
#define CUBIC_SPLINE_WRAPPER_HPP

#include <torch/script.h>
#include <vector>

std::vector<torch::Tensor> jit_evaluate_spline(torch::Tensor r,
                                           torch::Tensor r_knots,
                                           torch::Tensor coeffs, double r_width,
                                           double r_max);

torch::Tensor jit_backward_spline(torch::Tensor grad_output, torch::Tensor R_deriv);

#endif