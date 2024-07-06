#include "cubic_spline.h"
#include "cubic_spline_impl.cuh"
#include "torch_utils.cuh"

#include <iostream>
#include <torch/script.h>
#include <torch/serialize/archive.h>

using namespace std;
using namespace torch::indexing;
using namespace torch::autograd;

torch::Tensor CubicSplineAutograd::forward(AutogradContext *ctx,
                                           torch::Tensor r,
                                           torch::Tensor coeffs, double r_width,
                                           double r_max) {

  auto result = evaluate_spline(r, coeffs, r_width, r_max);

  if (r.requires_grad()) {
    ctx->save_for_backward({result[1]});
  }

  return result[0];
}

variable_list CubicSplineAutograd::backward(AutogradContext *ctx,
                                            variable_list grad_outputs) {
  auto saved_variables = ctx->get_saved_variables();

  torch::Tensor R_deriv = saved_variables[0];

  torch::Tensor result = backward_spline(grad_outputs[0].contiguous(), R_deriv);

  torch::Tensor undef;

  return {result, undef, undef, undef};
}

CubicSpline::CubicSpline(torch::Tensor r_basis, torch::Tensor R, double r_width,
                         double r_max) {

  this->coeffs = generate_coefficients(r_basis, R, r_width);

  this->r_width = r_width;

  this->r_max = r_max;
}

// wrapper class which we expose to the API.
torch::Tensor CubicSpline::forward(torch::Tensor r) {
  return CubicSplineAutograd::apply(r, this->coeffs, this->r_width,
                                    this->r_max);
}

torch::Tensor CubicSpline::get_coefficients() { return this->coeffs; }

// Method to save the state
std::vector<torch::Tensor> CubicSpline::__getstate__() const {
  return {this->coeffs};
}

// Method to load the state
void CubicSpline::__setstate__(const std::vector<torch::Tensor> &state) {
  this->coeffs = state[0];
}

TORCH_LIBRARY(cubic_spline, m) {
  m.class_<CubicSpline>("CubicSpline")
      .def(torch::init<torch::Tensor, torch::Tensor, double, double>())

      .def("forward", &CubicSpline::forward)
      .def("get_coefficients", &CubicSpline::get_coefficients)
      .def_pickle(
          [](const c10::intrusive_ptr<CubicSpline> &self)
              -> std::vector<torch::Tensor> { return self->__getstate__(); },
          [](const std::vector<torch::Tensor> &state)
              -> c10::intrusive_ptr<CubicSpline> {
            auto obj = c10::make_intrusive<CubicSpline>();
            obj->__setstate__(state);
            return obj;
          });
}