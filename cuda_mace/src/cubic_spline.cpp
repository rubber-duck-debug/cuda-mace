#include "cubic_spline.h"
#include "cubic_spline_wrapper.hpp"

#include <iostream>
#include <torch/script.h>
#include <torch/serialize/archive.h>

using namespace std;
using namespace torch::indexing;
using namespace torch::autograd;

torch::Tensor CubicSplineAutograd::forward(AutogradContext *ctx,
                                           torch::Tensor r,
                                           torch::Tensor r_knots,
                                           torch::Tensor coeffs, double r_width,
                                           double r_max) {

  auto result = jit_evaluate_spline(r, r_knots, coeffs, r_width, r_max);

  if (r.requires_grad()) {
    ctx->save_for_backward({result[1]});
  }

  return result[0];
}

variable_list CubicSplineAutograd::backward(AutogradContext *ctx,
                                            variable_list grad_outputs) {
  auto saved_variables = ctx->get_saved_variables();

  torch::Tensor R_deriv = saved_variables[0];

  torch::Tensor result = jit_backward_spline(grad_outputs[0].contiguous(), R_deriv);

  torch::Tensor undef;

  return {result, undef, undef, undef, undef};
}

// wrapper class which we expose to the API.
torch::Tensor CubicSpline::forward(torch::Tensor r, torch::Tensor r_knots,
                                   torch::Tensor coeffs, double r_width,
                                   double r_max) {
  return CubicSplineAutograd::apply(r, r_knots, coeffs, r_width, r_max);
}

TORCH_LIBRARY(cubic_spline, m) {
  m.class_<CubicSpline>("CubicSpline")
      .def(torch::init<>(), "", {})

      .def("forward", &CubicSpline::forward)
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