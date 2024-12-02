#ifndef CUBIC_SPLINE_H
#define CUBIC_SPLINE_H
#include <iostream>
#include <torch/script.h>

using namespace std;
using namespace torch::indexing;
using namespace torch::autograd;

class CubicSplineAutograd : public Function<CubicSplineAutograd> {
public:
  static torch::Tensor forward(AutogradContext *ctx, torch::Tensor r,
                               torch::Tensor r_knots, torch::Tensor coeffs,
                               double r_width, double r_max);

  static variable_list backward(AutogradContext *ctx,
                                variable_list grad_outputs);
};

class CubicSpline : public torch::CustomClassHolder {

public:
  CubicSpline() {}

  torch::Tensor forward(torch::Tensor r, torch::Tensor r_knots,
                        torch::Tensor coeffs, double r_width, double r_ma);

  std::vector<torch::Tensor> __getstate__() { return {}; }

  void __setstate__(const std::vector<torch::Tensor> &state) { return; }
};

#endif