#ifndef CUBIC_SPLINE_H
#define CUBIC_SPLINE_H
#include <torch/script.h>
#include <iostream>

using namespace std;
using namespace torch::indexing;
using namespace torch::autograd;

class CubicSplineAutograd : public Function<CubicSplineAutograd>
{
public:
    static torch::Tensor forward(
        AutogradContext *ctx,
        torch::Tensor r,
        torch::Tensor coeffs,
        torch::Tensor r_width);

    static variable_list backward(AutogradContext *ctx, variable_list grad_outputs);
};

class CubicSpline : public torch::CustomClassHolder
{

public:
    CubicSpline() {}

    CubicSpline(torch::Tensor r_basis, torch::Tensor R);

    torch::Tensor forward(
        torch::Tensor r);

    torch::Tensor get_coefficients();

    std::vector<torch::Tensor> __getstate__()
    {
        return {};
    }

    void __setstate__(const std::vector<torch::Tensor> &state)
    {
        return;
    }

private:
    torch::Tensor r_width;
    torch::Tensor coeffs;
};

#endif