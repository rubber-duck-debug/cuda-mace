#ifndef SPHERICAL_HARMONICS_H
#define SPHERICAL_HARMONICS_H
#include <torch/script.h>
#include <iostream>

using namespace std;
using namespace torch::indexing;
using namespace torch::autograd;

class SphericalHarmonicsAutograd : public Function<SphericalHarmonicsAutograd>
{
public:
    static torch::Tensor forward(
        AutogradContext *ctx,
        torch::Tensor xyz);

    static variable_list backward(AutogradContext *ctx, variable_list grad_outputs);
};

class SphericalHarmonics : public torch::CustomClassHolder
{

public:
    torch::Tensor forward(
        torch::Tensor xyz);

    std::vector<torch::Tensor> __getstate__()
    {
        return {};
    }

    void __setstate__(const std::vector<torch::Tensor> &state)
    {
        return;
    }
};

#endif