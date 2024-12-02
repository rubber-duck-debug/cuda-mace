#ifndef LINEAR_WMMA_H
#define LINEAR_WMMA_H

#include <torch/script.h>

using namespace std;
using namespace torch;
using namespace torch::autograd;
using namespace torch::indexing;

class LinearAutograd : public Function<LinearAutograd>
{
public:
    static torch::Tensor forward(AutogradContext *ctx, torch::Tensor X,
                                 torch::Tensor W,
                                 torch::Tensor W_transposed);

    static torch::autograd::tensor_list backward(AutogradContext *ctx, variable_list grad_outputs);
};

class Linear : public torch::CustomClassHolder
{
public:
    torch::Tensor forward(
        torch::Tensor X,
        torch::Tensor W,
        torch::Tensor W_transposed);

    std::vector<torch::Tensor> __getstate__()
    {
        return {};
    }

    void __setstate__(const std::vector<torch::Tensor> &state)
    {
        return;
    }
};

class ElementalLinearAutograd : public Function<ElementalLinearAutograd>
{
public:
    static torch::Tensor forward(
        AutogradContext *ctx,
        torch::Tensor X,
        torch::Tensor W,
        torch::Tensor W_transposed,
        torch::Tensor one_hot_embedding);

    static variable_list backward(AutogradContext *ctx, variable_list grad_outputs);
};

class ElementalLinear : public torch::CustomClassHolder
{
public:
    torch::Tensor forward(
        torch::Tensor X,
        torch::Tensor W,
        torch::Tensor W_transposed,
        torch::Tensor one_hot_embedding);

    std::vector<torch::Tensor> __getstate__()
    {
        return {};
    }

    void __setstate__(const std::vector<torch::Tensor> &state)
    {
        return;
    }
};

#endif // LINEAR_WMMA_H