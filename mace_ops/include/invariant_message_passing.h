#ifndef INVARIANT_MESSAGE_PASSING_H
#define INVARIANT_MESSAGE_PASSING_H
#include <torch/script.h>
#include <iostream>

using namespace std;
using namespace torch::indexing;
using namespace torch::autograd;

class InvariantMessagePassingTPAutograd : public Function<InvariantMessagePassingTPAutograd>
{
public:
    static torch::Tensor forward(
        AutogradContext *ctx,
        torch::Tensor X,
        torch::Tensor Y,
        torch::Tensor radial,
        torch::Tensor sender_list,
        torch::Tensor receiver_list,
        const int64_t nnodes);

    static variable_list backward(AutogradContext *ctx, variable_list grad_outputs);
};

class InvariantMessagePassingTP : public torch::CustomClassHolder
{
public:
    torch::Tensor forward(
        torch::Tensor X,
        torch::Tensor Y,
        torch::Tensor radial,
        torch::Tensor sender_list,
        torch::Tensor receiver_list,
        const int64_t nnodes);

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