#include "linear.h"
#include "linear_wrapper.hpp"
#include "utils.h"

using namespace std;
using namespace torch::autograd;
using namespace torch::indexing;

torch::Tensor LinearAutograd::forward(
    AutogradContext *ctx,
    torch::Tensor X,
    torch::Tensor W,
    torch::Tensor W_transposed)
{
    PUSH_RANGE("cuda_mace", 2)

    if (X.requires_grad())
    {
        ctx->save_for_backward({W_transposed});
    }

    torch::Tensor result = jit_linear(X, W);

    POP_RANGE

    return result;
}

variable_list LinearAutograd::backward(AutogradContext *ctx, variable_list grad_outputs)
{
    PUSH_RANGE("cuda_mace", 2)

    auto saved_variables = ctx->get_saved_variables();

    auto W_T = saved_variables[0];

    torch::Tensor dX = jit_linear(grad_outputs[0].contiguous(), W_T);

    torch::Tensor undef;

    POP_RANGE

    return {dX, undef, undef};
}

// wrapper class which we expose to the API.
torch::Tensor Linear::forward(
    torch::Tensor X,
    torch::Tensor W,
    torch::Tensor W_transposed)
{
    return LinearAutograd::apply(X, W, W_transposed);
}

torch::Tensor ElementalLinearAutograd::forward(
    AutogradContext *ctx,
    torch::Tensor X,
    torch::Tensor W,
    torch::Tensor W_transposed,
    torch::Tensor one_hot_embedding)
{

    PUSH_RANGE("cuda_mace", 3)

    if (X.requires_grad())
    {
        ctx->save_for_backward({one_hot_embedding, W_transposed});
    }

    torch::Tensor result = jit_elemental_linear(X, W, one_hot_embedding);

    POP_RANGE

    return result;
}

variable_list ElementalLinearAutograd::backward(AutogradContext *ctx, variable_list grad_outputs)
{
    PUSH_RANGE("cuda_mace", 3)

    auto saved_variables = ctx->get_saved_variables();

    auto one_hot_embedding = saved_variables[0];
    auto W_T = saved_variables[1];

    torch::Tensor dX = jit_elemental_linear(grad_outputs[0].contiguous(), W_T, one_hot_embedding);

    torch::Tensor undef;

    POP_RANGE

    return {dX, undef, undef, undef};
}

// wrapper class which we expose to the API.
torch::Tensor ElementalLinear::forward(
    torch::Tensor X,
    torch::Tensor W,
    torch::Tensor W_transposed,
    torch::Tensor one_hot_embedding)
{
    return ElementalLinearAutograd::apply(X, W, W_transposed, one_hot_embedding);
}

TORCH_LIBRARY(linear_wmma, m)
{
    m.class_<Linear>("Linear")
        .def(torch::init<>(), "", {})

        .def("forward", &Linear::forward, "", {torch::arg("X"), torch::arg("W"), torch::arg("W_T")})
        .def_pickle(
            [](const c10::intrusive_ptr<Linear> &self) -> std::vector<torch::Tensor>
            {
                return self->__getstate__();
            },
            [](const std::vector<torch::Tensor> &state) -> c10::intrusive_ptr<Linear>
            {
                auto obj = c10::make_intrusive<Linear>();
                obj->__setstate__(state);
                return obj;
            });

    m.class_<ElementalLinear>("ElementalLinear")
        .def(torch::init<>(), "", {})

        .def("forward", &ElementalLinear::forward, "", {torch::arg("X"), torch::arg("W"), torch::arg("W_T"), torch::arg("one_hot_encoding")})
        .def_pickle(
            [](const c10::intrusive_ptr<ElementalLinear> &self) -> std::vector<torch::Tensor>
            {
                return self->__getstate__();
            },
            [](const std::vector<torch::Tensor> &state) -> c10::intrusive_ptr<ElementalLinear>
            {
                auto obj = c10::make_intrusive<ElementalLinear>();
                obj->__setstate__(state);
                return obj;
            });
}