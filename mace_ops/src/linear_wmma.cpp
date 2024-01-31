#include "linear_wmma_impl.cuh"
#include "linear_wmma.h"

using namespace std;
using namespace torch::autograd;
using namespace torch::indexing;

torch::Tensor LinearAutograd::forward(
    AutogradContext *ctx,
    torch::Tensor X,
    torch::Tensor W,
    torch::Tensor W_transposed)
{
    if (X.requires_grad())
    {
        ctx->save_for_backward({W_transposed});
    }

    torch::Tensor result = linear_wmma(X, W);

    return result;
}

variable_list LinearAutograd::backward(AutogradContext *ctx, variable_list grad_outputs)
{
    auto saved_variables = ctx->get_saved_variables();

    auto W_T = saved_variables[0];

    if (!grad_outputs[0].is_contiguous())
    {
        grad_outputs[0] = grad_outputs[0].contiguous();
    }

    torch::Tensor dX = linear_wmma(grad_outputs[0], W_T);

    torch::Tensor undef;

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

    if (X.requires_grad())
    {
        ctx->save_for_backward({one_hot_embedding, W_transposed});
    }

    torch::Tensor result = elemental_linear_wmma(X, W, one_hot_embedding);

    return result;
}

variable_list ElementalLinearAutograd::backward(AutogradContext *ctx, variable_list grad_outputs)
{
    auto saved_variables = ctx->get_saved_variables();

    auto one_hot_embedding = saved_variables[0];
    auto W_T = saved_variables[1];

    if (!grad_outputs[0].is_contiguous())
    {
        grad_outputs[0] = grad_outputs[0].contiguous();
    }

    torch::Tensor dX = elemental_linear_wmma(grad_outputs[0], W_T, one_hot_embedding);

    torch::Tensor undef;

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