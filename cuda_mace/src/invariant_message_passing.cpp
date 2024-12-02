#include "invariant_message_passing_wrapper.hpp"
#include "invariant_message_passing.h"

#include <torch/script.h>
#include <iostream>

using namespace std;
using namespace torch::indexing;
using namespace torch::autograd;

torch::Tensor InvariantMessagePassingTPAutograd::forward(
    AutogradContext *ctx,
    torch::Tensor X,
    torch::Tensor Y,
    torch::Tensor radial,
    torch::Tensor sender_list,
    torch::Tensor receiver_list,
    const int64_t nnodes)
{
    torch::Tensor first_occurences = jit_calculate_first_occurences(receiver_list, nnodes);

    //auto result  = forward_gpu(X, Y, radial, sender_list, receiver_list, first_occurences, nnodes);
    std::vector<torch::Tensor> result = jit_forward_message_passing(X, Y, radial, sender_list, receiver_list, first_occurences, nnodes);
    
    if (X.requires_grad() || Y.requires_grad() || radial.requires_grad())
    {
        ctx->saved_data["nnodes"] = nnodes;
        ctx->save_for_backward({X, Y, radial, sender_list, receiver_list, first_occurences, result[1]});
    }
    
    return result[0];
}

variable_list InvariantMessagePassingTPAutograd::backward(AutogradContext *ctx, variable_list grad_outputs)
{

    auto saved_variables = ctx->get_saved_variables();

    auto X = saved_variables[0];
    auto Y = saved_variables[1];
    auto radial = saved_variables[2];
    auto sender_list = saved_variables[3];
    auto receiver_list = saved_variables[4];
    auto first_occurences = saved_variables[5];
    auto node_edge_index = saved_variables[6];

    int64_t nnodes = ctx->saved_data["nnodes"].toInt();

    //auto result = backward_gpu(X, Y, radial, grad_outputs[0], sender_list, receiver_list, first_occurences, node_edge_index, nnodes);
    std::vector<torch::Tensor> result = jit_backward_message_passing(X, Y, radial, grad_outputs[0], sender_list, receiver_list, first_occurences, node_edge_index, nnodes);

    torch::Tensor undef;

    return {result[0], result[1], result[2], undef, undef, undef};
}

// wrapper class which we expose to the API.
torch::Tensor InvariantMessagePassingTP::forward(
    torch::Tensor X,
    torch::Tensor Y,
    torch::Tensor radial,
    torch::Tensor sender_list,
    torch::Tensor receiver_list,
    const int64_t nnodes)
{
    return InvariantMessagePassingTPAutograd::apply(X, Y, radial, sender_list, receiver_list, nnodes);
}

TORCH_LIBRARY(inv_message_passing, m)
{
    m.class_<InvariantMessagePassingTP>("InvariantMessagePassingTP")
        .def(torch::init<>(), "", {})

        .def("forward", &InvariantMessagePassingTP::forward, "", {torch::arg("X"), torch::arg("Y"), torch::arg("raidal"), torch::arg("sender_list"), torch::arg("receiver_list"), torch::arg("nnodes")})
        .def_pickle(
            [](const c10::intrusive_ptr<InvariantMessagePassingTP> &self) -> std::vector<torch::Tensor>
            {
                return self->__getstate__();
            },
            [](const std::vector<torch::Tensor> &state) -> c10::intrusive_ptr<InvariantMessagePassingTP>
            {
                auto obj = c10::make_intrusive<InvariantMessagePassingTP>();
                obj->__setstate__(state);
                return obj;
            });
}
