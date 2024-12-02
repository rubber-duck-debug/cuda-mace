#include "symmetric_contraction_wrapper.hpp"
#include "symmetric_contraction.h"
#include <torch/script.h>
#include <iostream>

using namespace std;
using namespace torch::indexing;
using namespace torch::autograd;

torch::Tensor SymmetricContractionAutograd::forward(
    AutogradContext *ctx,
    torch::Tensor X,
    torch::Tensor atom_types,
    const int64_t U3_max_nonsparse,
    torch::Tensor U3_num_nonzero,
    torch::Tensor U3_indices,
    torch::Tensor U3_values,
    torch::Tensor U2_num_nonzero,
    torch::Tensor U2_indices,
    torch::Tensor U2_values,
    torch::Tensor U1_num_nonzero,
    torch::Tensor U1_index,
    torch::Tensor W3,
    torch::Tensor W2,
    torch::Tensor W1,
    const int64_t W3_size,
    const int64_t W2_size,
    const int64_t W1_size)
{

    std::vector<torch::Tensor> result;

    result = jit_symmetric_contraction_forward(
        X,
        atom_types,
        U3_max_nonsparse,
        U3_num_nonzero,
        U3_indices,
        U3_values,
        U2_num_nonzero,
        U2_indices,
        U2_values,
        U1_num_nonzero,
        U1_index,
        W3,
        W2,
        W1,
        W3_size,
        W2_size,
        W1_size);

    if (X.requires_grad())
    {
        ctx->save_for_backward({result[1]});
    }

    return result[0];
}

variable_list SymmetricContractionAutograd::backward(AutogradContext *ctx, variable_list grad_outputs)
{
    auto saved_variables = ctx->get_saved_variables();


    auto gradX = saved_variables[0];

    torch::Tensor result = jit_symmetric_contraction_backward(gradX, grad_outputs[0]);

    torch::Tensor undef;

    return {result,
            undef, undef, undef, undef, undef, undef, undef, undef, undef, undef, undef, undef, undef, undef, undef, undef};
}

torch::Tensor SymmetricContraction::forward(
    torch::Tensor X,
    torch::Tensor atom_types,
    const int64_t U3_max_nonsparse,
    torch::Tensor U3_num_nonzero,
    torch::Tensor U3_indices,
    torch::Tensor U3_values,
    torch::Tensor U2_num_nonzero,
    torch::Tensor U2_indices,
    torch::Tensor U2_values,
    torch::Tensor U1_num_nonzero,
    torch::Tensor U1_index,
    torch::Tensor W3,
    torch::Tensor W2,
    torch::Tensor W1,
    const int64_t W3_size,
    const int64_t W2_size,
    const int64_t W1_size)
{
    return SymmetricContractionAutograd::apply(
        X,
        atom_types,
        U3_max_nonsparse,
        U3_num_nonzero,
        U3_indices,
        U3_values,
        U2_num_nonzero,
        U2_indices,
        U2_values,
        U1_num_nonzero,
        U1_index,
        W3,
        W2,
        W1,
        W3_size,
        W2_size,
        W1_size);
}

TORCH_LIBRARY(symm_contract, m)
{
    m.class_<SymmetricContraction>("SymmetricContraction")
        .def(torch::init<>(), "", {})

        .def("forward", &SymmetricContraction::forward, "", {
            torch::arg("X"), 
            torch::arg("atom_types"),
            torch::arg("U3_max_nonsparse"),
            torch::arg("U3_num_nonzero"),
            torch::arg("U3_indices"),
            torch::arg("U3_values"),
            torch::arg("U2_num_nonzero"),
            torch::arg("U2_indices"),
            torch::arg("U2_values"),
            torch::arg("U1_num_nonzero"),
            torch::arg("U1_index"), 
            torch::arg("W3"),
            torch::arg("W2"),
            torch::arg("W1"), 
            torch::arg("W3_size"),
            torch::arg("W2_size"),
            torch::arg("W1_size")})
        .def_pickle(
            [](const c10::intrusive_ptr<SymmetricContraction> &self) -> std::vector<torch::Tensor>
            {
                return self->__getstate__();
            },
            [](const std::vector<torch::Tensor> &state) -> c10::intrusive_ptr<SymmetricContraction>
            {
                auto obj = c10::make_intrusive<SymmetricContraction>();
                obj->__setstate__(state);
                return obj;
            });
}
