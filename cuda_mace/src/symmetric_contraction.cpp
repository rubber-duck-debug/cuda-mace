#include "symmetric_contraction_impl.cuh"
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
    torch::Tensor U3_num_nonzero_1,
    torch::Tensor U3_num_nonzero_2,
    torch::Tensor U3_num_nonzero_3,
    torch::Tensor U3_indices_0,
    torch::Tensor U3_indices_1,
    torch::Tensor U3_indices_2,
    torch::Tensor U3_indices_3,
    torch::Tensor U3_values_0,
    torch::Tensor U3_values_1,
    torch::Tensor U3_values_2,
    torch::Tensor U3_values_3,
    torch::Tensor U2_num_nonzero_1,
    torch::Tensor U2_num_nonzero_2,
    torch::Tensor U2_indices_1,
    torch::Tensor U2_indices_2,
    torch::Tensor U2_values_1,
    torch::Tensor U2_values_2,
    torch::Tensor U1_num_nonzero,
    torch::Tensor U1_index,
    torch::Tensor W3,
    torch::Tensor W2,
    torch::Tensor W1,
    const int64_t W3_L0_size,
    const int64_t W2_L0_size,
    const int64_t W1_L0_size,
    torch::Tensor W3_size,          // nlout
    torch::Tensor W2_size,          // nlout
    torch::Tensor W1_size,          // nlout
    torch::Tensor U3_max_nonsparse, // nlout
    const int64_t nthreadx,
    const int64_t nthready,
    const int64_t nthreadz)
{
    const int nlout = U3_values_1.size(0);

    std::vector<torch::Tensor> result;

    if (nlout == 1)
    {
        // use special l=0 kernel
        result = symmetric_contraction_L0_forwards_gpu(
            X,
            atom_types,
            U3_num_nonzero_1,
            U3_indices_0,
            U3_values_0,
            U2_num_nonzero_1,
            U2_indices_1,
            U2_values_1,
            U1_num_nonzero,
            U1_index,
            W3,
            W2,
            W1,
            W3_L0_size,
            W2_L0_size,
            W1_L0_size,
            U3_max_nonsparse,
            nthreadx,
            nthready,
            nthreadz);

        if (X.requires_grad())
        {
            ctx->save_for_backward({result[1]});
        }
    }
    else
    {
        // use generic kernel
        result = symmetric_contraction_LGT0_forwards_gpu(
            X,
            atom_types,
            U3_num_nonzero_1,
            U3_num_nonzero_2,
            U3_num_nonzero_3,
            U3_indices_0,
            U3_indices_1,
            U3_indices_2,
            U3_indices_3,
            U3_values_0,
            U3_values_1,
            U3_values_2,
            U3_values_3,
            U2_num_nonzero_1,
            U2_num_nonzero_2,
            U2_indices_1,
            U2_indices_2,
            U2_values_1,
            U2_values_2,
            U1_num_nonzero,
            U1_index,
            W3,
            W2,
            W1,
            W3_L0_size,
            W2_L0_size,
            W1_L0_size,
            W3_size,
            W2_size,
            W1_size,
            U3_max_nonsparse,
            nthreadx,
            nthready,
            nthreadz);

        if (X.requires_grad())
        {
            ctx->save_for_backward({result[1]});
        }
    }

    if (X.requires_grad())
    {
        ctx->saved_data["nthreadx"] = nthreadx;
        ctx->saved_data["nthready"] = nthready;
        ctx->saved_data["nthreadz"] = nthreadz;
    }

    return result[0];
}

variable_list SymmetricContractionAutograd::backward(AutogradContext *ctx, variable_list grad_outputs)
{
    auto saved_variables = ctx->get_saved_variables();

    int nthreadx = ctx->saved_data["nthreadx"].toInt();
    int nthready = ctx->saved_data["nthready"].toInt();
    int nthreadz = ctx->saved_data["nthreadz"].toInt();

    auto gradX = saved_variables[0];

    torch::Tensor result = symm_contraction_backward(gradX, grad_outputs[0], nthreadx, nthready, nthreadz);

    torch::Tensor undef;

    return {result,
            undef, undef, undef, undef, undef, undef, undef, undef, undef, undef, undef, undef, undef, undef, undef, undef, undef, undef, undef, undef,
            undef, undef, undef, undef, undef, undef, undef, undef, undef, undef, undef, undef, undef,
            undef};
}

torch::Tensor SymmetricContraction::forward(
    torch::Tensor X,
    torch::Tensor atom_types,
    torch::Tensor U3_num_nonzero_1,
    torch::Tensor U3_num_nonzero_2,
    torch::Tensor U3_num_nonzero_3,
    torch::Tensor U3_indices_0, // L=0 specific
    torch::Tensor U3_indices_1,
    torch::Tensor U3_indices_2,
    torch::Tensor U3_indices_3,
    torch::Tensor U3_values_0, // L=0 specific
    torch::Tensor U3_values_1,
    torch::Tensor U3_values_2,
    torch::Tensor U3_values_3,
    torch::Tensor U2_num_nonzero_1,
    torch::Tensor U2_num_nonzero_2,
    torch::Tensor U2_indices_1,
    torch::Tensor U2_indices_2,
    torch::Tensor U2_values_1,
    torch::Tensor U2_values_2,
    torch::Tensor U1_num_nonzero,
    torch::Tensor U1_index,
    torch::Tensor W3,
    torch::Tensor W2,
    torch::Tensor W1,
    const int64_t W3_L0_size,
    const int64_t W2_L0_size,
    const int64_t W1_L0_size,
    torch::Tensor W3_size,
    torch::Tensor W2_size,
    torch::Tensor W1_size,
    torch::Tensor U3_max_nonsparse,
    const int64_t nthreadx,
    const int64_t nthready,
    const int64_t nthreadz)
{

    return SymmetricContractionAutograd::apply(
        X,
        atom_types,
        U3_num_nonzero_1,
        U3_num_nonzero_2,
        U3_num_nonzero_3,
        U3_indices_0,
        U3_indices_1,
        U3_indices_2,
        U3_indices_3,
        U3_values_0,
        U3_values_1,
        U3_values_2,
        U3_values_3,
        U2_num_nonzero_1,
        U2_num_nonzero_2,
        U2_indices_1,
        U2_indices_2,
        U2_values_1,
        U2_values_2,
        U1_num_nonzero,
        U1_index,
        W3,
        W2,
        W1,
        W3_L0_size,
        W2_L0_size,
        W1_L0_size,
        W3_size,
        W2_size,
        W1_size,
        U3_max_nonsparse,
        nthreadx,
        nthready,
        nthreadz);
}

TORCH_LIBRARY(symm_contract, m)
{
    m.class_<SymmetricContraction>("SymmetricContraction")
        .def(torch::init<>(), "", {})

        .def("forward", &SymmetricContraction::forward, "", {torch::arg("X"), torch::arg("atom_types"), torch::arg("U3_num_nonzero_1"), torch::arg("U3_num_nonzero_2"), torch::arg("U3_num_nonzero_3"), torch::arg("U3_indices_0"), torch::arg("U3_indices_1"), torch::arg("U3_indices_2"), torch::arg("U3_indices_3"), torch::arg("U3_values_0"), torch::arg("U3_values_1"), torch::arg("U3_values_2"), torch::arg("U3_values_3"), torch::arg("U2_num_nonzero_1"), torch::arg("U2_num_nonzero_2"), torch::arg("U2_indices_1"), torch::arg("U2_indices_2"), torch::arg("U2_values_1"), torch::arg("U2_values_2"), torch::arg("U1_num_nonzero"), torch::arg("U1_index"), torch::arg("W3"), torch::arg("W2"), torch::arg("W1"), torch::arg("W3_L0_size"), torch::arg("W2_L0_size"), torch::arg("W1_L0_size"), torch::arg("W3_size"), torch::arg("W2_size"), torch::arg("W1_size"), torch::arg("U3_max_nonsparse"), torch::arg("nthreadx"), torch::arg("nthready"), torch::arg("nthreadz")})
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

    m.def("set_shared_mem_size", &set_shared_mem_size);
	m.def("curr_shared_mem", &curr_shared_mem);
	m.def("LGT0_shared_memory_required", &LGT0_shared_memory_required);
}
