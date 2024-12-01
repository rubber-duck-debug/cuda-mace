#ifndef SYMMETRIC_CONTRACTION_WRAPPER_HPP
#define SYMMETRIC_CONTRACTION_WRAPPER_HPP

#include <torch/script.h>

using namespace torch;

std::vector<torch::Tensor> jit_symmetric_contraction_forward(
    torch::Tensor X, torch::Tensor atom_types, const int u3_n_nonsparse,
    torch::Tensor U3_num_nonzero,
    torch::Tensor U3_indices, torch::Tensor U3_values,
    torch::Tensor U2_num_nonzero, torch::Tensor U2_indices,
    torch::Tensor U2_values, torch::Tensor U1_num_nonzero,
    torch::Tensor U1_indices, torch::Tensor W3, torch::Tensor W2,
    torch::Tensor W1,const int W3_size, const int W2_size, const int W1_size);


torch::Tensor jit_symmetric_contraction_backward(torch::Tensor gradX,
                                        torch::Tensor grad_input);


#endif