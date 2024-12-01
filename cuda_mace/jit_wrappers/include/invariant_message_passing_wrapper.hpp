#ifndef INVARIANT_MESSAGE_PASSING_WRAPPER_HPP
#define INVARIANT_MESSAGE_PASSING_WRAPPER_HPP

#include <torch/script.h>
#include <vector>

using namespace std;
using namespace torch;

torch::Tensor jit_calculate_first_occurences(torch::Tensor receiver_list,
                                             const int64_t nnodes);
std::vector<torch::Tensor>
jit_forward_message_passing(torch::Tensor X, torch::Tensor Y, torch::Tensor radial,
            torch::Tensor sender_list, torch::Tensor receiver_list,
            torch::Tensor first_occurences, const int64_t nnodes);

std::vector<torch::Tensor>
jit_backward_message_passing(torch::Tensor X, torch::Tensor Y, torch::Tensor radial,
             torch::Tensor grad_in, torch::Tensor sender_list,
             torch::Tensor receiver_list, torch::Tensor first_occurences,
             torch::Tensor node_edge_index, const int64_t nnodes);

#endif // INVARIANT_MESSAGE_PASSING_WRAPPER_HPP