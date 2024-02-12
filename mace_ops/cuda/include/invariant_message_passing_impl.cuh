#ifndef INVARIANT_MESSAGE_PASSING_IMPL_CUH
#define INVARIANT_MESSAGE_PASSING_IMPL_CUH

#include <torch/script.h>
#include <vector>

using namespace std;
using namespace torch;

torch::Tensor forward_gpu(
    torch::Tensor X,
    torch::Tensor Y,
    torch::Tensor radial,
    torch::Tensor sender_list,
    torch::Tensor receiver_list,
    torch::Tensor first_occurences,
    const int64_t nnodes);

std::vector<torch::Tensor> backward_gpu(torch::Tensor X,
                                        torch::Tensor Y,
                                        torch::Tensor radial,
                                        torch::Tensor grad_in,
                                        torch::Tensor sender_list,
                                        torch::Tensor receiver_list,
                                        torch::Tensor first_occurences,
                                        const int64_t nnodes);

#endif // INVARIANT_MESSAGE_PASSING_IMPL_CUH