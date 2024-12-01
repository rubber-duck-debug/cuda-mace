#include "invariant_message_passing_wrapper.hpp"
#include "cuda_utils.hpp"
#include "cuda_cache.hpp"

#include <iostream>
#include <torch/script.h>

using namespace std;
using namespace torch::indexing;
using namespace torch::autograd;

#define FULL_MASK 0xffffffff

#define WARP_SIZE 32
#define NWARPS_PER_BLOCK 4
#define NEIGHBOUR_NEDGES_PER_BLOCK 512
#define NELEMENTS_PER_BLOCK 512

torch::Tensor jit_calculate_first_occurences(torch::Tensor receiver_list,
                                             const int64_t nnodes) {

    static const char* CUDA_CODE =
#include "generated/wrapped_invariant_message_passing_impl.cu"
        ;

    torch::Tensor first_occurences =
        torch::empty(2 * nnodes, torch::TensorOptions()
                                .dtype(receiver_list.dtype())
                                .device(receiver_list.device()));

    int nbx =
        find_integer_divisor(receiver_list.size(0), NELEMENTS_PER_BLOCK);

    dim3 gdim(nbx);

    dim3 bdim(128, 1, 1);

    unsigned int space = 0;
    space += (NELEMENTS_PER_BLOCK + 1) * sizeof(int);

    int * _receiver_list  = receiver_list.data_ptr<int> ();
    int _nedges = receiver_list.size(0);
    int _nnodes = nnodes;
    int * _sort_idx  = nullptr;
    bool _use_sort = false;
    int * _first_occurences  = first_occurences.data_ptr<int>();
    int * _first_occurences_shift = _first_occurences + nnodes;
    
    std::vector<void*> args = {
    &_receiver_list,
    &_nedges,
    &_nnodes,
    &_sort_idx,
    &_use_sort,
    &_first_occurences,
    &_first_occurences_shift,
    };

    auto kernel_name =  getKernelName("calculate_first_occurences_kernel_ptr");

    auto& kernel_factory = KernelFactory::instance();

    CachedKernel* kernel = kernel_factory.create(
        kernel_name, std::string(CUDA_CODE), "wrapped_invariant_message_passing_impl.cu", {"--std=c++17", "-lineinfo"}
    );

    kernel->launch(gdim, bdim, space, 0, args);

  return first_occurences;
}

std::vector<torch::Tensor>
jit_forward_message_passing(torch::Tensor X, torch::Tensor Y, torch::Tensor radial,
            torch::Tensor sender_list, torch::Tensor receiver_list,
            torch::Tensor first_occurences, const int64_t nnodes) {

  static const char* CUDA_CODE =
#include "generated/wrapped_invariant_message_passing_impl.cu"
        ;

  const uint nedges = Y.size(1);
  const int nspherical_harm = Y.size(0);
  const int nfeatures = X.size(1);

  TORCH_CHECK(nfeatures % WARP_SIZE == 0,
              "feature dimension must be a multiple of 32");
  TORCH_CHECK(nspherical_harm == 16,
              "number of edge spherical harmonics must be 16");
  TORCH_CHECK(nfeatures <= 128, "feature dimension cannot be greater than 128");

  torch::Tensor node_edge_index = torch::empty(
      {nnodes, nnodes},
      torch::TensorOptions().dtype(torch::kInt32).device(X.device()));

  torch::Tensor output =
      torch::empty({nnodes, nspherical_harm, nfeatures},
                   torch::TensorOptions().dtype(X.dtype()).device(X.device()));

  dim3 gdim(nnodes);
  dim3 bdim(NWARPS_PER_BLOCK * WARP_SIZE, 1, 1);

  AT_DISPATCH_FLOATING_TYPES(
      X.scalar_type(), "forward_gpu",
      ([&] {
        unsigned int space = 0;
        void *sptr;
        shared_array<int>(512, sptr, &space);

        scalar_t * _X  = X.data_ptr<scalar_t> ();
        scalar_t * _Y  = Y.data_ptr<scalar_t> ();
        scalar_t * _radial  = radial.data_ptr<scalar_t> ();
        int * _sender_list  = sender_list.data_ptr<int> ();
        int * _receiver_list  = receiver_list.data_ptr<int>();
        int * _first_occurences  = first_occurences.data_ptr<int>();
        int * _node_edge_index  = node_edge_index.data_ptr<int>();
        scalar_t * _output  = output.data_ptr<scalar_t>();

        int _nedges = Y.size(1);
        int _nchannels = X.size(1);
        int _nnodes = nnodes;

        std::vector<void*> args = {
        &_X,
        &_Y,
        &_radial,
        &_sender_list,
        &_receiver_list,
        &_first_occurences,
        &_node_edge_index,
        &_nedges,
        &_nchannels,
        &_nnodes,
        &_output
        };

        auto kernel_name = [&]() -> std::string {
            if (nfeatures >= 128) return getKernelName<scalar_t, std::integral_constant<int, 4>, std::integral_constant<int, 4>>("inv_tp_kernel_ptr");
            if (nfeatures == 96)  return getKernelName<scalar_t, std::integral_constant<int, 4>, std::integral_constant<int, 3>>("inv_tp_kernel_ptr");
            if (nfeatures == 64)  return getKernelName<scalar_t, std::integral_constant<int, 4>, std::integral_constant<int, 2>>("inv_tp_kernel_ptr");
            if (nfeatures == 32)  return getKernelName<scalar_t, std::integral_constant<int, 4>, std::integral_constant<int, 1>>("inv_tp_kernel_ptr");
            return ""; // Default case
        }();

        auto& kernel_factory = KernelFactory::instance();

        CachedKernel* kernel = kernel_factory.create(
            kernel_name, std::string(CUDA_CODE), "wrapped_invariant_message_passing_impl.cu", {"--std=c++17", "-lineinfo"}
        );

        kernel->launch(gdim, bdim, space, 0, args);

        }));

  return {output, node_edge_index};
}

std::vector<torch::Tensor>
jit_backward_message_passing(torch::Tensor X, torch::Tensor Y, torch::Tensor radial,
             torch::Tensor grad_in, torch::Tensor sender_list,
             torch::Tensor receiver_list, torch::Tensor first_occurences,
             torch::Tensor node_edge_index, const int64_t nnodes) {

    static const char* CUDA_CODE =
        #include "generated/wrapped_invariant_message_passing_impl.cu"
    ;

    int _nedges = Y.size(1);
    int _nchannels = X.size(1);
    int _nnodes = nnodes;

  TORCH_CHECK(X.requires_grad(), "X must require grad for invariant message "
                                 "passing backwards_kernel to be called.");
  TORCH_CHECK(Y.requires_grad(), "Y must require grad for invariant message "
                                 "passing backwards_kernel to be called.");
  TORCH_CHECK(radial.requires_grad(),
              "radial must require grad for invariant message passing "
              "backwards_kernel to be called.");

  torch::Tensor gradRadial = torch::empty_like(
      radial,
      torch::TensorOptions().dtype(radial.dtype()).device(radial.device()));

  torch::Tensor gradX = torch::empty_like(
      X, torch::TensorOptions().dtype(X.dtype()).device(X.device()));

  torch::Tensor gradY = torch::empty_like(
      Y, torch::TensorOptions().dtype(Y.dtype()).device(Y.device()));

  AT_DISPATCH_FLOATING_TYPES(
      X.scalar_type(), "backward_gpu", ([&] {
        dim3 bdim(NWARPS_PER_BLOCK * WARP_SIZE, 1, 1);
        dim3 gdim(nnodes, 1);

        void *sptr = nullptr;
        unsigned int space = 0;

        shared_array<scalar_t>(16 * _nchannels, sptr, &space);
        shared_array<scalar_t>(2 * NWARPS_PER_BLOCK * 16, sptr,
                               &space); // buffer_Y, buffer_dY

        void *sptr_node = nullptr;
        unsigned int space_node = 0;

        shared_array<scalar_t>(NWARPS_PER_BLOCK * WARP_SIZE, sptr_node,
                               &space_node);
        shared_array<int>(512 * 2, sptr_node, &space_node);

        scalar_t * _X  = X.data_ptr<scalar_t> ();
        scalar_t * _Y  = Y.data_ptr<scalar_t> ();
        scalar_t * _radial  = radial.data_ptr<scalar_t> ();
        int * _sender_list  = sender_list.data_ptr<int> ();
        int * _receiver_list  = receiver_list.data_ptr<int>();
        int * _first_occurences  = first_occurences.data_ptr<int>();
        int * _node_edge_index  = node_edge_index.data_ptr<int>();
        scalar_t * _grad_in  = grad_in.data_ptr<scalar_t>();
        scalar_t * _gradY  = gradY.data_ptr<scalar_t>();
        scalar_t * _gradX  = gradX.data_ptr<scalar_t>();
        scalar_t * _gradRadial  = gradRadial.data_ptr<scalar_t>();



        std::vector<void*> args_edge = {
        &_X,
        &_Y,
        &_radial,
        &_grad_in,
        &_sender_list,
        &_receiver_list,
        &_first_occurences,
        &_nedges,
        &_nchannels,
        &_nnodes,
        &_gradY,
        &_gradRadial
        };

         std::vector<void*> args_node = {
        &_Y,
        &_radial,
        &_grad_in,
        &_sender_list,
        &_receiver_list,
        &_first_occurences,
        &_node_edge_index,
        &_nedges,
        &_nchannels,
        &_nnodes,
        &_gradX
        };

        auto kernel_name_edge = [&]() -> std::string {
            if (_nchannels >= 128) return getKernelName<scalar_t, std::integral_constant<int, 4>, std::integral_constant<int, 4>>("backward_edge_inv_tp_kernel_ptr");
            if (_nchannels == 96)  return getKernelName<scalar_t, std::integral_constant<int, 4>, std::integral_constant<int, 3>>("backward_edge_inv_tp_kernel_ptr");
            if (_nchannels == 64)  return getKernelName<scalar_t, std::integral_constant<int, 4>, std::integral_constant<int, 2>>("backward_edge_inv_tp_kernel_ptr");
            if (_nchannels == 32)  return getKernelName<scalar_t, std::integral_constant<int, 4>, std::integral_constant<int, 1>>("backward_edge_inv_tp_kernel_ptr");
            return ""; // Default case
        }();

        auto kernel_name_node = [&]() -> std::string {
            if (_nchannels >= 128) return getKernelName<scalar_t, std::integral_constant<int, 4>, std::integral_constant<int, 4>>("backward_node_inv_tp_kernel_ptr");
            if (_nchannels == 96)  return getKernelName<scalar_t, std::integral_constant<int, 4>, std::integral_constant<int, 3>>("backward_node_inv_tp_kernel_ptr");
            if (_nchannels == 64)  return getKernelName<scalar_t, std::integral_constant<int, 4>, std::integral_constant<int, 2>>("backward_node_inv_tp_kernel_ptr");
            if (_nchannels == 32)  return getKernelName<scalar_t, std::integral_constant<int, 4>, std::integral_constant<int, 1>>("backward_node_inv_tp_kernel_ptr");
            return ""; // Default case
        }();

        auto& kernel_factory = KernelFactory::instance();

        CachedKernel* kernel_node = kernel_factory.create(
            kernel_name_node, std::string(CUDA_CODE), "wrapped_invariant_message_passing_impl.cu", {"--std=c++17", "-lineinfo"}
        );

        CachedKernel* kernel_edge = kernel_factory.create(
            kernel_name_edge, std::string(CUDA_CODE), "wrapped_invariant_message_passing_impl.cu", {"--std=c++17", "-lineinfo"}
        );


        kernel_edge->launch(gdim, bdim, space, 0, args_edge);
        kernel_node->launch(gdim, bdim, space_node, 0, args_node);
      }));


  return {gradX, gradY, gradRadial};
}