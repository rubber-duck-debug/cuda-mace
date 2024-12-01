#include "symmetric_contraction_wrapper.hpp"
#include "cuda_cache.hpp"
#include "cuda_utils.hpp"
#include <iostream>

using namespace c10;
using namespace std;
using namespace torch::autograd;
using namespace torch::indexing;

#define FULL_MASK 0xffffffff
#define WARP_SIZE 32
#define NWARPS_PER_BLOCK 4

std::vector<torch::Tensor> jit_symmetric_contraction_forward(
    torch::Tensor X, 
    torch::Tensor atom_types,
    const int u3_n_nonsparse,
    torch::Tensor U3_num_nonzero,
    torch::Tensor U3_indices, 
    torch::Tensor U3_values,
    torch::Tensor U2_num_nonzero, 
    torch::Tensor U2_indices,
    torch::Tensor U2_values, 
    torch::Tensor U1_num_nonzero,
    torch::Tensor U1_indices, 
    torch::Tensor W3, 
    torch::Tensor W2,
    torch::Tensor W1,
    const int W3_size,
    const int W2_size,
    const int W1_size
    ) {
  
  static const char* CUDA_CODE =
#include "generated/wrapped_symmetric_contraction_impl.cu"
        ;
  
  torch::Tensor output =
      torch::empty({X.size(0), 1, X.size(2)},
                   torch::TensorOptions().dtype(X.dtype()).device(X.device()));
  torch::Tensor grad;

  if (X.requires_grad()) {
    grad = torch::empty(
        {X.size(0), 1, X.size(1), X.size(2)},
        torch::TensorOptions().dtype(X.dtype()).device(X.device()));
  } else {
    grad = torch::empty(
        {1, 1, 1, 1},
        torch::TensorOptions().dtype(X.dtype()).device(X.device()));
  }

  const int nnodes = X.size(0);
  const int nl = 16;
  const int nchannels = X.size(2);

  int _w3_size = W3_size;
  int _w2_size = W2_size;
  int _w1_size = W1_size;

  auto find_num_blocks = [](int x, int bdim) { return (x + bdim - 1) / bdim; };

  dim3 gdim(nnodes, nchannels / WARP_SIZE);

  dim3 bdim(WARP_SIZE, NWARPS_PER_BLOCK, 1);

  AT_DISPATCH_FLOATING_TYPES(
      X.type(), "symmetric_contraction_forwards", ([&] {
        unsigned int shared_size = 0;

        void *sptr = nullptr;

        shared_array<scalar_t>(WARP_SIZE * nl, sptr, &shared_size);
        shared_array<scalar_t>(NWARPS_PER_BLOCK * WARP_SIZE, sptr, &shared_size);
        shared_array<float>(u3_n_nonsparse * nl * nl, sptr,
                            &shared_size);
        shared_array<float>(nl * nl, sptr, &shared_size);
        shared_array<scalar_t>(_w3_size * WARP_SIZE, sptr, &shared_size);
        shared_array<scalar_t>(_w2_size * WARP_SIZE, sptr, &shared_size);
        shared_array<scalar_t>(_w1_size * WARP_SIZE, sptr, &shared_size);

        shared_array<int>(u3_n_nonsparse * nl * nl, sptr,
                          &shared_size);
        shared_array<short>(nl * nl, sptr, &shared_size);
        shared_array<short>(nl * nl, sptr, &shared_size);
        shared_array<short>(nl * nl, sptr, &shared_size);

        scalar_t * _X  = X.data_ptr<scalar_t> ();
        int * _atom_types  = atom_types.data_ptr<int> ();
        short * _U3_num_nonzero  = U3_num_nonzero.data_ptr<short> ();
        int * _U3_indices  = U3_indices.data_ptr<int> ();
        float * _U3_values  = U3_values.data_ptr<float> ();

        short * _U2_num_nonzero  = U2_num_nonzero.data_ptr<short> ();
        short * _U2_indices  = U2_indices.data_ptr<short> ();
        float * _U2_values = U2_values.data_ptr<float>();

        short * _U1_num_nonzero  = U1_num_nonzero.data_ptr<short> ();
        short * _U1_indices = U1_indices.data_ptr<short>();

        scalar_t * _W3 = W3.data_ptr<scalar_t>();
        scalar_t * _W2 = W2.data_ptr<scalar_t>();
        scalar_t * _W1 = W1.data_ptr<scalar_t>();

        int _nelements = W3.size(1);
        int _u3_maxn_nonsparse = u3_n_nonsparse;
        int _nnodes = nnodes;
        int _nchannels = nchannels;
        
        scalar_t * _output = output.data_ptr<scalar_t>();
        scalar_t * _grad = grad.data_ptr<scalar_t>();

        std::vector<void*> args = {
        &_X,
        &_atom_types,
        &_U3_num_nonzero,
        &_U3_indices,
        &_U3_values,
        &_U2_num_nonzero,
        &_U2_indices,
        &_U2_values,
        &_U1_num_nonzero,
        &_U1_indices,
        &_W3,
        &_W2,
        &_W1,
        &_nelements,
        &_w3_size,
        &_w2_size,
        &_w1_size,
        &_u3_maxn_nonsparse,
        &_nnodes,
        &_nchannels,
        &_output,
        &_grad
        };

        auto kernel_name = [&]() -> std::string {
            if (X.requires_grad()) return getKernelName<scalar_t, std::integral_constant<bool, true>>("symmetric_contraction_kernel");
            if (!X.requires_grad()) return getKernelName<scalar_t, std::integral_constant<bool, false>>("symmetric_contraction_kernel");
            return ""; // Default case
        }();

        auto& kernel_factory = KernelFactory::instance();

        CachedKernel* kernel = kernel_factory.create(
            kernel_name, std::string(CUDA_CODE), "wrapped_symmetric_contraction_impl.cu", {"--std=c++17", "-lineinfo"}
        );

        kernel->launch(gdim, bdim, shared_size, 0, args);

      }));

  return {output, grad};
}

torch::Tensor jit_symmetric_contraction_backward(torch::Tensor gradX,
                                        torch::Tensor grad_input) {

   static const char* CUDA_CODE =
#include "generated/wrapped_symmetric_contraction_impl.cu"
        ;

  int nnodes = gradX.size(0);
  int nchannels = gradX.size(3);

  torch::Tensor output = torch::empty(
      {nnodes, 16, nchannels},
      torch::TensorOptions().dtype(gradX.dtype()).device(gradX.device()));

  dim3 gdim(nnodes, nchannels / WARP_SIZE);
  dim3 bdim(WARP_SIZE, 4, 1);

  AT_DISPATCH_FLOATING_TYPES(
      gradX.type(), "symm_contraction_backward", ([&] {
        unsigned int space =
            WARP_SIZE * 16 * sizeof(scalar_t); // buffer_grad storage

        scalar_t * _gradX  = gradX.data_ptr<scalar_t> ();
        scalar_t * _grad_input  = grad_input.data_ptr<scalar_t> ();
        scalar_t * _output  = output.data_ptr<scalar_t> ();
        
        std::vector<void*> args = {
        &_gradX,
        &_grad_input,
        &nnodes,
        &nchannels,
        &_output
        };

        auto kernel_name = getKernelName<scalar_t>("symm_contraction_backward_kernel");

        auto& kernel_factory = KernelFactory::instance();

        CachedKernel* kernel = kernel_factory.create(
            kernel_name, std::string(CUDA_CODE), "wrapped_symmetric_contraction_impl.cu", {"--std=c++17", "-lineinfo"}
        );

        kernel->launch(gdim, bdim, space, 0, args);
  
      }));

  return output;
}