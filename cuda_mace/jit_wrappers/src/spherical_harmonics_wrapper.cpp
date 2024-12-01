#include "spherical_harmonics_wrapper.hpp"
#include "cuda_cache.hpp"
#include "cuda_utils.hpp"

using namespace c10;
using namespace std;
using namespace torch::autograd;
using namespace torch::indexing;

#define FULL_MASK 0xffffffff
#define WARP_SIZE 32
#define NWARPS_PER_BLOCK 4


std::vector<torch::Tensor> jit_spherical_harmonics(torch::Tensor xyz) {

  static const char* CUDA_CODE =
#include "generated/wrapped_spherical_harmonics_impl.cu"
        ;

 int _nsamples = xyz.size(0);

  torch::Tensor sph_harmonics = torch::empty(
      {16, _nsamples},
      torch::TensorOptions().dtype(xyz.dtype()).device(xyz.device()));


  torch::Tensor sph_harmonics_deriv = torch::empty(
      {16, 3, _nsamples},
      torch::TensorOptions().dtype(xyz.dtype()).device(xyz.device()));

  dim3 gdim(find_integer_divisor(_nsamples, WARP_SIZE * NWARPS_PER_BLOCK));

  dim3 bdim(WARP_SIZE * NWARPS_PER_BLOCK, 1, 1);

  AT_DISPATCH_FLOATING_TYPES(
      xyz.scalar_type(), "spherical_harmonics", ([&] {
        
        scalar_t * _xyz  = xyz.data_ptr<scalar_t>();
        scalar_t * _sph_harmonics  = sph_harmonics.data_ptr<scalar_t>();
        scalar_t * _sph_harmonics_deriv  = sph_harmonics_deriv.data_ptr<scalar_t>();
        bool _norm = true;
        bool _grad = false;

        unsigned int space = 0;
        void *sptr;

        shared_array<scalar_t>(WARP_SIZE * NWARPS_PER_BLOCK * 3, sptr, &space);
        shared_array<scalar_t>(WARP_SIZE * NWARPS_PER_BLOCK * 16, sptr, &space);

        if (xyz.requires_grad()) {
          shared_array<scalar_t>(WARP_SIZE * NWARPS_PER_BLOCK * 16 * 3, sptr,
                                 &space);
          _grad=true;
        }

        std::vector<void*> args = {
        &_xyz,
        &_sph_harmonics,
        &_sph_harmonics_deriv,
        &_nsamples,
        &_norm,
        &_grad
        };

        std::string kernel_name = getKernelName<scalar_t>("spherical_harmonics_kernel_ptr");

        auto& kernel_factory = KernelFactory::instance();

        CachedKernel* kernel = kernel_factory.create(
            kernel_name, std::string(CUDA_CODE), "wrapped_spherical_harmonics_impl.cu", {"--std=c++17", "-lineinfo"}
        );

        kernel->launch(gdim, bdim, space, 0, args);

      }));

   if (xyz.requires_grad()) {
    return {sph_harmonics, sph_harmonics_deriv};
  } else {
    return {sph_harmonics};
  }
}

torch::Tensor jit_spherical_harmonics_backward(torch::Tensor sph_deriv,
                                           torch::Tensor grad_output) {

  static const char* CUDA_CODE =
#include "generated/wrapped_spherical_harmonics_impl.cu"
        ;

    int _nsamples = sph_deriv.size(2);

    torch::Tensor xyz_grad =
      torch::empty({_nsamples, 3}, torch::TensorOptions()
                                      .dtype(sph_deriv.dtype())
                                      .device(sph_deriv.device()));

    

  dim3 gdim(find_integer_divisor(_nsamples, WARP_SIZE * NWARPS_PER_BLOCK));

  dim3 bdim(WARP_SIZE * NWARPS_PER_BLOCK, 1, 1);

  AT_DISPATCH_FLOATING_TYPES(
      sph_deriv.scalar_type(), "jit_spherical_harmonics_backward", ([&] {

        unsigned int space = 0;
        void *sptr;

        shared_array<scalar_t>(WARP_SIZE * NWARPS_PER_BLOCK * 3, sptr, &space);

         scalar_t * _sph_deriv  = sph_deriv.data_ptr<scalar_t> ();
        scalar_t * _grad_output  = grad_output.data_ptr<scalar_t> ();
        scalar_t * _xyz_grad  = xyz_grad.data_ptr<scalar_t> ();

        std::vector<void*> args = {
          &_sph_deriv,
          &_grad_output,
          &_nsamples,
          &_xyz_grad
        };

        std::string kernel_name = getKernelName<scalar_t>("spherical_harmonics_backward_kernel_ptr");

        auto& kernel_factory = KernelFactory::instance();

        CachedKernel* kernel = kernel_factory.create(
          kernel_name, std::string(CUDA_CODE), "wrapped_spherical_harmonics_impl.cu", {"--std=c++17", "-lineinfo"}
        );

        kernel->launch(gdim, bdim, space, 0, args);

      }));

  return xyz_grad;
}