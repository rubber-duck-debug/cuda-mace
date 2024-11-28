#include "cubic_spline_wrapper.hpp"
#include "cuda_cache.hpp"

using namespace c10;
using namespace std;
using namespace torch::autograd;
using namespace torch::indexing;

#define FULL_MASK 0xffffffff
#define WARP_SIZE 32
#define NWARPS_PER_BLOCK 4


std::vector<torch::Tensor> jit_evaluate_spline(torch::Tensor r,
                                           torch::Tensor r_knots,
                                           torch::Tensor coeffs, double r_width,
                                           double r_max) {
  static const char* CUDA_CODE =
#include "generated/wrapped_cubic_spline_impl2.cu"
        ;

  int nsamples = r.size(0);
  int nknots = r_knots.size(0);
  int noutputs = coeffs.size(2);
  torch::Tensor R_out =
      torch::empty({nsamples, noutputs},
                   torch::TensorOptions().dtype(r.dtype()).device(r.device()));

  torch::Tensor R_deriv = torch::empty(
      {1, 1}, torch::TensorOptions().dtype(r.dtype()).device(r.device()));

  if (r.requires_grad()) {
    R_deriv = torch::empty(
        {nsamples, noutputs},
        torch::TensorOptions().dtype(r.dtype()).device(r.device()));
  }
  

auto find_integer_divisor = [](int x, int y) -> int {
        if (y == 0) {
            throw std::invalid_argument("Divisor cannot be zero.");
        }
        return (x + y - 1) / y;
    };

  dim3 gdim(find_integer_divisor(nsamples, NWARPS_PER_BLOCK));

  dim3 bdim(32, NWARPS_PER_BLOCK, 1);

  AT_DISPATCH_FLOATING_TYPES(
      r.scalar_type(), "evaluate_spline", ([&] {
        size_t space = 0;
        void *sptr;

        scalar_t * _r  = r.data_ptr<scalar_t> ();
        scalar_t * _r_knots  = r_knots.data_ptr<scalar_t> ();
        scalar_t * _coeff  = coeffs.data_ptr<scalar_t> ();
        scalar_t * _R_out  = R_out.data_ptr<scalar_t> ();
        scalar_t * _R_out_deriv  = R_deriv.data_ptr<scalar_t>();
        float _r_width = r_width;
        float _r_max = r_max;

        std::vector<void*> args = {
        &_r,
        &_r_knots,
        &_coeff,
        &nsamples,
        &nknots,
        &noutputs,
        &_r_width,
        &_r_max,
        &_R_out,
        &_R_out_deriv
        };

        std::string kernel_name;
          if (r.requires_grad()) {
         kernel_name= getKernelName<scalar_t, std::integral_constant<bool, true>>("evaluate_spline_kernel_ptr");
          } else {
          kernel_name =  getKernelName<scalar_t, std::integral_constant<bool, false>>("evaluate_spline_kernel_ptr");
    }

    auto& kernel_factory = KernelFactory::instance();

    CachedKernel* kernel = kernel_factory.create(
        kernel_name, std::string(CUDA_CODE), "wrapped_cubic_spline_impl2.cu", {"--std=c++17"}
    );

        kernel->launch(gdim, bdim, 0, 0, args);
      }));

  if (r.requires_grad()) {
    return {R_out, R_deriv};
  } else {
    return {R_out};
  }
}

torch::Tensor jit_backward_spline(torch::Tensor grad_output,
                              torch::Tensor R_deriv) {

  static const char* CUDA_CODE =
#include "generated/wrapped_cubic_spline_impl2.cu"
        ;

  auto find_integer_divisor = [](int x, int y) -> int {
        if (y == 0) {
            throw std::invalid_argument("Divisor cannot be zero.");
        }
        return (x + y - 1) / y;
    };

  int nsamples = R_deriv.size(0);
  int noutputs = R_deriv.size(1);
  torch::Tensor r_grad = torch::empty(
      {nsamples},
      torch::TensorOptions().dtype(R_deriv.dtype()).device(R_deriv.device()));

  dim3 gdim(find_integer_divisor(nsamples, NWARPS_PER_BLOCK));

  dim3 bdim(WARP_SIZE * NWARPS_PER_BLOCK, 1, 1);

  AT_DISPATCH_FLOATING_TYPES(
      R_deriv.scalar_type(), "backward_spline", ([&] {
        size_t space = 0;
        void *sptr;

         scalar_t * _grad_output  = grad_output.data_ptr<scalar_t> ();
        scalar_t * _R_deriv  = R_deriv.data_ptr<scalar_t> ();
        scalar_t * _r_grad  = r_grad.data_ptr<scalar_t> ();

        std::vector<void*> args = {
          &_grad_output,
          &_R_deriv,
          &_r_grad,
          &nsamples,
          &noutputs,
        };
        
        std::string kernel_name = getKernelName<scalar_t>("backward_spline_kernel_ptr");

    auto& kernel_factory = KernelFactory::instance();

    CachedKernel* kernel = kernel_factory.create(
        kernel_name, std::string(CUDA_CODE), "wrapped_cubic_spline_impl2.cu", {"--std=c++17"}
    );

        kernel->launch(gdim, bdim, 0, 0, args);

      }));

  return r_grad;
}