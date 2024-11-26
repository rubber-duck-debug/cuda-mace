#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "cubic_spline_impl.cuh"
#include "cuda_utils.cuh"

using namespace c10;
using namespace std;
using namespace torch::autograd;
using namespace torch::indexing;

#define FULL_MASK 0xffffffff
#define WARP_SIZE 32
#define NWARPS_PER_BLOCK 4

template <typename scalar_t, const bool evaluate_deriv>
__global__ void evaluate_spline_kernel(
    const torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits>
        r,
    const torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits>
        r_knots,
    scalar_t *coeff, const float r_width, const float r_max,
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits>
        R_out, /* [(r.shape[0] -1) * 4, R.shape[-1]]  */
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits>
        R_deriv) {

  extern __shared__ char buffer[];

  const int i = blockIdx.x * NWARPS_PER_BLOCK + threadIdx.y;

  if (i >= r.size(0)) {
    return;
  }

  const scalar_t r_i = r[i];

  if (r_i > r_max) {
    for (int j = threadIdx.x; j < R_out.size(1); j += blockDim.x) {

      R_out[i][j] = 0.0;

      if (evaluate_deriv) {
        R_deriv[i][j] = 0.0;
      }
    }
    return;
  }

  int idx = (int)(r_i / r_width);

  if (idx < 0) {
    idx = 0;
  } else if (idx >= r_knots.size(0)) {
    idx = r_knots.size(0) - 1;
  }

  const scalar_t x = r_i - r_knots[idx];
  const scalar_t xx = x * x;
  const scalar_t xxx = xx * x;

  __syncthreads();

  for (int j = threadIdx.x; j < R_out.size(1); j += blockDim.x) {

    scalar_t coeffs[4] = {0.0};
    // float4 coeffs = reinterpret_cast<float4 *>(coeff)[idx * 4 *
    // R_out.size(1)];
    for (int l = 0; l < 4; l++) {
      coeffs[l] = coeff[idx * 4 * R_out.size(1) + l * R_out.size(1) + j];
    }

    R_out[i][j] = coeffs[0] + coeffs[1] * x + coeffs[2] * xx + coeffs[3] * xxx;

    if (evaluate_deriv) {
      R_deriv[i][j] = coeffs[1] + scalar_t(2.0) * coeffs[2] * x +
                      scalar_t(3.0) * coeffs[3] * xx;
    }
  }
}

std::vector<torch::Tensor> evaluate_spline(torch::Tensor r,
                                           torch::Tensor r_knots,
                                           torch::Tensor coeffs, double r_width,
                                           double r_max) {
  const uint nsamples = r.size(0);
  const int noutputs = coeffs.size(2);
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

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  dim3 gdim(find_integer_divisor(nsamples, NWARPS_PER_BLOCK));

  dim3 bdim(32, NWARPS_PER_BLOCK, 1);

  AT_DISPATCH_FLOATING_TYPES(
      r.scalar_type(), "evaluate_spline", ([&] {
        size_t space = 0;
        void *sptr;

        if (r.requires_grad()) {
          evaluate_spline_kernel<scalar_t, true><<<gdim, bdim, space, stream>>>(
              r.packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>(),
              r_knots
                  .packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>(),
              coeffs.data_ptr<scalar_t>(), r_width, r_max,
              R_out.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
              R_deriv
                  .packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>());
        } else {
          evaluate_spline_kernel<scalar_t,
                                 false><<<gdim, bdim, space, stream>>>(
              r.packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>(),
              r_knots
                  .packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>(),
              coeffs.data_ptr<scalar_t>(), r_width, r_max,
              R_out.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
              R_deriv
                  .packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>());
        }
      }));

  if (r.requires_grad()) {
    return {R_out, R_deriv};
  } else {
    return {R_out};
  }
}

template <typename scalar_t>
__global__ void backward_spline_kernel(
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits>
        grad_output,
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits>
        R_deriv,
    torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits>
        r_grad) {

  const int laneID = threadIdx.x % WARP_SIZE;
  const int warpID = threadIdx.x / WARP_SIZE;

  const int i = blockIdx.x * NWARPS_PER_BLOCK + warpID;

  scalar_t reg_r_deriv[4] = {0.0};
  scalar_t reg_grad_output[4] = {0.0};
  scalar_t reg_output = 0.0;

  if (i >= r_grad.size(0)) {
    return;
  }
  // try to fix reduction precision issues here...

  for (int j = laneID; j < R_deriv.size(1); j += WARP_SIZE * 4) {

    for (int l = 0; l < 4; l++) {
      reg_r_deriv[l] = R_deriv[i][l * WARP_SIZE + j];
      reg_grad_output[l] = grad_output[i][l * WARP_SIZE + j];
    }

    for (int l = 0; l < 4; l++) {
      reg_output += reg_r_deriv[l] * reg_grad_output[l];
    }
  }

  for (int offset = 16; offset > 0; offset /= 2) {
    reg_output += __shfl_down_sync(FULL_MASK, reg_output, offset, WARP_SIZE);
  }

  if (laneID == 0) {
    r_grad[i] = reg_output;
  }
}

torch::Tensor backward_spline(torch::Tensor grad_output,
                              torch::Tensor R_deriv) {
  const uint nsamples = R_deriv.size(0);

  torch::Tensor r_grad = torch::empty(
      {nsamples},
      torch::TensorOptions().dtype(R_deriv.dtype()).device(R_deriv.device()));

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  dim3 gdim(find_integer_divisor(nsamples, NWARPS_PER_BLOCK));

  dim3 bdim(WARP_SIZE * NWARPS_PER_BLOCK, 1, 1);

  AT_DISPATCH_FLOATING_TYPES(
      R_deriv.scalar_type(), "backward_spline", ([&] {
        size_t space = 0;
        void *sptr;
        backward_spline_kernel<scalar_t><<<gdim, bdim, space, stream>>>(
            grad_output
                .packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
            R_deriv.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
            r_grad.packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>());
      }));

  return r_grad;
}