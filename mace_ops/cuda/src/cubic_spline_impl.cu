#include "cuda_runtime.h"
#include <cuda.h>

#include "cubic_spline_impl.cuh"
#include "cuda_utils.cuh"

using namespace std;
using namespace torch::autograd;
using namespace torch::indexing;

#define FULL_MASK 0xffffffff
#define WARP_SIZE 32
#define NWARPS_PER_BLOCK 4

template <typename scalar_t>
__global__ void generate_coefficients_kernel(
    const torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits>
        r,
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits>
        R,
    const torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits>
        r_width,
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> alpha,
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> l,
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> mu,
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> z,
    torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits>
        coeffs /* [(r.shape[0] -1), 4, R.shape[-1]]  */
) {

  const int laneID = threadIdx.x;

  const int nbasis = r.size(0) - 1;

  scalar_t h = r_width[0];
  scalar_t h2 = h * h;
  scalar_t h3 = h2 * h;

  /* compute alpha*/

  for (int j = laneID; j < R.size(1); j += blockDim.x) {
    alpha[0][j] = 0.0;
    for (int i = 1; i < nbasis; i++) {
      alpha[i][j] = (3.0 / h) * (R[i + 1][j] - R[i][j]) -
                    (3.0 / h) * (R[i][j] - R[i - 1][j]);
    }

    l[0][j] = 1.0;
    mu[0][j] = 0.0;
    z[0][j] = 0.0;

    for (int i = 1; i < nbasis; i++) {
      scalar_t l_tmp = 2.0 * (r[i + 1] - r[i - 1]) - h * mu[i - 1][j];
      l[i][j] = l_tmp;
      mu[i][j] = h / l_tmp;
      z[i][j] = (alpha[i][j] - h * z[i - 1][j]) / l_tmp;
    }

    l[nbasis][j] = 1.0;
    z[nbasis][j] = 0.0;

    /* solve for coefficients */
    scalar_t c_curr = 0.0;

    for (int i = nbasis - 1; i > -1; i--) {
      scalar_t c_next = z[i][j] - mu[i][j] * c_curr;

      coeffs[i][0][j] = R[i][j]; // a
      coeffs[i][2][j] = c_next;  // c

      coeffs[i][1][j] =
          (R[i + 1][j] - R[i][j]) / h - h * (c_curr + 2.0 * c_next) / 3.0; // b
      coeffs[i][3][j] = (c_curr - c_next) / (3.0 * h);                     // d

      c_curr = c_next;
    }
  }
}

torch::Tensor generate_coefficients(torch::Tensor r, torch::Tensor R,
                                    torch::Tensor r_width) {
  const uint nbasis = r.size(0) - 1;
  const int noutputs = R.size(1);

  torch::Tensor coeffs =
      torch::zeros({nbasis, 4, noutputs},
                   torch::TensorOptions().dtype(R.dtype()).device(R.device()));

  torch::Tensor alpha =
      torch::zeros({nbasis, noutputs},
                   torch::TensorOptions().dtype(R.dtype()).device(R.device()));

  torch::Tensor l =
      torch::zeros({nbasis + 1, noutputs},
                   torch::TensorOptions().dtype(R.dtype()).device(R.device()));

  torch::Tensor mu =
      torch::zeros({nbasis + 1, noutputs},
                   torch::TensorOptions().dtype(R.dtype()).device(R.device()));

  torch::Tensor z =
      torch::zeros({nbasis + 1, noutputs},
                   torch::TensorOptions().dtype(R.dtype()).device(R.device()));

  dim3 gridDim(1);

  dim3 blockDim(WARP_SIZE * NWARPS_PER_BLOCK, 1, 1);

  AT_DISPATCH_FLOATING_TYPES(
      R.scalar_type(), "generate_coefficients", ([&] {
        size_t space = 0;
        void *sptr;

        generate_coefficients_kernel<scalar_t><<<gridDim, blockDim, space>>>(
            r.packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>(),
            R.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
            r_width.packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>(),
            alpha.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
            l.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
            mu.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
            z.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
            coeffs.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>());
      }));

  // cudaDeviceSynchronize();

  return coeffs;
}

template <typename scalar_t, const bool evaluate_deriv>
__global__ void evaluate_kernel(
    const torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits>
        r,
    const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits>
        coeff,
    const torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits>
        r_width,
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits>
        R_out, /* [(r.shape[0] -1) * 4, R.shape[-1]]  */
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits>
        R_deriv) {

  const int laneID = threadIdx.x % WARP_SIZE;
  const int warpID = threadIdx.x / WARP_SIZE;

  const int i = blockIdx.x * NWARPS_PER_BLOCK + warpID;

  scalar_t reg_coeffs[4 * 4] = {0.0};

  if (i >= r.size(0)) {
    return;
  }

  scalar_t h = r_width[0];

  scalar_t r_i = r[i];
  int k = (int)(r_i / h);

  scalar_t x = r_i - h * k;
  scalar_t xx = x * x;
  scalar_t xxx = xx * x;

  for (int j = laneID; j < R_out.size(1); j += WARP_SIZE * 4) {

#pragma unroll
    for (int l = 0; l < 4; l++) {
      reg_coeffs[l * 4 + 0] = coeff[k][0][l * WARP_SIZE + j];
      reg_coeffs[l * 4 + 1] = coeff[k][1][l * WARP_SIZE + j];
      reg_coeffs[l * 4 + 2] = coeff[k][2][l * WARP_SIZE + j];
      reg_coeffs[l * 4 + 3] = coeff[k][3][l * WARP_SIZE + j];
    }

#pragma unroll
    for (int l = 0; l < 4; l++) {
      R_out[i][l * WARP_SIZE + j] =
          reg_coeffs[l * 4 + 0] + reg_coeffs[l * 4 + 1] * x +
          reg_coeffs[l * 4 + 2] * xx + reg_coeffs[l * 4 + 3] * xxx;

      if (evaluate_deriv) {
        R_deriv[i][l * WARP_SIZE + j] = reg_coeffs[l * 4 + 1] +
                                        2.0 * reg_coeffs[l * 4 + 2] * x +
                                        3 * reg_coeffs[l * 4 + 3] * xx;
      }
    }
  }
}

std::vector<torch::Tensor>
evaluate_spline(torch::Tensor r, torch::Tensor coeffs, torch::Tensor r_width) {
  const uint nbasis = r.size(0);
  const int noutputs = coeffs.size(2);

  // printf("nbasis %d, noutputs %d\n", nbasis, noutputs);

  torch::Tensor R_out =
      torch::empty({nbasis, noutputs},
                   torch::TensorOptions().dtype(r.dtype()).device(r.device()));

  torch::Tensor R_deriv = torch::empty(
      {1, 1}, torch::TensorOptions().dtype(r.dtype()).device(r.device()));

  if (r.requires_grad()) {
    R_deriv = torch::empty(
        {nbasis, noutputs},
        torch::TensorOptions().dtype(r.dtype()).device(r.device()));
  }

  dim3 gridDim(find_integer_divisor(nbasis, NWARPS_PER_BLOCK));

  dim3 blockDim(WARP_SIZE * NWARPS_PER_BLOCK, 1, 1);

  AT_DISPATCH_FLOATING_TYPES(
      r.scalar_type(), "evaluate_spline", ([&] {
        size_t space = 0;
        void *sptr;

        if (r.requires_grad()) {
          evaluate_kernel<scalar_t, true><<<gridDim, blockDim, space>>>(
              r.packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>(),
              coeffs.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
              r_width
                  .packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>(),
              R_out.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
              R_deriv
                  .packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>());
        } else {
          evaluate_kernel<scalar_t, false><<<gridDim, blockDim, space>>>(
              r.packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>(),
              coeffs.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
              r_width
                  .packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>(),
              R_out.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
              R_deriv
                  .packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>());
        }
      }));

  // cudaDeviceSynchronize();

  // cudaError_t err = cudaDeviceSynchronize();
  // if (err != cudaSuccess) {
  //   throw std::runtime_error("CUDA Error: " +
  //                            std::string(cudaGetErrorString(err)));
  // }

  if (r.requires_grad()) {
    return {R_out, R_deriv};
  } else {
    return {R_out};
  }
}

template <typename scalar_t>
__global__ void backward_kernel(
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

  dim3 gridDim(find_integer_divisor(nsamples, NWARPS_PER_BLOCK));

  dim3 blockDim(WARP_SIZE * NWARPS_PER_BLOCK, 1, 1);

  AT_DISPATCH_FLOATING_TYPES(
      R_deriv.scalar_type(), "backward_spline", ([&] {
        size_t space = 0;
        void *sptr;
        backward_kernel<scalar_t><<<gridDim, blockDim, space>>>(
            grad_output
                .packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
            R_deriv.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
            r_grad.packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>());
      }));

  // cudaDeviceSynchronize();

  // cudaError_t err = cudaDeviceSynchronize();
  // if (err != cudaSuccess) {
  //   throw std::runtime_error("CUDA Error: " +
  //                            std::string(cudaGetErrorString(err)));
  // }

  return r_grad;
}