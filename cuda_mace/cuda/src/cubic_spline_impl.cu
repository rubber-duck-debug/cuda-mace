#define FULL_MASK 0xffffffff
#define WARP_SIZE 32
#define NWARPS_PER_BLOCK 4

template <typename scalar_t, bool evaluate_deriv>
__global__ void evaluate_spline_kernel_ptr(scalar_t *r, scalar_t *r_knots,
                                           scalar_t *coeff, const int nsamples,
                                           const int nknots, const int noutputs,
                                           const float r_width,
                                           const float r_max, scalar_t *R_out,
                                           scalar_t *R_deriv) {

  extern __shared__ char buffer[];

  const int i = blockIdx.x * NWARPS_PER_BLOCK + threadIdx.y;

  if (i >= nsamples) {
    return;
  }

  const scalar_t r_i = r[i];

  if (r_i > r_max) {
    for (int j = threadIdx.x; j < noutputs; j += blockDim.x) {

      R_out[i * noutputs + j] = 0.0;

      if (evaluate_deriv) {
        R_deriv[i * noutputs + j] = 0.0;
      }
    }
    return;
  }

  int idx = (int)(r_i / r_width);

  if (idx < 0) {
    idx = 0;
  } else if (idx >= nknots) {
    idx = nknots - 1;
  }

  const scalar_t x = r_i - r_knots[idx];
  const scalar_t xx = x * x;
  const scalar_t xxx = xx * x;

  __syncthreads();

  for (int j = threadIdx.x; j < noutputs; j += blockDim.x) {

    scalar_t coeffs[4] = {0.0};
    // float4 coeffs = reinterpret_cast<float4 *>(coeff)[idx * 4 *
    // R_out.size(1)];
    for (int l = 0; l < 4; l++) {
      coeffs[l] = coeff[idx * 4 * noutputs + l * noutputs + j];
    }

    R_out[i * noutputs + j] =
        coeffs[0] + coeffs[1] * x + coeffs[2] * xx + coeffs[3] * xxx;

    if (evaluate_deriv) {
      R_deriv[i * noutputs + j] = coeffs[1] + scalar_t(2.0) * coeffs[2] * x +
                                  scalar_t(3.0) * coeffs[3] * xx;
    }
  }
}

template <typename scalar_t>
__global__ void backward_spline_kernel_ptr(scalar_t *grad_output,
                                           scalar_t *R_deriv, scalar_t *r_grad,
                                           const int nsamples,
                                           const int noutputs) {

  const int laneID = threadIdx.x % WARP_SIZE;
  const int warpID = threadIdx.x / WARP_SIZE;

  const int i = blockIdx.x * NWARPS_PER_BLOCK + warpID;

  scalar_t reg_r_deriv[4] = {0.0};
  scalar_t reg_grad_output[4] = {0.0};
  scalar_t reg_output = 0.0;

  if (i >= nsamples) {
    return;
  }
  // try to fix reduction precision issues here...

  for (int j = laneID; j < noutputs; j += WARP_SIZE * 4) {

    for (int l = 0; l < 4; l++) {
      reg_r_deriv[l] = R_deriv[i * noutputs + l * WARP_SIZE + j];
      reg_grad_output[l] = grad_output[i * noutputs + l * WARP_SIZE + j];
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
