#include "cuda_runtime.h"
#include <cuda.h>

#include "cuda_utils.cuh"
#include "spherical_harmonics_impl.cuh"

using namespace std;
using namespace torch::autograd;
using namespace torch::indexing;

#define FULL_MASK 0xffffffff
#define WARP_SIZE 32
#define NWARPS_PER_BLOCK 4

/*
This code has been temporarily transplanted from sphericart. Code will be modified
to be able to revert back to sphericart implementaton. In the meantime,\
Please **CITE** sphericart if this code is used in any of your work.

https://github.com/lab-cosmo/sphericart

@article{sphericart,
    title={Fast evaluation of spherical harmonics with sphericart},
    author={Bigi, Filippo and Fraux, Guillaume and Browning, Nicholas J. and Ceriotti, Michele},
    journal={J. Chem. Phys.},
    year={2023},
    number={159},
    pages={064802},
}
*/

template <typename scalar_t>
__global__ void spherical_harmonics_kernel(
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits>
        xyz,
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> sph,
    torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits>
        sph_deriv,
    bool normalize, bool requires_grad) {

  extern __shared__ char buffer[];
  const int32_t nsamples = xyz.size(0);

  int32_t laneID = threadIdx.x % WARP_SIZE;
  int32_t warpID = threadIdx.x / WARP_SIZE;

  void *sptr = buffer;
  size_t space = 0;

  const scalar_t sqrt_4pi = 3.5449077018110318;

  scalar_t *buffer_xyz = shared_array<scalar_t>(blockDim.x * 3, sptr, &space);
  scalar_t *buffer_sph = shared_array<scalar_t>(blockDim.x * 16, sptr, &space);

  scalar_t *buffer_sph_deriv_x;
  scalar_t *buffer_sph_deriv_y;
  scalar_t *buffer_sph_deriv_z;

  if (requires_grad) {
    buffer_sph_deriv_x = shared_array<scalar_t>(blockDim.x * 16, sptr, &space);
    buffer_sph_deriv_y = shared_array<scalar_t>(blockDim.x * 16, sptr, &space);
    buffer_sph_deriv_z = shared_array<scalar_t>(blockDim.x * 16, sptr, &space);
  }

  int32_t edge_start = blockIdx.x * blockDim.x;

  for (int i = 0; i < 3; i++) {
    buffer_xyz[i * blockDim.x + threadIdx.x] =
        (edge_start + threadIdx.x < nsamples) ? xyz[edge_start + threadIdx.x][i]
                                              : 0.0;
  }

  __syncthreads();

  // scalar_t x = buffer_xyz[0 * blockDim.x + threadIdx.x];
  // scalar_t y = buffer_xyz[1 * blockDim.x + threadIdx.x];
  // scalar_t z = buffer_xyz[2 * blockDim.x + threadIdx.x];

  // MACE ordering x[:, [2, 0, 1]]
  scalar_t x = buffer_xyz[2 * blockDim.x + threadIdx.x];
  scalar_t y = buffer_xyz[0 * blockDim.x + threadIdx.x];
  scalar_t z = buffer_xyz[1 * blockDim.x + threadIdx.x];

  __syncthreads();

  scalar_t x2 = x * x;
  scalar_t y2 = y * y;
  scalar_t z2 = z * z;

  scalar_t ir = 0.0;

  if (normalize) {
    scalar_t ir2 = 1.0 / (x2 + y2 + z2);
    ir = sqrt(ir2);
    x *= ir;
    y *= ir;
    z *= ir;
    x2 *= ir2;
    y2 *= ir2;
    z2 *= ir2;
  }

  buffer_sph[0 * blockDim.x + threadIdx.x] = 0.282094791773878;

  buffer_sph[1 * blockDim.x + threadIdx.x] = 0.48860251190292 * y;
  buffer_sph[2 * blockDim.x + threadIdx.x] = 0.48860251190292 * z;
  buffer_sph[3 * blockDim.x + threadIdx.x] = 0.48860251190292 * x;

  auto tmp = 2.23606797749979 * x;

  buffer_sph[4 * blockDim.x + threadIdx.x] =
      tmp * buffer_sph[1 * blockDim.x + threadIdx.x];
  buffer_sph[5 * blockDim.x + threadIdx.x] =
      2.23606797749979 * z * buffer_sph[1 * blockDim.x + threadIdx.x];
  buffer_sph[6 * blockDim.x + threadIdx.x] =
      -0.315391565252520 * (x2 + y2 - 2 * z2);
  buffer_sph[7 * blockDim.x + threadIdx.x] =
      tmp * buffer_sph[2 * blockDim.x + threadIdx.x];
  buffer_sph[8 * blockDim.x + threadIdx.x] = 0.54627421529604 * (x2 - y2);

  buffer_sph[9 * blockDim.x + threadIdx.x] =
      -0.59004358992664 * y * (y2 - 3 * x2);
  buffer_sph[10 * blockDim.x + threadIdx.x] =
      2.64575131106459 * z * buffer_sph[4 * blockDim.x + threadIdx.x];
  tmp = -0.457045799464466 * (x2 + y2 - 4 * z2);
  buffer_sph[11 * blockDim.x + threadIdx.x] = y * tmp;
  buffer_sph[12 * blockDim.x + threadIdx.x] =
      -1.49270533036046 * z *
      (z2 - 2.37799637856361 * buffer_sph[6 * blockDim.x + threadIdx.x]);
  buffer_sph[13 * blockDim.x + threadIdx.x] = x * tmp;
  buffer_sph[14 * blockDim.x + threadIdx.x] = 1.44530572132028 * z * (x2 - y2);
  buffer_sph[15 * blockDim.x + threadIdx.x] =
      0.59004358992664 * x * (x2 - 3 * y2);

  __syncthreads();

  for (int i = warpID; i < 16; i += NWARPS_PER_BLOCK) {
    for (int j = laneID; j < blockDim.x; j += WARP_SIZE) {

      if (edge_start + j < xyz.size(0)) {
        sph[i][edge_start + j] = sqrt_4pi * buffer_sph[i * blockDim.x + j];
      }
    }
  }

  if (requires_grad) {
    // dx components first...
    // l = 0
    buffer_sph_deriv_x[0 * blockDim.x + threadIdx.x] = 0.0;
    buffer_sph_deriv_y[0 * blockDim.x + threadIdx.x] = 0.0;
    buffer_sph_deriv_z[0 * blockDim.x + threadIdx.x] = 0.0;
    // l = 1
    buffer_sph_deriv_x[1 * blockDim.x + threadIdx.x] = 0.0;
    buffer_sph_deriv_x[2 * blockDim.x + threadIdx.x] = 0.0;
    buffer_sph_deriv_x[3 * blockDim.x + threadIdx.x] = 0.48860251190292;

    buffer_sph_deriv_y[1 * blockDim.x + threadIdx.x] = 0.48860251190292;
    buffer_sph_deriv_y[2 * blockDim.x + threadIdx.x] = 0.0;
    buffer_sph_deriv_y[3 * blockDim.x + threadIdx.x] = 0.0;

    buffer_sph_deriv_z[1 * blockDim.x + threadIdx.x] = 0.0;
    buffer_sph_deriv_z[2 * blockDim.x + threadIdx.x] = 0.48860251190292;
    buffer_sph_deriv_z[3 * blockDim.x + threadIdx.x] = 0.0;

    // l = 2
    buffer_sph_deriv_x[4 * blockDim.x + threadIdx.x] =
        2.23606797749979 * buffer_sph[1 * blockDim.x + threadIdx.x];
    buffer_sph_deriv_x[5 * blockDim.x + threadIdx.x] = 0.0;
    buffer_sph_deriv_x[6 * blockDim.x + threadIdx.x] =
        -1.29099444873581 * buffer_sph[3 * blockDim.x + threadIdx.x];
    buffer_sph_deriv_x[7 * blockDim.x + threadIdx.x] =
        2.23606797749979 * buffer_sph[2 * blockDim.x + threadIdx.x];
    buffer_sph_deriv_x[8 * blockDim.x + threadIdx.x] =
        2.23606797749979 * buffer_sph[3 * blockDim.x + threadIdx.x];

    buffer_sph_deriv_y[4 * blockDim.x + threadIdx.x] =
        -1.73205080756888 * buffer_sph_deriv_x[6 * blockDim.x + threadIdx.x];
    buffer_sph_deriv_y[5 * blockDim.x + threadIdx.x] =
        buffer_sph_deriv_x[7 * blockDim.x + threadIdx.x];
    buffer_sph_deriv_y[6 * blockDim.x + threadIdx.x] =
        -0.577350269189626 * buffer_sph_deriv_x[4 * blockDim.x + threadIdx.x];
    buffer_sph_deriv_y[7 * blockDim.x + threadIdx.x] = 0.0;
    buffer_sph_deriv_y[8 * blockDim.x + threadIdx.x] =
        -buffer_sph_deriv_x[4 * blockDim.x + threadIdx.x];

    buffer_sph_deriv_z[4 * blockDim.x + threadIdx.x] = 0.0;
    buffer_sph_deriv_z[5 * blockDim.x + threadIdx.x] =
        buffer_sph_deriv_x[4 * blockDim.x + threadIdx.x];
    buffer_sph_deriv_z[6 * blockDim.x + threadIdx.x] =
        1.15470053837925 * buffer_sph_deriv_x[7 * blockDim.x + threadIdx.x];
    buffer_sph_deriv_z[7 * blockDim.x + threadIdx.x] =
        buffer_sph_deriv_y[4 * blockDim.x + threadIdx.x];
    buffer_sph_deriv_z[8 * blockDim.x + threadIdx.x] = 0.0;

    // l = 3
    buffer_sph_deriv_x[9 * blockDim.x + threadIdx.x] =
        3.24037034920393 * buffer_sph[4 * blockDim.x + threadIdx.x];
    buffer_sph_deriv_x[10 * blockDim.x + threadIdx.x] =
        2.64575131106459 * buffer_sph[5 * blockDim.x + threadIdx.x];
    buffer_sph_deriv_x[11 * blockDim.x + threadIdx.x] =
        -0.83666002653408 * buffer_sph[4 * blockDim.x + threadIdx.x];
    buffer_sph_deriv_x[12 * blockDim.x + threadIdx.x] =
        -2.04939015319192 * buffer_sph[7 * blockDim.x + threadIdx.x];
    buffer_sph_deriv_x[13 * blockDim.x + threadIdx.x] =
        0.91409159892893 *
        (y2 - z2 + 4.75599275712721 * buffer_sph[6 * blockDim.x + threadIdx.x]);
    buffer_sph_deriv_x[14 * blockDim.x + threadIdx.x] =
        2.64575131106459 * buffer_sph[7 * blockDim.x + threadIdx.x];
    buffer_sph_deriv_x[15 * blockDim.x + threadIdx.x] =
        3.24037034920393 * buffer_sph[8 * blockDim.x + threadIdx.x];

    buffer_sph_deriv_y[9 * blockDim.x + threadIdx.x] =
        buffer_sph_deriv_x[15 * blockDim.x + threadIdx.x];
    buffer_sph_deriv_y[10 * blockDim.x + threadIdx.x] =
        buffer_sph_deriv_x[14 * blockDim.x + threadIdx.x];
    buffer_sph_deriv_y[11 * blockDim.x + threadIdx.x] =
        -0.91409159892893 *
        (y2 - z2 - 1.58533091904240 * buffer_sph[6 * blockDim.x + threadIdx.x]);
    buffer_sph_deriv_y[12 * blockDim.x + threadIdx.x] =
        -2.04939015319192 * buffer_sph[5 * blockDim.x + threadIdx.x];
    buffer_sph_deriv_y[13 * blockDim.x + threadIdx.x] =
        -0.83666002653408 * buffer_sph[4 * blockDim.x + threadIdx.x];
    buffer_sph_deriv_y[14 * blockDim.x + threadIdx.x] =
        -buffer_sph_deriv_x[10 * blockDim.x + threadIdx.x];
    buffer_sph_deriv_y[15 * blockDim.x + threadIdx.x] =
        -buffer_sph_deriv_x[9 * blockDim.x + threadIdx.x];

    buffer_sph_deriv_z[9 * blockDim.x + threadIdx.x] = 0.0;
    buffer_sph_deriv_z[10 * blockDim.x + threadIdx.x] =
        2.64575131106459 * buffer_sph[4 * blockDim.x + threadIdx.x];
    buffer_sph_deriv_z[11 * blockDim.x + threadIdx.x] =
        3.34664010613630 * buffer_sph[5 * blockDim.x + threadIdx.x];
    buffer_sph_deriv_z[12 * blockDim.x + threadIdx.x] =
        3.54964786985977 * buffer_sph[6 * blockDim.x + threadIdx.x];
    buffer_sph_deriv_z[13 * blockDim.x + threadIdx.x] =
        3.34664010613630 * buffer_sph[7 * blockDim.x + threadIdx.x];
    buffer_sph_deriv_z[14 * blockDim.x + threadIdx.x] =
        2.64575131106459 * buffer_sph[8 * blockDim.x + threadIdx.x];
    buffer_sph_deriv_z[15 * blockDim.x + threadIdx.x] = 0.0;

    __syncthreads();

    for (int j = laneID; j < blockDim.x; j += WARP_SIZE) {

      if (edge_start + j < xyz.size(0)) {

        for (int i = warpID; i < 16; i += NWARPS_PER_BLOCK) {

          // MACE ordering x[:, [2, 0, 1]]

          scalar_t tmp_dx = buffer_sph_deriv_x[i * blockDim.x + j];
          scalar_t tmp_dy = buffer_sph_deriv_y[i * blockDim.x + j];
          scalar_t tmp_dz = buffer_sph_deriv_z[i * blockDim.x + j];

          // corrects derivatives for normalization
          if (normalize) {

            scalar_t x = buffer_xyz[2 * blockDim.x + j];
            scalar_t y = buffer_xyz[0 * blockDim.x + j];
            scalar_t z = buffer_xyz[1 * blockDim.x + j];

            scalar_t x2 = x * x;
            scalar_t y2 = y * y;
            scalar_t z2 = z * z;

            scalar_t ir2 = 1.0 / (x2 + y2 + z2);

            scalar_t ir = sqrt(ir2);
            x *= ir;
            y *= ir;
            z *= ir;

            scalar_t tmp_n = (tmp_dx * x + tmp_dy * y + tmp_dz * z);

            scalar_t new_tmp_dx = (tmp_dx - x * tmp_n) * ir;
            scalar_t new_tmp_dy = (tmp_dy - y * tmp_n) * ir;
            scalar_t new_tmp_dz = (tmp_dz - z * tmp_n) * ir;

            sph_deriv[i][0][edge_start + j] = sqrt_4pi * new_tmp_dx;
            sph_deriv[i][1][edge_start + j] = sqrt_4pi * new_tmp_dy;
            sph_deriv[i][2][edge_start + j] = sqrt_4pi * new_tmp_dz;

          } else {
            sph_deriv[i][0][edge_start + j] = sqrt_4pi * tmp_dx;
            sph_deriv[i][1][edge_start + j] = sqrt_4pi * tmp_dy;
            sph_deriv[i][2][edge_start + j] = sqrt_4pi * tmp_dz;
          }
        }
      }
    }
  }
}

std::vector<torch::Tensor> spherical_harmonics(torch::Tensor xyz) {
  const int32_t nsamples = xyz.size(0);

  torch::Tensor sph_harmonics = torch::empty(
      {16, nsamples},
      torch::TensorOptions().dtype(xyz.dtype()).device(xyz.device()));

  // if (xyz.requires_grad()) {
  torch::Tensor sph_harmonics_deriv = torch::empty(
      {16, 3, nsamples},
      torch::TensorOptions().dtype(xyz.dtype()).device(xyz.device()));
  //}

  dim3 gridDim(find_integer_divisor(nsamples, WARP_SIZE * NWARPS_PER_BLOCK));

  dim3 blockDim(WARP_SIZE * NWARPS_PER_BLOCK, 1, 1);

  AT_DISPATCH_FLOATING_TYPES(
      xyz.scalar_type(), "spherical_harmonics", ([&] {
        size_t space = 0;
        void *sptr;

        shared_array<scalar_t>(WARP_SIZE * NWARPS_PER_BLOCK * 3, sptr, &space);
        shared_array<scalar_t>(WARP_SIZE * NWARPS_PER_BLOCK * 16, sptr, &space);

        if (xyz.requires_grad()) {

          shared_array<scalar_t>(WARP_SIZE * NWARPS_PER_BLOCK * 16 * 3, sptr,
                                 &space);
          spherical_harmonics_kernel<scalar_t><<<gridDim, blockDim, space>>>(
              xyz.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
              sph_harmonics
                  .packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
              sph_harmonics_deriv
                  .packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
              true, true);
        } else {
          spherical_harmonics_kernel<scalar_t><<<gridDim, blockDim, space>>>(
              xyz.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
              sph_harmonics
                  .packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
              sph_harmonics_deriv
                  .packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
              true, false);
        }
      }));

  if (xyz.requires_grad()) {
    return {sph_harmonics, sph_harmonics_deriv};
  } else {
    return {sph_harmonics};
  }
}

template <typename scalar_t>
__global__ void spherical_harmonics_backward_kernel(
    const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits>
        sph_deriv,
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits>
        grad_output,
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits>
        xyz_grad) {

  extern __shared__ char buffer[];
  const int32_t nsamples = sph_deriv.size(2);

  int32_t laneID = threadIdx.x % 16;
  int32_t warpID = threadIdx.x / 16;

  void *sptr = buffer;
  size_t space = 0;

  scalar_t *buffer_sum = shared_array<scalar_t>(blockDim.x * 3, sptr, &space);

  // sph_deriv: 16, 3, nsamples;
  // grad: 16, nsamples
  //  xyz: nsamples, 3

  int32_t k_to_idx[3] = {2, 0, 1};

  int32_t edge_start = blockIdx.x * blockDim.x;

  for (int j = warpID; j < blockDim.x; j += 8) {

    scalar_t g =
        (edge_start + j < nsamples) ? grad_output[laneID][edge_start + j] : 0.0;

    for (int k = 0; k < 3; k++) {

      scalar_t sph = (edge_start + j < nsamples)
                         ? sph_deriv[laneID][k][edge_start + j]
                         : 0.0;

      scalar_t prod = sph * g;

      // reduce across the sub-warp
      for (int offset = 8; offset > 0; offset /= 2) {
        prod += __shfl_down_sync(FULL_MASK, prod, offset);
      }

      if (laneID == 0) {
        if (edge_start + j < nsamples)
          xyz_grad[edge_start + j][k_to_idx[k]] = prod;
      }
    }
  }
}

torch::Tensor spherical_harmonics_backward(torch::Tensor sph_deriv,
                                           torch::Tensor grad_output) {
  const int32_t nsamples = sph_deriv.size(2);

  torch::Tensor xyz_grad =
      torch::empty({nsamples, 3}, torch::TensorOptions()
                                      .dtype(sph_deriv.dtype())
                                      .device(sph_deriv.device()));

  dim3 gridDim(find_integer_divisor(nsamples, WARP_SIZE * NWARPS_PER_BLOCK));

  dim3 blockDim(WARP_SIZE * NWARPS_PER_BLOCK, 1, 1);

  AT_DISPATCH_FLOATING_TYPES(
      sph_deriv.scalar_type(), "spherical_harmonics_backward", ([&] {
        size_t space = 0;
        void *sptr;

        shared_array<scalar_t>(WARP_SIZE * NWARPS_PER_BLOCK * 3, sptr, &space);
        spherical_harmonics_backward_kernel<scalar_t>
            <<<gridDim, blockDim, space>>>(
                sph_deriv
                    .packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                grad_output
                    .packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                xyz_grad.packed_accessor64<scalar_t, 2,
                                           torch::RestrictPtrTraits>());
      }));

  return xyz_grad;
}