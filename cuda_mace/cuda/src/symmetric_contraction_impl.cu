#define FULL_MASK 0xffffffff

#define NATOMS_PER_BLOCK 4
#define WARP_SIZE 32

template <typename scalar_t, bool requires_grad>
__global__ void symmetric_contraction_kernel(
    const scalar_t *__restrict__ X, const int *__restrict__ atom_types,
    const short *__restrict__ U3_num_nonzero,
    const int *__restrict__ U3_indices, const float *__restrict__ U3_values,

    const short *__restrict__ U2_num_nonzero,
    const short *__restrict__ U2_indices, const float *__restrict__ U2_values,

    const short *__restrict__ U1_num_nonzero,
    const short *__restrict__ U1_indices,

    const scalar_t *__restrict__ W3, const scalar_t *__restrict__ W2,
    const scalar_t *__restrict__ W1, const int nelements, const int w3_size,
    const int w2_size, const int w1_size, const int u3_maxn_nonsparse,

    const int nnodes, const int nchannels, scalar_t *__restrict__ out,
    scalar_t *__restrict__ grad) {

  extern __shared__ char buffer[];
  void *sptr = buffer;
  unsigned int space = 0;

  const int nl = 16;

  volatile scalar_t *buffer_X =
      shared_array<scalar_t>(blockDim.x * nl, sptr, &space);
  volatile scalar_t *buffer_out =
      shared_array<scalar_t>(blockDim.y * blockDim.x, sptr, &space);
  volatile scalar_t *buffer_W3 =
      shared_array<scalar_t>(w3_size * blockDim.x, sptr, &space);
  volatile scalar_t *buffer_W2 =
      shared_array<scalar_t>(w2_size * blockDim.x, sptr, &space);
  volatile scalar_t *buffer_W1 =
      shared_array<scalar_t>(w1_size * blockDim.x, sptr, &space);

  volatile float *buffer_u3_values =
      shared_array<float>(u3_maxn_nonsparse * nl * nl, sptr, &space);
  volatile float *buffer_u2_values = shared_array<float>(nl * nl, sptr, &space);

  volatile int *buffer_u3_indices =
      shared_array<int>(u3_maxn_nonsparse * nl * nl, sptr, &space);

  volatile short *buffer_u3_nonzeros =
      shared_array<short>(nl * nl, sptr, &space);
  volatile short *buffer_u2_nonzero =
      shared_array<short>(nl * nl, sptr, &space);
  volatile short *buffer_u2_indices =
      shared_array<short>(nl * nl, sptr, &space);

  __syncthreads();

  int channel_id = blockIdx.y * blockDim.x + threadIdx.x;

  int element = atom_types[blockIdx.x];

  for (int i = threadIdx.y; i < w3_size; i += blockDim.y) {
    buffer_W3[i * blockDim.x + threadIdx.x] =
        W3[0 * nelements * w3_size * nchannels + element * w3_size * nchannels +
           i * nchannels + channel_id];

    if (i < w2_size) {
      buffer_W2[i * blockDim.x + threadIdx.x] =
          W2[0 * nelements * w2_size * nchannels +
             element * w2_size * nchannels + i * nchannels + channel_id];
    }

    if (i < w1_size) {
      buffer_W1[i * blockDim.x + threadIdx.x] =
          W1[0 * nelements * w1_size * nchannels +
             element * w1_size * nchannels + i * nchannels + channel_id];
    }

    if (i < nl) {
      buffer_X[i * blockDim.x + threadIdx.x] =
          X[blockIdx.x * 16 * nchannels + i * nchannels + channel_id];

      for (int j = threadIdx.x; j < nl; j += blockDim.x) {
        int num_nonzero_u3 = U3_num_nonzero[i * 16 + j];

        buffer_u3_nonzeros[i * nl + j] = num_nonzero_u3;

        for (int k = 0; k < num_nonzero_u3; k++) {
          buffer_u3_indices[i * (nl * u3_maxn_nonsparse) + (k * nl) + j] =
              U3_indices[k * 16 * 16 + i * 16 +
                         j]; // packed 32 bit integer containing 4 x uint8
                             // indices
          buffer_u3_values[i * (nl * u3_maxn_nonsparse) + (k * nl) + j] =
              U3_values[k * 16 * 16 + i * 16 + j];
        }

        buffer_u2_nonzero[i * nl + j] = U2_num_nonzero[i * 16 + j];
        buffer_u2_indices[i * nl + j] = U2_indices[i * 16 + j];
        buffer_u2_values[i * nl + j] = U2_values[i * 16 + j];
      }
    }
  }

  buffer_out[threadIdx.y * blockDim.x + threadIdx.x] = 0.0;

  __syncthreads();

  scalar_t output_1 = 0.0;

  for (int i = threadIdx.y; i < nl; i += blockDim.y) {
    scalar_t Xi = buffer_X[i * blockDim.x + threadIdx.x];

    scalar_t uw1 = 0.0;

    if (i == 0) {
      uw1 = buffer_W1[0 * blockDim.x + threadIdx.x];
    }

    scalar_t output_2 = 0.0;
    scalar_t deriv1_tmp = 0.0;

    if (requires_grad)
      deriv1_tmp = uw1;

    for (int j = 0; j < nl; j++) {
      scalar_t Xj = buffer_X[j * blockDim.x + threadIdx.x];

      scalar_t uw2 = 0.0;

      if (buffer_u2_nonzero[i * nl + j] > 0) {
        uw2 =
            buffer_u2_values[i * nl + j] *
            buffer_W2[buffer_u2_indices[i * nl + j] * blockDim.x + threadIdx.x];
      }

      // int uw3_num_nonzero = buffer_u3_nonzeros[i * nl + j];

      scalar_t output_3 = 0.0;
      scalar_t deriv_1_j_tmp = 0.0;

      if (requires_grad)
        deriv_1_j_tmp = uw2;

      for (int k = 0; k < buffer_u3_nonzeros[i * nl + j]; k++) {
        int u3_mem_idx = i * (nl * u3_maxn_nonsparse) + (k * nl) + j;

        int compressed_indices = buffer_u3_indices[u3_mem_idx];

        int u3_ldx1 = compressed_indices & 0xFF;
        int u3_kdx = (compressed_indices >> 8) & 0xFF;

        scalar_t w3_1 = buffer_W3[u3_ldx1 * blockDim.x + threadIdx.x];

        scalar_t u3 = buffer_u3_values[u3_mem_idx];

        scalar_t Xk = buffer_X[u3_kdx * blockDim.x + threadIdx.x];

        output_3 += u3 * w3_1 * Xk;

        if (requires_grad) {
          deriv_1_j_tmp +=
              u3 *
              (w3_1 +
               buffer_W3[((compressed_indices >> 24) & 0xFF) * blockDim.x +
                         threadIdx.x] +
               buffer_W3[((compressed_indices >> 16) & 0xFF) * blockDim.x +
                         threadIdx.x]) *
              Xk;
        }
      }
      output_2 += (output_3 + uw2) * Xj;
      deriv1_tmp += (uw2 + deriv_1_j_tmp) * Xj;
    }

    output_1 += (output_2 + uw1) * Xi;

    if (requires_grad) {
      grad[blockIdx.x * 16 * nchannels + i * nchannels + channel_id] =
          deriv1_tmp;
    }

    buffer_out[threadIdx.y * blockDim.x + threadIdx.x] = output_1;

    __syncthreads();

    if (threadIdx.y == 0) {
      scalar_t output = 0.0;

      for (int i = 0; i < blockDim.y; i++) {
        output += buffer_out[i * blockDim.x + threadIdx.x];
      }

      out[blockIdx.x * nchannels + channel_id] = output;
    }
  }
}

template <typename scalar_t>
__global__ void
symm_contraction_backward_kernel(const scalar_t *__restrict__ grad_X,
                                 const scalar_t *__restrict__ grad_input,
                                 const int nnodes, const int nchannels,
                                 scalar_t *__restrict__ grad_output) {
  extern __shared__ char buffer[];

  const int nl = 16;
  unsigned int offset = 0;

  scalar_t *buffer_grad = reinterpret_cast<scalar_t *>(buffer + offset);
  offset += blockDim.x * nl * sizeof(scalar_t);

  int atom_idx = blockIdx.x;
  int channel_id = blockIdx.y * blockDim.x + threadIdx.x;

  for (int sph = threadIdx.y; sph < nl; sph += blockDim.y) {
    buffer_grad[sph * blockDim.x + threadIdx.x] = 0.0;
  }

  __syncthreads();

  scalar_t grad = grad_input[atom_idx * nchannels + channel_id];

  for (int sph = threadIdx.y; sph < nl; sph += blockDim.y) {
    buffer_grad[sph * blockDim.x + threadIdx.x] +=
        grad * grad_X[atom_idx * 16 * nchannels + sph * nchannels + channel_id];
  }

  __syncthreads();

  for (int sph = threadIdx.y; sph < nl; sph += blockDim.y) {
    grad_output[atom_idx * 16 * nchannels + sph * nchannels + channel_id] =
        buffer_grad[sph * blockDim.x + threadIdx.x];
  }
}