#define FULL_MASK 0xffffffff

#define WARP_SIZE 32
#define NWARPS_PER_BLOCK 4
#define NEIGHBOUR_NEDGES_PER_BLOCK 512
#define NELEMENTS_PER_BLOCK 512

__global__ void calculate_first_occurences_kernel_ptr(
    const int *__restrict__ receiver_list,
    const int nelements_input,
    const int nelements_output,
    const int * sort_idx,
    const bool use_sort,
    int * __restrict__ first_occurences_start,
    int * __restrict__ first_occurences_end)
{

  extern __shared__ char buffer[];
  unsigned int offset = 0;
  int *smem = reinterpret_cast<int *>(buffer + offset);

  int block_start = blockIdx.x * NELEMENTS_PER_BLOCK;

  // load all elements of senderlist needed by block into shared memory
  for (int i = threadIdx.x; i < NELEMENTS_PER_BLOCK + 1; i += blockDim.x)
  {
    int idx = block_start + i;

    if (idx < nelements_input)
    {
      if (use_sort)
      {
        smem[i] = receiver_list[sort_idx[idx]];
      }
      else
      {
        smem[i] = receiver_list[idx];
      }
    }
  }

  __syncthreads();

  // deal with even boundaries
  for (int i = 2 * threadIdx.x; i < NELEMENTS_PER_BLOCK; i += 2 * blockDim.x)
  {
    int idx = block_start + i;

    if (idx + 1 < nelements_input)
    {
      int loc1 = smem[i];
      int loc2 = smem[i + 1];

      if (loc1 != loc2)
      {
        first_occurences_end[loc1] = idx + 1;
        first_occurences_start[loc2] = idx + 1;
      }
    }
  }

  // deal with odd boundaries
  for (int i = 2 * threadIdx.x + 1; i < NELEMENTS_PER_BLOCK + 1; i += 2 * blockDim.x)
  {
    int idx = block_start + i;

    if (idx + 1 < nelements_input)
    {
      int loc1 = smem[i];
      int loc2 = smem[i + 1];

      if (loc1 != loc2)
      {
        first_occurences_end[loc1] = idx + 1;
        first_occurences_start[loc2] = idx + 1;
      }
    }
  }

  // deal with 0th and last element specifically, so we dont need to use torch::zeros
  if (blockIdx.x == 0 && threadIdx.x == 0)
  {
    first_occurences_start[receiver_list[0]] = 0;
    first_occurences_end[receiver_list[nelements_input - 1]] = nelements_input;
  }
}

template <typename scalar_t, const int TM, const int TN>
__global__ void inv_tp_kernel_ptr(
    const scalar_t *__restrict__ X, const scalar_t *__restrict__ Y,
    const scalar_t *__restrict__ radial, const int *__restrict__ sender_list,
    const int *__restrict__ receiver_list,
    const int *__restrict__ first_occurences, int *__restrict__ node_edge_index,
    const int nedges, const int nchannels, const int nnodes,
    scalar_t *__restrict__ output) {

  extern __shared__ char buffer[];

  void *sptr = buffer;
  unsigned int space = 0;

  int *buffer_sender = shared_array<int>(512, sptr, &space);

  const int threadCol = threadIdx.x % WARP_SIZE;
  const int threadRow = threadIdx.x / WARP_SIZE;

  const int edge_start = first_occurences[blockIdx.x];
  const int edge_end = first_occurences[nnodes + blockIdx.x];
  const int node_index = receiver_list[edge_start];

  scalar_t regY[TM] = {0.0};
  scalar_t regX[TN] = {0.0};
  scalar_t regRadial[TM * TN] = {0.0};
  scalar_t regOut[TN * TM] = {0.0};
  scalar_t regC[TN * TM] = {0.0};
  // check if this node has neighbours
  if (edge_end - edge_start == 0) {
    return;
  }

  for (int tid = threadIdx.x; tid < edge_end - edge_start; tid += blockDim.x) {
    buffer_sender[tid] = sender_list[edge_start + tid];
  }

  __syncthreads();

  for (int edge = edge_start + threadIdx.x; edge < edge_end;
       edge += blockDim.x) {
    int sender = buffer_sender[edge - edge_start];
    node_edge_index[sender * nnodes + node_index] = edge;
  }

  for (int feature = threadCol; feature < nchannels;
       feature += WARP_SIZE * TN) {
    for (int m = threadRow; m < 16; m += NWARPS_PER_BLOCK * TM) {

      for (int i = 0; i < TN; i++) {
        for (int j = 0; j < TM; j++) {
          regC[i * TM + j] = 0.0;
          regOut[i * TM + j] = 0.0;
        }
      }

      for (int edge = edge_start; edge < edge_end; edge++) {
        for (int i = 0; i < TN; i++) {
          regX[i] = X[buffer_sender[edge - edge_start] * nchannels +
                      i * WARP_SIZE + feature];
        }
        for (int i = 0; i < TN; i++) {
          for (int j = 0; j < TM; j++) {
            int lm_index = (int)sqrt((float)j * NWARPS_PER_BLOCK + m);

            regRadial[i * TM + j] =
                radial[edge * 4 * nchannels + lm_index * nchannels +
                       i * WARP_SIZE + feature];
          }
        }

        for (int j = 0; j < TM; j++) {
          regY[j] = Y[(j * NWARPS_PER_BLOCK + m) * nedges + edge];
        }

        for (int i = 0; i < TN; i++) {
          for (int j = 0; j < TM; j++) {
            scalar_t val = regX[i] * regY[j] * regRadial[i * TM + j];
            scalar_t val_compensated = val - regC[i * TM + j];
            scalar_t tmp_new = regOut[i * TM + j] + val_compensated;
            regC[i * TM + j] = (tmp_new - regOut[i * TM + j]) - val_compensated;
            regOut[i * TM + j] = tmp_new;
          }
        }
      }

      __syncthreads();

      for (int i = 0; i < TN; i++) {
        for (int j = 0; j < TM; j++) {
          output[node_index * 16 * nchannels +
                 (j * NWARPS_PER_BLOCK + m) * nchannels + i * WARP_SIZE +
                 feature] = regOut[i * TM + j];
        }
      }
    }
  }
}

template <typename scalar_t, const int TM, const int TN>
__global__ void backward_edge_inv_tp_kernel_ptr(
    const scalar_t *__restrict__ X, const scalar_t *__restrict__ Y,
    const scalar_t *__restrict__ radial, const scalar_t *__restrict__ grad_in,
    const int *__restrict__ sender_list, const int *__restrict__ receiver_list,
    const int *__restrict__ first_occurences, const int nedges,
    const int nchannels, const int nnodes, scalar_t *__restrict__ gradY,
    scalar_t *__restrict__ grad_radial) {

  extern __shared__ char buffer[];

  void *sptr = buffer;
  unsigned int space = 0;

  scalar_t *buffer_grad_in =
      shared_array<scalar_t>(16 * nchannels, sptr, &space);
  scalar_t *buffer_Y =
      shared_array<scalar_t>(NWARPS_PER_BLOCK * 16, sptr, &space);
  scalar_t *buffer_dY =
      shared_array<scalar_t>(NWARPS_PER_BLOCK * 16, sptr, &space);

  const int threadCol = threadIdx.x % WARP_SIZE;
  const int threadRow = threadIdx.x / WARP_SIZE;

  const int edge_start = first_occurences[blockIdx.x];
  const int node_index = receiver_list[edge_start];
  const int edge_end = first_occurences[nnodes + blockIdx.x];

  const int N_start = blockIdx.y * TN * WARP_SIZE;

  scalar_t regX[TN] = {0.0};
  scalar_t regW[4 * TN] = {0.0};

  scalar_t regGradW[4 * TN] = {0.0};

  if (edge_end - edge_start == 0) {
    return;
  }

  for (int m = 0; m < 16 / NWARPS_PER_BLOCK; m++) {
    for (int n = 0; n < TN; n++) {
      buffer_grad_in[(m * NWARPS_PER_BLOCK + threadRow) * nchannels +
                     n * WARP_SIZE + threadCol] =
          grad_in[node_index * 16 * nchannels +
                  (m * NWARPS_PER_BLOCK + threadRow) * nchannels + N_start +
                  n * WARP_SIZE + threadCol];
    }
  }

  __syncthreads();

  int niter = find_integer_divisor(edge_end - edge_start, NWARPS_PER_BLOCK);

  for (int ni = 0; ni < niter; ni++) {
    int edge = edge_start + ni * NWARPS_PER_BLOCK + threadRow;

    if (edge < edge_end) {
      int sender_id = sender_list[edge];

      if (threadCol < 16) {
        buffer_Y[threadCol * NWARPS_PER_BLOCK + threadRow] =
            Y[threadCol * nedges + edge];
        buffer_dY[threadCol * NWARPS_PER_BLOCK + threadRow] = 0.0;
      }

      __syncwarp();

      for (int n = 0; n < TN; n++) {

        regX[n] =
            X[sender_id * nchannels + N_start + n * WARP_SIZE + threadCol];
      }

      for (int n = 0; n < TN; n++) {
        for (int L = 0; L < 4; L++) {
          regGradW[L * TN + n] = 0.0;
          regW[L * TN + n] = radial[edge * 4 * nchannels + L * nchannels +
                                    N_start + n * WARP_SIZE + threadCol];
        }
      }

      for (int L = 0; L < 4; L++) {
        int mstart = L * L;
        int mend = (L + 1) * (L + 1);

        for (int m = mstart; m < mend; m++) {
          scalar_t sph = buffer_Y[m * NWARPS_PER_BLOCK + threadRow];
          scalar_t dgradY = 0.0;

          for (int n = 0; n < TN; n++) {
            scalar_t gradin =
                buffer_grad_in[m * nchannels + n * WARP_SIZE + threadCol];
            scalar_t w = regW[L * TN + n];

            regGradW[L * TN + n] += sph * regX[n] * gradin;

            dgradY += gradin * w * regX[n];
          }

          for (int offset = 16; offset > 0; offset /= 2) {
            dgradY += __shfl_down_sync(FULL_MASK, dgradY, offset, WARP_SIZE);
          }

          // threadIdx % WARP_SIZE = 0 dgradY contains the derivative of the
          // output wrt. Y
          if (threadCol == 0)
            buffer_dY[m * NWARPS_PER_BLOCK + threadRow] = dgradY;
        }
      }
    }

    __syncthreads();

    if (edge < edge_end) {
      if (threadCol < 16) {
        gradY[threadCol * nedges + edge] =
            buffer_dY[threadCol * NWARPS_PER_BLOCK + threadRow];
      }

      for (int n = 0; n < TN; n++) {
        //  write grad_radial
        for (int L = 0; L < 4; L++) {
          grad_radial[edge * 4 * nchannels + L * nchannels + N_start +
                      n * WARP_SIZE + threadCol] = regGradW[L * TN + n];
        }
      }
    }
  }
}

template <typename scalar_t, const int TM, const int TN>
__global__ void backward_node_inv_tp_kernel_ptr(
    const scalar_t *__restrict__ Y, const scalar_t *__restrict__ radial,
    const scalar_t *__restrict__ grad_in, const int *__restrict__ sender_list,
    const int *__restrict__ receiver_list,
    const int *__restrict__ first_occurences,
    const int *__restrict__ node_edge_index, const int nedges,
    const int nchannels, const int nnodes, scalar_t *__restrict__ gradX) {

  extern __shared__ char buffer[];

  void *sptr = buffer;
  unsigned int space = 0;

  scalar_t regY[TM] = {0.0};
  scalar_t regRadial[TM * TN] = {0.0};
  scalar_t regGradIn[TM * TN] = {0.0};
  scalar_t regGradX[TN] = {0.0};
  scalar_t regGradXC[TN] = {0.0};

  scalar_t *buffer_out =
      shared_array<scalar_t>(NWARPS_PER_BLOCK * WARP_SIZE, sptr, &space);
  int *buffer_sorted_sender_idx = shared_array<int>(512, sptr, &space);
  int *buffer_receiver_list = shared_array<int>(512, sptr, &space);

  const int threadCol = threadIdx.x % WARP_SIZE;
  const int threadRow = threadIdx.x / WARP_SIZE;

  const int edge_start = first_occurences[blockIdx.x];
  const int node_index = receiver_list[edge_start];
  const int edge_end = first_occurences[nnodes + blockIdx.x];

  if (edge_end - edge_start == 0) {
    return;
  }

  for (int tid = threadIdx.x; tid < edge_end - edge_start; tid += blockDim.x) {
    int sender = sender_list[edge_start + tid];
    int sorted_id = node_edge_index[node_index * nnodes + sender];
    buffer_sorted_sender_idx[tid] = sorted_id;
    buffer_receiver_list[tid] = receiver_list[sorted_id];
  }

  __syncthreads();

  for (int feature = threadCol; feature < nchannels;
       feature += WARP_SIZE * TN) {

    __syncthreads();

    for (int i = 0; i < TN; i++) {
      regGradX[i] = 0.0;
      regGradXC[i] = 0.0;
    }

    for (int edge = edge_start; edge < edge_end; edge++) {

      int sorted_id = buffer_sorted_sender_idx[edge - edge_start];
      int receiver_id = buffer_receiver_list[edge - edge_start];

      for (int m = threadRow; m < 16; m += NWARPS_PER_BLOCK * TM) {
        for (int j = 0; j < TM; j++) {
          regY[j] = Y[(j * NWARPS_PER_BLOCK + m) * nedges + sorted_id];
        }

        for (int j = 0; j < TM; j++) {
          int lm_index = (int)sqrt((float) j * NWARPS_PER_BLOCK + m);
          for (int i = 0; i < TN; i++) {
            regRadial[i * TM + j] =
                radial[sorted_id * 4 * nchannels + lm_index * nchannels +
                       i * WARP_SIZE + feature];

            regGradIn[i * TM + j] =
                grad_in[receiver_id * 16 * nchannels +
                        (j * NWARPS_PER_BLOCK + m) * nchannels + i * WARP_SIZE +
                        feature];
          }
        }

        for (int i = 0; i < TN; i++) {
          for (int j = 0; j < TM; j++) {
            scalar_t val =
                regGradIn[i * TM + j] * regRadial[i * TM + j] * regY[j];
            scalar_t val_compensated = val - regGradXC[i];
            scalar_t tmp_new = regGradX[i] + val_compensated;
            regGradXC[i] = (tmp_new - regGradX[i]) - val_compensated;
            regGradX[i] = tmp_new;
          }
        }
      }
    }

    for (int i = 0; i < TN; i++) {
      __syncthreads();

      buffer_out[threadRow * WARP_SIZE + threadCol] = regGradX[i];

      __syncthreads();
      /* need to reduce over m here*/
      if (threadRow == 0) {
        scalar_t tmp = 0.0;
        for (int j = 0; j < NWARPS_PER_BLOCK; j++) {
          tmp += buffer_out[j * WARP_SIZE + threadCol];
        }

        gradX[node_index * nchannels + i * WARP_SIZE + feature] = tmp;
      }
    }
  }
}