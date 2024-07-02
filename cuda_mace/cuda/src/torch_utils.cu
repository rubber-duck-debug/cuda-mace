#include "cuda_utils.cuh"
#include "torch_utils.cuh"

#define NEIGHBOUR_NEDGES_PER_BLOCK 512

using namespace torch;

__global__ void calculate_first_occurences_kernel(
    const torch::PackedTensorAccessor64<int32_t, 1, torch::RestrictPtrTraits>
        receiver_list,
    const int32_t *__restrict__ sort_idx, bool use_sort,
    torch::PackedTensorAccessor64<int32_t, 1, torch::RestrictPtrTraits>
        first_occurences)
{
  extern __shared__ char buffer[];
  size_t offset = 0;
  int32_t *smem = reinterpret_cast<int32_t *>(buffer + offset);

  int32_t block_start = blockIdx.x * NEIGHBOUR_NEDGES_PER_BLOCK;

  int32_t nedges = receiver_list.size(0);

  // load all elements of senderlist needed by block into shared memory
  for (int32_t i = threadIdx.x; i < NEIGHBOUR_NEDGES_PER_BLOCK + 1;
       i += blockDim.x)
  {
    int32_t idx = block_start + i;

    if (idx < nedges)
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
  for (int32_t i = 2 * threadIdx.x; i < NEIGHBOUR_NEDGES_PER_BLOCK;
       i += 2 * blockDim.x)
  {
    int32_t idx = block_start + i;

    if (idx + 1 < nedges)
    {
      int32_t loc1 = smem[i];
      int32_t loc2 = smem[i + 1];

      if (loc1 != loc2)
      {
        first_occurences[loc2] = idx + 1;
      }
    }
  }

  // deal with odd boundaries
  for (int32_t i = 2 * threadIdx.x + 1; i < NEIGHBOUR_NEDGES_PER_BLOCK + 1;
       i += 2 * blockDim.x)
  {
    int32_t idx = block_start + i;

    if (idx + 1 < nedges)
    {
      int32_t loc1 = smem[i];
      int32_t loc2 = smem[i + 1];

      if (loc1 != loc2)
      {
        first_occurences[loc2] = idx + 1;
      }
    }
  }

  // deal with 0th element specifically, so we dont need to use torch::zeros
  if (blockIdx.x == 0 && threadIdx.x == 0)
  {
    first_occurences[0] = 0;
  }
}

torch::Tensor calculate_first_occurences_gpu(torch::Tensor receiver_list,
                                             int64_t natoms, int64_t nthreadx)
{
  torch::Tensor first_occurences =
      torch::empty(natoms, torch::TensorOptions()
                               .dtype(receiver_list.dtype())
                               .device(receiver_list.device()));

  int32_t nbx =
      find_integer_divisor(receiver_list.size(0), NEIGHBOUR_NEDGES_PER_BLOCK);

  dim3 block_dim(nbx);

  dim3 grid_dim(nthreadx, 1, 1);

  size_t total_buff_size = 0;

  total_buff_size += (NEIGHBOUR_NEDGES_PER_BLOCK + 1) * sizeof(int32_t);

  calculate_first_occurences_kernel<<<block_dim, grid_dim, total_buff_size>>>(
      receiver_list.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>(),
      nullptr, false,
      first_occurences
          .packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>());

  // cudaDeviceSynchronize();

  return first_occurences;
}

torch::Tensor
calculate_first_occurences_gpu_with_sort(torch::Tensor receiver_list,
                                         int64_t natoms, int64_t nthreadx,
                                         torch::Tensor sort_indices)
{
  torch::Tensor first_occurences =
      torch::empty(natoms, torch::TensorOptions()
                               .dtype(receiver_list.dtype())
                               .device(receiver_list.device()));

  int32_t nbx =
      find_integer_divisor(receiver_list.size(0), NEIGHBOUR_NEDGES_PER_BLOCK);

  dim3 block_dim(nbx);

  dim3 grid_dim(nthreadx, 1, 1);

  size_t total_buff_size = 0;

  total_buff_size += (NEIGHBOUR_NEDGES_PER_BLOCK + 1) * sizeof(int32_t);

  if (sort_indices.defined() && sort_indices.numel() != 0)
  {
    calculate_first_occurences_kernel<<<block_dim, grid_dim, total_buff_size>>>(
        receiver_list.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>(),
        sort_indices.data_ptr<int32_t>(), true,
        first_occurences
            .packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>());
  }

  // cudaDeviceSynchronize();

  return first_occurences;
}