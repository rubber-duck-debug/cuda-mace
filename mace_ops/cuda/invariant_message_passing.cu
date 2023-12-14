#include <torch/script.h>
#include <iostream>
#include <cuda/barrier>

using namespace std;
using namespace torch::indexing;
using namespace torch::autograd;

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

#define FULL_MASK 0xffffffff

template <class T>
__host__ __device__ T *shared_array(std::size_t n_elements, void *&ptr,
                                    std::size_t *space = nullptr) noexcept
{
    const std::uintptr_t inptr = reinterpret_cast<uintptr_t>(ptr);
    const std::uintptr_t end = inptr + n_elements * sizeof(T);
    if (space)
        *space += static_cast<std::size_t>(end - inptr);
    ptr = reinterpret_cast<void *>(end);
    return reinterpret_cast<T *>(inptr);
}

#define WARP_SIZE 32

__host__ __device__ int32_t find_integer_divisor(int32_t x, int32_t bdim)
{
    return (x + bdim - 1) / bdim;
}

#define NWARPS_PER_BLOCK 4

template <typename scalar_t, const int TM, const int TN>
__global__ __launch_bounds__(128) void forward_kernel(
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> X,               // [nedges nchannels]
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> Y,               // [nedges, (L+1)**2]
    const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> radial,          // [nedges, L+1, nchannels]
    const torch::PackedTensorAccessor64<int32_t, 1, torch::RestrictPtrTraits> receiver_list,    // which index we need to sum a particular edge into -> monotonically increasing.
    const torch::PackedTensorAccessor64<int32_t, 1, torch::RestrictPtrTraits> first_occurences, // the indexes in reciever_list which deliniate the set of edges per node.
    torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> output)
{

    extern __shared__ char buffer[];

    void *sptr = buffer;
    size_t space = 0;

    scalar_t *buffer_out = shared_array<scalar_t>(16 * 128, sptr, &space);

    float regM[16] = {0.0};
    float regN[TN] = {0.0};
    float regWeights[4 * TN] = {0.0};
    float result[16 * TN] = {0.0};

    const uint threadCol = threadIdx.x % 32;
    const uint threadRow = threadIdx.x / 32;

    const uint N = X.size(1);
    const uint edge_start = first_occurences[blockIdx.x];
    const uint edge_end = (blockIdx.x == first_occurences.size(0) - 1) ? receiver_list.size(0) : first_occurences[blockIdx.x + 1];
    const uint node_index = receiver_list[edge_start];

    const uint N_start = blockIdx.y * TN * 32;

    for (int tid = threadIdx.x; tid < 16 * 128; tid += blockDim.x)
    {
        buffer_out[tid] = 0.0;
    }

    __syncthreads();

    // check if this node has neighbours
    if (edge_end - edge_start == 0)
    {
        return;
    }

    int niter = find_integer_divisor(edge_end - edge_start, NWARPS_PER_BLOCK);

    for (uint ni = 0; ni < niter; ni++)
    {
        uint edge = edge_start + ni * NWARPS_PER_BLOCK + threadRow;

        if (edge < edge_end)
        {
            for (uint n = 0; n < TN; n++)
            {
                if (N_start + n * 32 + threadCol < N)
                    regN[n] = X[edge][N_start + n * 32 + threadCol];
            }

            // load first into registers
            for (uint m = 0; m < 16; m++)
            {
                regM[m] = Y[edge][m];
            }

            for (int L = 0; L < 4; L++)
            {
                for (uint n = 0; n < TN; n++)
                {
                    if (N_start + n * 32 + threadCol < N)
                        regWeights[L * 4 + n] = radial[edge][L][n * 32 + threadCol];
                }
            }

            // perform outer product in registers
            for (uint m = 0; m < 16; m++)
            {
                int32_t lm_index = sqrt(m);
                for (uint n = 0; n < TN; n++)
                {
                    if (N_start + n * 32 + threadCol < N)
                    {
                        result[m * 4 + n] += regWeights[lm_index * 4 + n] * regM[m] * regN[n];
                    }
                }
            }
        }
    }

    __syncthreads();

    // need to sum over partial sums from each subset of edges iterated by each threadRow
    for (int m = 0; m < 16; m++)
    {
        for (int n = 0; n < TN; n++)
        {
            // buffer_out[threadRow * 16 * 128 + m * 128 + n * 32 + threadCol] = result[m * 4 + n];
            atomicAdd(&buffer_out[m * 128 + n * 32 + threadCol], result[m * 4 + n]);
        }
    }

    __syncthreads();

    for (int m = threadRow; m < 16; m += NWARPS_PER_BLOCK)
    {
        for (int n = 0; n < TN; n++)
        {

            if (N_start + n * 32 + threadCol < N)
                output[node_index][m][N_start + n * 32 + threadCol] = buffer_out[m * 128 + n * 32 + threadCol];
        }
    }
}

template <typename scalar_t, const int TM, const int TN>
__global__ __launch_bounds__(128) void forward_kernel2(
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> X,               // [nedges nchannels]
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> Y,               // [nedges, (L+1)**2]
    const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> radial,          // [nedges, L+1, nchannels]
    const torch::PackedTensorAccessor64<int32_t, 1, torch::RestrictPtrTraits> receiver_list,    // which index we need to sum a particular edge into -> monotonically increasing.
    const torch::PackedTensorAccessor64<int32_t, 1, torch::RestrictPtrTraits> first_occurences, // the indexes in reciever_list which deliniate the set of edges per node.
    torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> output)
{

    extern __shared__ char buffer[];

    void *sptr = buffer;
    size_t space = 0;

    float regM[4] = {0.0};
    float regN[TN] = {0.0};
    float regWeights[4 * TN] = {0.0};
    float result[4 * TN] = {0.0};

    const uint threadCol = threadIdx.x % 32;
    const uint threadRow = threadIdx.x / 32;

    const uint N = X.size(1);
    const uint edge_start = first_occurences[blockIdx.x];
    const uint edge_end = (blockIdx.x == first_occurences.size(0) - 1) ? receiver_list.size(0) : first_occurences[blockIdx.x + 1];
    const uint node_index = receiver_list[edge_start];

    const uint N_start = blockIdx.y * TN * 32;

    __syncthreads();

    // check if this node has neighbours
    if (edge_end - edge_start == 0)
    {
        return;
    }

    for (uint edge = edge_start; edge < edge_end; edge++)
    {
        for (uint n = 0; n < TN; n++)
        {
            if (N_start + n * 32 + threadCol < N)
                regN[n] = X[edge][N_start + n * 32 + threadCol];
        }

        // load first into registers
        for (uint m = 0; m < 4; m++)
        {
            regM[m] = Y[edge][m * 4 + threadRow];
        }

        for (int L = 0; L < 4; L++)
        {
            for (uint n = 0; n < TN; n++)
            {
                if (N_start + n * 32 + threadCol < N)
                    regWeights[L * 4 + n] = radial[edge][L][n * 32 + threadCol];
            }
        }

        // perform outer product in registers
        for (uint n = 0; n < TN; n++)
        {
            if (N_start + n * 32 + threadCol < N)
            {
                for (uint m = 0; m < 4; m++)
                {
                    int32_t lm_index = sqrt(m * 4 + threadRow);

                    result[m * 4 + n] += regWeights[lm_index * 4 + n] * regM[m] * regN[n];
                }
            }
        }
    }

    __syncthreads();

    for (uint n = 0; n < TN; n++)
    {
        if (N_start + n * 32 + threadCol < N)
        {
            for (uint m = 0; m < 4; m++)
            {
                output[node_index][m * 4 + threadRow][N_start + n * 32 + threadCol] = result[m * 4 + n];
            }
        }
    }
}

torch::Tensor forward_gpu(
    torch::Tensor X,
    torch::Tensor Y,
    torch::Tensor radial,
    torch::Tensor receiver_list,
    torch::Tensor first_occurences,
    const int64_t nnodes)
{

    const uint nedges = Y.size(0);
    const int nspherical_harm = Y.size(1);
    const int nfeatures = X.size(1);

    TORCH_CHECK(nfeatures % 32 == 0, "feature dimension must be a multiple of 32");
    TORCH_CHECK(nspherical_harm == 16, "number of edge spherical harmonics must be 16");
    TORCH_CHECK(nfeatures <= 128, "feature dimension cannot be greater than 128");

    torch::Tensor output = torch::empty({nnodes, nspherical_harm, nfeatures},
                                        torch::TensorOptions()
                                            .dtype(X.dtype())
                                            .device(X.device()));

    dim3 gridDim(nnodes, find_integer_divisor(nfeatures, 128));

    dim3 blockDim(128, 1, 1);

    AT_DISPATCH_FLOATING_TYPES(
        X.type(), "forward_gpu", ([&]
                                  {
                                      size_t space = 0;
                                      void *sptr = nullptr;

                                      shared_array<scalar_t>(16 * 128, sptr, &space); // 64 * 128 * 8 = 65kb of shared memory...too large...

                                      forward_kernel<scalar_t, 4, 4><<<gridDim, blockDim, space>>>(
                                          X.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                                          Y.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                                          radial.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                                          receiver_list.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>(),
                                          first_occurences.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>(),
                                          output.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>()); }));

    cudaDeviceSynchronize();

    return output;
}

template <typename scalar_t, const int TM, const int TN>
__global__ void __launch_bounds__(NWARPS_PER_BLOCK * 32) backward_edge_kernel(
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> X,       // [nedges, feat]
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> Y,       // [nedges, m]
    const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> radial,  // [nedges, LMAX, feat]
    const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> grad_in, // [nnodes, m, feat]
    const torch::PackedTensorAccessor64<int32_t, 1, torch::RestrictPtrTraits> receiver_list,
    const torch::PackedTensorAccessor64<int32_t, 1, torch::RestrictPtrTraits> first_occurences,
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> gradX,
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> gradY,
    torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> grad_radial)
{

    extern __shared__ char buffer[];

    void *sptr = buffer;
    size_t space = 0;

    scalar_t *buffer_grad_in = shared_array<scalar_t>(16 * X.size(1), sptr, &space);
    scalar_t *buffer_Y = shared_array<scalar_t>(NWARPS_PER_BLOCK * 16, sptr, &space);
    // scalar_t *buffer_dY = shared_array<scalar_t>(NWARPS_PER_BLOCK * 16, sptr, &space);

    const uint threadCol = threadIdx.x % 32;
    const uint threadRow = threadIdx.x / 32;

    const uint edge_start = first_occurences[blockIdx.x];
    const uint node_index = receiver_list[edge_start];
    const uint edge_end = (blockIdx.x == first_occurences.size(0) - 1) ? receiver_list.size(0) : first_occurences[blockIdx.x + 1];

    const uint N_start = blockIdx.y * TN * 32;

    scalar_t regX[TN] = {0.0};
    scalar_t regW[4 * TN] = {0.0};

    scalar_t regGradX[TN] = {0.0};
    scalar_t regGradY[16] = {0.0};
    scalar_t regGradW[4 * TN] = {0.0};

    if (edge_end - edge_start == 0)
    {
        return;
    }

    for (int n = 0; n < TN; n++)
    {
        for (int m = 0; m < 4; m++)
        {
            if (N_start + n * 32 + threadCol < X.size(1))
                buffer_grad_in[(m * 4 + threadRow) * X.size(1) + n * 32 + threadCol] = grad_in[node_index][m * 4 + threadRow][N_start + n * 32 + threadCol];
        }
    }

    __syncthreads();

    int niter = find_integer_divisor(edge_end - edge_start, NWARPS_PER_BLOCK);

    for (uint ni = 0; ni < niter; ni++)
    {
        uint edge = edge_start + ni * NWARPS_PER_BLOCK + threadRow;

        if (edge < edge_end)
        {
            for (int n = 0; n < TN; n++)
            {
                if (N_start + n * 32 + threadCol < X.size(1))
                    regX[n] = X[edge][N_start + n * 32 + threadCol];

                regGradX[n] = 0.0;
            }

            if (threadCol < 16)
            {
                buffer_Y[threadRow * 16 + threadCol] = Y[edge][threadCol];
            }

            __syncwarp();

            for (int L = 0; L < 4; L++)
            {
                uint mstart = L * L;
                uint mend = (L + 1) * (L + 1);

                for (int n = 0; n < TN; n++)
                {
                    regGradW[L * TN + n] = 0.0;

                    if (N_start + n * 32 + threadCol < X.size(1))
                        regW[L * TN + n] = radial[edge][L][N_start + n * 32 + threadCol];
                }

                for (int m = mstart; m < mend; m++)
                {
                    scalar_t sph = buffer_Y[threadRow * 16 + m]; // Y[edge][m];

                    scalar_t dgradY = 0.0;

                    for (int n = 0; n < TN; n++)
                    {
                        if (N_start + n * 32 + threadCol < X.size(1))
                        {
                            scalar_t gradin = buffer_grad_in[m * X.size(1) + n * 32 + threadCol];
                            scalar_t w = regW[L * TN + n];

                            regGradW[L * TN + n] += sph * regX[n] * gradin;

                            regGradX[n] += sph * w * gradin;

                            dgradY += gradin * w * regX[n];
                        }
                    }

                    for (int offset = 16; offset > 0; offset /= 2)
                    {
                        dgradY += __shfl_down_sync(FULL_MASK, dgradY, offset, 32);
                    }

                    // threadIdx 0 dgradY contains the derivative of the output wrt. Y
                    regGradY[m] = dgradY;
                }
            }
        }

        if (threadCol == 0)
        {
            for (int m = 0; m < 16; m++)
            {
                buffer_Y[threadRow * 16 + m] = regGradY[m];
            }
        }

        __syncthreads();

        if (edge < edge_end)
        {
            if (threadCol < 16)
            {
                gradY[edge][threadCol] = buffer_Y[threadRow * 16 + threadCol];
            }
            for (int n = 0; n < TN; n++)
            {
                if (N_start + n * 32 + threadCol < X.size(1))
                {
                    // write grad_radial
                    for (int L = 0; L < 4; L++)
                    {
                        grad_radial[edge][L][N_start + n * 32 + threadCol] = regGradW[L * TN + n];
                    }

                    gradX[edge][N_start + n * 32 + threadCol] = regGradX[n];
                }
            }
        }
    }
}

std::vector<torch::Tensor> backward_gpu(torch::Tensor X,
                                        torch::Tensor Y,
                                        torch::Tensor radial,
                                        torch::Tensor grad_in,
                                        torch::Tensor receiver_list,
                                        torch::Tensor first_occurences,
                                        const int64_t nnodes)
{
    uint nedges = Y.size(0);
    uint nfeatures = X.size(1);

    TORCH_CHECK(X.requires_grad(), "X must require grad for invariant message passing backwards_kernel to be called.");
    TORCH_CHECK(Y.requires_grad(), "Y must require grad for invariant message passing backwards_kernel to be called.");
    TORCH_CHECK(radial.requires_grad(), "radial must require grad for invariant message passing backwards_kernel to be called.");

    torch::Tensor gradRadial = torch::empty_like(radial,
                                                 torch::TensorOptions()
                                                     .dtype(radial.dtype())
                                                     .device(radial.device()));

    torch::Tensor gradX = torch::empty_like(X,
                                            torch::TensorOptions()
                                                .dtype(X.dtype())
                                                .device(X.device()));

    torch::Tensor gradY = torch::empty_like(Y,
                                            torch::TensorOptions()
                                                .dtype(Y.dtype())
                                                .device(Y.device()));

    AT_DISPATCH_FLOATING_TYPES(
        X.type(), "backward_gpu", ([&]
                                   {
                                       dim3 blockDim(NWARPS_PER_BLOCK * 32, 1, 1);
                                       dim3 gridDim(nnodes, 1);

                                       void *sptr = nullptr;
                                       size_t space = 0;

                                       shared_array<scalar_t>(16 * X.size(1), sptr, &space);
                                       shared_array<scalar_t>(NWARPS_PER_BLOCK * 16, sptr, &space);

                                       if (nfeatures == 96)
                                       {
                                           backward_edge_kernel<scalar_t, 4, 3><<<gridDim, blockDim, space>>>(
                                               X.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                                               Y.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                                               radial.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                                               grad_in.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                                               receiver_list.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>(),
                                               first_occurences.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>(),
                                               gradX.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                                               gradY.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                                               gradRadial.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>());
                                       }
                                       else if (nfeatures == 64)
                                       {
                                           backward_edge_kernel<scalar_t, 4, 2><<<gridDim, blockDim, space>>>(
                                               X.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                                               Y.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                                               radial.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                                               grad_in.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                                               receiver_list.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>(),
                                               first_occurences.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>(),
                                               gradX.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                                               gradY.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                                               gradRadial.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>());
                                       }
                                       else if (nfeatures == 32)
                                       {
                                           backward_edge_kernel<scalar_t, 4, 1><<<gridDim, blockDim, space>>>(
                                               X.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                                               Y.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                                               radial.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                                               grad_in.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                                               receiver_list.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>(),
                                               first_occurences.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>(),
                                               gradX.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                                               gradY.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                                               gradRadial.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>());
                                       }
                                       else
                                       {
                                           backward_edge_kernel<scalar_t, 4, 4><<<gridDim, blockDim, space>>>(
                                               X.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                                               Y.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                                               radial.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                                               grad_in.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                                               receiver_list.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>(),
                                               first_occurences.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>(),
                                               gradX.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                                               gradY.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                                               gradRadial.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>());
                                       }
                                       //}
                                   }));

    cudaDeviceSynchronize();

    return {gradX, gradY, gradRadial};
}

#define NEIGHBOUR_NEDGES_PER_BLOCK 512

/*
This function takes a sorted input sender_list, which maps each edge to a node by index, and finds the positions of first occurences

This is required by the CUDA code so we can send all calculations per-node to a single block.

the function loads NEIGHBOUR_NEDGES_PER_BLOCK + 1 elements into shared memory, and then loops through the buffer twice. Once for even boundaries, once for odd boundaries.
*/

__global__ void calculate_first_occurences_kernel(const torch::PackedTensorAccessor64<int32_t, 1, torch::RestrictPtrTraits> receiver_list,
                                                  const int32_t *__restrict__ sort_idx,
                                                  bool use_sort,
                                                  torch::PackedTensorAccessor64<int32_t, 1, torch::RestrictPtrTraits> first_occurences)
{
    extern __shared__ char buffer[];
    size_t offset = 0;
    int32_t *smem = reinterpret_cast<int32_t *>(buffer + offset);

    int32_t block_start = blockIdx.x * NEIGHBOUR_NEDGES_PER_BLOCK;

    int32_t nedges = receiver_list.size(0);

    // load all elements of senderlist needed by block into shared memory
    for (int32_t i = threadIdx.x; i < NEIGHBOUR_NEDGES_PER_BLOCK + 1; i += blockDim.x)
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
    for (int32_t i = 2 * threadIdx.x; i < NEIGHBOUR_NEDGES_PER_BLOCK; i += 2 * blockDim.x)
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
    for (int32_t i = 2 * threadIdx.x + 1; i < NEIGHBOUR_NEDGES_PER_BLOCK + 1; i += 2 * blockDim.x)
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

torch::Tensor calculate_first_occurences_gpu(torch::Tensor receiver_list, int64_t natoms, int64_t nthreadx)
{
    torch::Tensor first_occurences = torch::empty(natoms,
                                                  torch::TensorOptions()
                                                      .dtype(receiver_list.dtype())
                                                      .device(receiver_list.device()));

    int32_t nbx = find_integer_divisor(receiver_list.size(0), NEIGHBOUR_NEDGES_PER_BLOCK);

    dim3 block_dim(nbx);

    dim3 grid_dim(nthreadx, 1, 1);

    size_t total_buff_size = 0;

    total_buff_size += (NEIGHBOUR_NEDGES_PER_BLOCK + 1) * sizeof(int32_t);

    calculate_first_occurences_kernel<<<block_dim, grid_dim, total_buff_size>>>(
        receiver_list.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>(),
        nullptr,
        false,
        first_occurences.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>());

    cudaDeviceSynchronize();

    return first_occurences;
}

torch::Tensor calculate_first_occurences_gpu_with_sort(torch::Tensor receiver_list, int64_t natoms, int64_t nthreadx, torch::Tensor sort_indices)
{
    torch::Tensor first_occurences = torch::empty(natoms,
                                                  torch::TensorOptions()
                                                      .dtype(receiver_list.dtype())
                                                      .device(receiver_list.device()));

    int32_t nbx = find_integer_divisor(receiver_list.size(0), NEIGHBOUR_NEDGES_PER_BLOCK);

    dim3 block_dim(nbx);

    dim3 grid_dim(nthreadx, 1, 1);

    size_t total_buff_size = 0;

    total_buff_size += (NEIGHBOUR_NEDGES_PER_BLOCK + 1) * sizeof(int32_t);

    if (sort_indices.defined() && sort_indices.numel() != 0)
    {
        calculate_first_occurences_kernel<<<block_dim, grid_dim, total_buff_size>>>(
            receiver_list.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>(),
            sort_indices.data_ptr<int32_t>(),
            true,
            first_occurences.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>());
    }

    cudaDeviceSynchronize();

    return first_occurences;
}

class InvariantMessagePassingTPAutograd : public Function<InvariantMessagePassingTPAutograd>
{
public:
    static torch::Tensor forward(
        AutogradContext *ctx,
        torch::Tensor X,
        torch::Tensor Y,
        torch::Tensor radial,
        torch::Tensor receiver_list,
        const int64_t nnodes)
    {

        torch::Tensor first_occurences = calculate_first_occurences_gpu(receiver_list, nnodes, 64);

        if (X.requires_grad() || Y.requires_grad() || radial.requires_grad())
        {
            ctx->saved_data["nnodes"] = nnodes;
            ctx->save_for_backward({X, Y, radial, receiver_list, first_occurences});
        }

        return forward_gpu(X, Y, radial, receiver_list, first_occurences, nnodes);
    }

    static variable_list backward(AutogradContext *ctx, variable_list grad_outputs)
    {

        auto saved_variables = ctx->get_saved_variables();

        auto X = saved_variables[0];
        auto Y = saved_variables[1];
        auto radial = saved_variables[2];
        auto receiver_list = saved_variables[3];
        auto first_occurences = saved_variables[4];

        int64_t nnodes = ctx->saved_data["nnodes"].toInt();

        auto result = backward_gpu(X, Y, radial, grad_outputs[0].contiguous(), receiver_list, first_occurences, nnodes);

        torch::Tensor undef;

        return {result[0], result[1], result[2], undef, undef};
    }
};

torch::Tensor invariant_message_passing_tensor_product(
    torch::Tensor X,
    torch::Tensor Y,
    torch::Tensor radial,
    torch::Tensor receiver_list,
    const int64_t nnodes)
{
    return InvariantMessagePassingTPAutograd::apply(X, Y, radial, receiver_list, nnodes);
}

TORCH_LIBRARY(invariant_tp, m)
{
    m.def("forward", &invariant_message_passing_tensor_product);
    m.def("calculate_first_occurences", &calculate_first_occurences_gpu);
    m.def("calculate_first_occurences_with_sort", &calculate_first_occurences_gpu_with_sort);
}
