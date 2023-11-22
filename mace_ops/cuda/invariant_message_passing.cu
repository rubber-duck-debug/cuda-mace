#include <torch/script.h>
#include <iostream>

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

template <typename scalar_t, const int TM, const int TN>
__global__ __launch_bounds__(128) void forward2_kernel(
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> X,
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> Y,
    const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> radial,
    const torch::PackedTensorAccessor64<int32_t, 1, torch::RestrictPtrTraits> lm_to_L,
    const torch::PackedTensorAccessor64<int32_t, 1, torch::RestrictPtrTraits> receiver_list,
    const torch::PackedTensorAccessor64<int32_t, 1, torch::RestrictPtrTraits> neighbour_indices,
    torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> output)
{

    extern __shared__ char buffer[];

    void *sptr = buffer;
    size_t space = 0;

    // scalar_t *buffer_X = shared_array<scalar_t>(X.size(1), sptr, &space);
    // scalar_t *buffer_Y = shared_array<scalar_t>(32 * 33, sptr, &space);
    int32_t *buffer_lm_to_L = shared_array<int32_t>(Y.size(1), sptr, &space);

    float regM[TM] = {0.0};
    float regN[TN] = {0.0};
    float regWeights[TM * TN] = {0.0};
    float result[TM * TN] = {0.0};

    const uint threadCol = threadIdx.x % 32;
    const uint threadRow = threadIdx.x / 32;

    const uint edge_start = neighbour_indices[blockIdx.x];
    const uint node_index = receiver_list[edge_start];
    const uint edge_end = (blockIdx.x == neighbour_indices.size(0) - 1) ? X.size(0) : neighbour_indices[blockIdx.x + 1];
    const uint nedges = edge_end - edge_start;

    __syncthreads();

    // check if this node has neighbours
    if (nedges == 0)
    {
        return;
    }

    if (threadRow == 0)
    {
        for (uint i = threadCol; i < Y.size(1); i += 32)
        {
            buffer_lm_to_L[i] = lm_to_L[i];
        }
    }

    for (uint n = 0; n < TN; n++)
    {
        regN[n] = X[edge_start][n * 32 + threadCol];
    }

    __syncthreads();

    for (uint edge = edge_start; edge < edge_end; edge++)
    {
        // compute outer product segment

        // load first into registers
        for (uint m = 0; m < TM; m++)
        {
            regM[m] = Y[edge][m * TM + threadRow];

            int32_t lm_index = buffer_lm_to_L[m * TM + threadRow];
            for (uint n = 0; n < TN; n++)
            {
                regWeights[m * TM + n] = radial[edge][lm_index][n * 32 + threadCol];
            }
        }

        // perform outer product in registers
        for (uint m = 0; m < TM; m++)
        {
            for (uint n = 0; n < TN; n++)
            {
                result[m * TM + n] += regWeights[m * TM + n] * regM[m] * regN[n];
            }
        }
    }

    for (int m = 0; m < TM; m++)
    {
        for (int n = 0; n < TN; n++)
        {
            output[node_index][m * TM + threadRow][n * 32 + threadCol] = result[m * TM + n];
        }
    }
}

torch::Tensor forward_gpu2(
    torch::Tensor X,
    torch::Tensor Y,
    torch::Tensor radial,
    torch::Tensor lm_to_L,
    torch::Tensor receiver_list,
    torch::Tensor neighbour_indices,
    int64_t natoms,
    int64_t nthreadx,
    int64_t nthready,
    int64_t nthreadz)
{

    const int nspherical_harm = Y.size(1);
    const int nfeatures = X.size(1);

    torch::Tensor output = torch::empty({natoms, nspherical_harm, nfeatures},
                                        torch::TensorOptions()
                                            .dtype(X.dtype())
                                            .device(X.device()));

    dim3 gridDim(natoms);

    dim3 blockDim(128, 1, 1);

    AT_DISPATCH_FLOATING_TYPES(
        X.type(), "forward_gpu2", ([&]
                                   {

        size_t shared_size = 0;
        void* sptr = nullptr;

        //shared_array<scalar_t>(nfeatures, sptr, &shared_size);
        //shared_array<scalar_t>(32 * 33, sptr, &shared_size);
        shared_array<int32_t>(nspherical_harm, sptr, &shared_size);

        forward2_kernel<scalar_t,4,3><<<gridDim, blockDim, shared_size>>>(
            X.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
            Y.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
            radial.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
            lm_to_L.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>(),
            receiver_list.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>(),
            neighbour_indices.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>(),
            output.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>()); }));

    cudaDeviceSynchronize();

    return output;
}

template <typename scalar_t>
__global__ void forward_kernel(
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> X,
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> Y,
    const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> radial,
    const torch::PackedTensorAccessor64<int32_t, 1, torch::RestrictPtrTraits> lm_to_L,
    const torch::PackedTensorAccessor64<int32_t, 1, torch::RestrictPtrTraits> receiver_list,
    const torch::PackedTensorAccessor64<int32_t, 1, torch::RestrictPtrTraits> neighbour_indices,
    torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> output)
{

    extern __shared__ char buffer[];

    void *sptr = buffer;
    size_t space = 0;

    scalar_t *buffer_out = shared_array<scalar_t>(blockDim.x * output.size(1), sptr, &space);
    int32_t *buffer_lm_to_L = shared_array<int32_t>(Y.size(0), sptr, &space);

    __shared__ int32_t edge_start;
    __shared__ int32_t edge_end;
    __shared__ int32_t node_index;
    __shared__ int32_t nedges;

    if (threadIdx.x == 0 && threadIdx.y == 0)
    {
        edge_start = neighbour_indices[blockIdx.x];
        node_index = receiver_list[edge_start]; // get the index of the node we need to sum into.

        if (blockIdx.x == neighbour_indices.size(0) - 1) // nnodes -1
        {
            edge_end = Y.size(0); // nedges -1
        }
        else
        {
            edge_end = neighbour_indices[blockIdx.x + 1];
        }

        nedges = edge_end - edge_start;
    }

    if (threadIdx.y == 0)
    {
        for (int i = threadIdx.x; i < Y.size(0); i += blockDim.x)
        {
            buffer_lm_to_L[i] = lm_to_L[i];
        }
    }

    __syncthreads();

    // check if this node has neighbours
    if (nedges == 0)
    {
        return;
    }

    int32_t feat = blockIdx.y * blockDim.x + threadIdx.x;

    for (int i = threadIdx.y; i < output.size(1); i += blockDim.y)
    {
        buffer_out[i * blockDim.x + threadIdx.x] = 0.0;
    }

    __syncthreads();

    for (int i = threadIdx.y; i < Y.size(0); i += blockDim.y)
    {
        scalar_t tmp = 0.0;
        int32_t lm_index = buffer_lm_to_L[i];

        for (int32_t edge = edge_start; edge < edge_end; edge++)
        {
            scalar_t y = Y[i][edge];
            scalar_t x = 0.0;
            scalar_t r = 0.0;

            if (feat < X.size(1))
            {
                x = X[edge][feat];
                r = radial[edge][lm_index][feat];
            }

            tmp += x * y * r;
        }

        buffer_out[i * blockDim.x + threadIdx.x] = tmp;
    }

    __syncthreads();

    for (int i = threadIdx.y; i < output.size(1); i += blockDim.y)
    {
        if (feat < X.size(1))
        {
            output[node_index][i][feat] = buffer_out[i * blockDim.x + threadIdx.x];
        }
    }
}

/*
    torch::Tensor X,      // [nedges, nfeatures]
    torch::Tensor Y,      // [nedges, spherical_harmonics]
    torch::Tensor radial, // [nedges, L_max, nfeatures]
*/

torch::Tensor forward_gpu(
    torch::Tensor X,
    torch::Tensor Y,
    torch::Tensor radial,
    torch::Tensor lm_to_L,
    torch::Tensor receiver_list,
    torch::Tensor neighbour_indices,
    int64_t natoms,
    int64_t nthreadx,
    int64_t nthready,
    int64_t nthreadz)
{

    const int nspherical_harm = Y.size(0);
    const int nfeatures = X.size(1);

    torch::Tensor output = torch::empty({natoms, nspherical_harm, nfeatures},
                                        torch::TensorOptions()
                                            .dtype(X.dtype())
                                            .device(X.device()));

    int32_t nby = find_integer_divisor(nfeatures, nthreadx);

    dim3 block_dim(natoms, nby);

    dim3 grid_dim(nthreadx, nthready, 1);

    AT_DISPATCH_FLOATING_TYPES(
        X.type(), "forward_gpu", ([&]
                                  {

                    size_t shared_size = 0;
                    void* sptr = nullptr;

                    shared_array<scalar_t>(nthreadx * output.size(1), sptr, &shared_size);
                    shared_array<int32_t>(Y.size(0), sptr, &shared_size);

                    forward_kernel<scalar_t><<<block_dim, grid_dim, shared_size>>>(
                        X.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                        Y.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                        radial.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                        lm_to_L.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>(),
                        receiver_list.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>(),
                        neighbour_indices.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>(),
                        output.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>()); }));

    cudaDeviceSynchronize();

    return output;
}

#define DX_FEATS_PER_BLOCK 32
#define DX_NEDGES_PER_BLOCK 64

template <typename scalar_t>
__global__ void backward_dXdRadial_kernel(
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> X,
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> Y,
    const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> radial,
    const torch::PackedTensorAccessor64<int32_t, 1, torch::RestrictPtrTraits> lm_to_L,
    const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> grad_in, // [nnodes, m, feat]
    const torch::PackedTensorAccessor64<int32_t, 1, torch::RestrictPtrTraits> receiver_list,
    const torch::PackedTensorAccessor64<int32_t, 1, torch::RestrictPtrTraits> neighbour_indices,

    const bool requires_grad_X,
    const bool requires_grad_radial,
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> grad_X,
    torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> grad_radial)
{
    extern __shared__ char buffer[];
    void *sptr = buffer;
    size_t space = 0;

    int32_t *buffer_lm_to_L = shared_array<int32_t>(Y.size(0), sptr, &space);

    scalar_t *buffer_grad_radial;

    if (requires_grad_radial)
    {
        buffer_grad_radial = shared_array<scalar_t>(4 * blockDim.x * blockDim.y, sptr, &space);
    }

    if (threadIdx.y == 0)
    {
        for (int lm = threadIdx.x; lm < Y.size(0); lm += blockDim.x)
        {
            buffer_lm_to_L[lm] = lm_to_L[lm];
        }
    }

    __syncthreads();

    int32_t feat = blockIdx.y * blockDim.x + threadIdx.x;

    for (int edge_idx = threadIdx.y; edge_idx < DX_NEDGES_PER_BLOCK; edge_idx += blockDim.y)
    {
        scalar_t grad_x_out = 0.0;
        int edge = blockIdx.x * DX_NEDGES_PER_BLOCK + edge_idx;

        for (int l = 0; l < 4; l++)
        {
            buffer_grad_radial[threadIdx.y * (4 * blockDim.x) + l * blockDim.x + threadIdx.x] = 0.0;
        }

        __syncwarp();

        if (edge < Y.size(1))
        {
            int node_idx = receiver_list[edge];

            scalar_t x = X[edge][feat];

#pragma unroll
            for (int lm = 0; lm < 16; lm++)
            {
                int32_t L_index = buffer_lm_to_L[lm];

                scalar_t y = Y[lm][edge];
                scalar_t r = radial[edge][L_index][feat];
                scalar_t g = grad_in[node_idx][lm][feat];

                if (requires_grad_X)
                {
                    grad_x_out += y * r * g;
                }

                if (requires_grad_radial)
                {
                    buffer_grad_radial[threadIdx.y * (4 * blockDim.x) + L_index * blockDim.x + threadIdx.x] += y * x * g;
                }
            }

            __syncwarp();

            if (requires_grad_X)
            {
                grad_X[edge][feat] = grad_x_out;
            }

            if (requires_grad_radial)
            {

                grad_radial[edge][0][feat] = buffer_grad_radial[threadIdx.y * (4 * blockDim.x) + 0 * blockDim.x + threadIdx.x];
                grad_radial[edge][1][feat] = buffer_grad_radial[threadIdx.y * (4 * blockDim.x) + 1 * blockDim.x + threadIdx.x];
                grad_radial[edge][2][feat] = buffer_grad_radial[threadIdx.y * (4 * blockDim.x) + 2 * blockDim.x + threadIdx.x];
                grad_radial[edge][3][feat] = buffer_grad_radial[threadIdx.y * (4 * blockDim.x) + 3 * blockDim.x + threadIdx.x];
            }
        }
    }
}

template <typename scalar_t>
__global__ void backward_dY_kernel(
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> X,
    const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> radial,
    const torch::PackedTensorAccessor64<int32_t, 1, torch::RestrictPtrTraits> lm_to_L,
    const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> grad_in, // [nnodes, m, feat]
    const torch::PackedTensorAccessor64<int32_t, 1, torch::RestrictPtrTraits> receiver_list,
    const torch::PackedTensorAccessor64<int32_t, 1, torch::RestrictPtrTraits> neighbour_indices,
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> grad_Y)
{

    extern __shared__ char buffer[];
    void *sptr = buffer;
    size_t space = 0;

    int32_t *buffer_lm_to_L = shared_array<int32_t>(grad_in.size(1), sptr, &space);
    // scalar_t *buffer_feats = shared_array<scalar_t>(blockDim.y * blockDim.x, sptr, &space);
    scalar_t *buffer_grad_Y = shared_array<scalar_t>(blockDim.y * 16, sptr, &space);

    if (threadIdx.y == 0)
    {
        for (int lm = threadIdx.x; lm < grad_in.size(1); lm += blockDim.x)
        {
            buffer_lm_to_L[lm] = lm_to_L[lm];
        }
    }

    __syncthreads();

    for (int edge_idx = threadIdx.y; edge_idx < DX_NEDGES_PER_BLOCK; edge_idx += blockDim.y)
    {
        int edge = blockIdx.x * DX_NEDGES_PER_BLOCK + edge_idx;
        int node_idx = receiver_list[edge];

        for (int lm = threadIdx.x; lm < 16; lm += blockDim.x)
        {
            buffer_grad_Y[lm * blockDim.y + threadIdx.y] = 0.0;
        }

        if (edge < X.size(0))
        {
            for (int lm = 0; lm < 16; lm++)
            {
                // buffer_feats[threadIdx.y * blockDim.x + threadIdx.x] = 0.0;

                __syncwarp();

                int L_index = buffer_lm_to_L[lm];

                scalar_t tmp = X[edge][threadIdx.x] * radial[edge][L_index][threadIdx.x] * grad_in[node_idx][lm][threadIdx.x];

                for (int offset = 16; offset > 0; offset /= 2)
                {
                    tmp += __shfl_down_sync(FULL_MASK, tmp, offset);
                }

                if (threadIdx.x % 32 == 0)
                {
                    atomicAdd(&buffer_grad_Y[lm * blockDim.y + threadIdx.y], tmp);
                }
            }

            // TODO this will cause issues - better to balot sync
            __syncthreads();

            for (int lm = threadIdx.x; lm < 16; lm += blockDim.y)
            {
                grad_Y[lm][edge] = buffer_grad_Y[lm * blockDim.y + threadIdx.y];
            }
        }
    }
}

std::vector<torch::Tensor> backward_gpu(torch::Tensor X,
                                        torch::Tensor Y,
                                        torch::Tensor radial,
                                        torch::Tensor lm_to_L,
                                        torch::Tensor grad_in,
                                        torch::Tensor receiver_list,
                                        torch::Tensor neighbour_indices,
                                        int64_t natoms)
{

    torch::Tensor gradX;

    if (X.requires_grad())
    {
        gradX = torch::empty_like(X,
                                  torch::TensorOptions()
                                      .dtype(X.dtype())
                                      .device(X.device()));
    }
    else
    {
        gradX = torch::Tensor();
    }

    torch::Tensor gradY;

    if (Y.requires_grad())
    {
        gradY = torch::empty_like(Y,
                                  torch::TensorOptions()
                                      .dtype(Y.dtype())
                                      .device(Y.device()));
    }
    else
    {
        gradY = torch::Tensor();
    }

    torch::Tensor gradRadial;

    if (radial.requires_grad())
    {
        gradRadial = torch::empty_like(radial,
                                       torch::TensorOptions()
                                           .dtype(radial.dtype())
                                           .device(radial.device()));
    }
    else
    {
        gradRadial = torch::Tensor();
    }

    cudaStream_t stream1, stream2;

    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    AT_DISPATCH_FLOATING_TYPES(
        X.type(), "backward_gpu",
        ([&]
         {
             if (X.requires_grad() || radial.requires_grad())
             {

                 dim3 grid_dim_x(DX_FEATS_PER_BLOCK, 8, 1);
                 int32_t nbx = find_integer_divisor(X.size(1), DX_FEATS_PER_BLOCK);
                 dim3 block_dim_x(find_integer_divisor(X.size(0), DX_NEDGES_PER_BLOCK), nbx);

                 void *sptr = nullptr;
                 size_t space = 0;

                 // shared_array<scalar_t>(DX_NEDGES_PER_BLOCK * Y.size(1), sptr, &space);
                 shared_array<int32_t>(Y.size(0), sptr, &space);

                 if (radial.requires_grad())
                 {
                     shared_array<scalar_t>(4 * 32 * grid_dim_x.y, sptr, &space);
                 }
                 backward_dXdRadial_kernel<scalar_t><<<block_dim_x, grid_dim_x, space, stream1>>>(
                     X.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                     Y.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                     radial.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                     lm_to_L.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>(),
                     grad_in.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                     receiver_list.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>(),
                     neighbour_indices.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>(),
                     X.requires_grad(),
                     radial.requires_grad(),
                     gradX.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                     gradRadial.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>());
             }

             if (Y.requires_grad())
             {

                dim3 grid_dim(X.size(1), 8, 1);
                dim3 block_dim(find_integer_divisor(X.size(0), DX_NEDGES_PER_BLOCK));

                void *sptr = nullptr;
                size_t space = 0;
        
                shared_array<int32_t>(Y.size(0), sptr, &space);
                //shared_array<scalar_t>(grid_dim.y * grid_dim.x, sptr, &space);
                shared_array<scalar_t>(grid_dim.y * 16, sptr, &space);

                backward_dY_kernel<scalar_t><<<block_dim, grid_dim, space, stream2>>>(
                     X.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                     radial.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                     lm_to_L.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>(),
                     grad_in.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                     receiver_list.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>(),
                     neighbour_indices.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>(),
                     gradY.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>()
                     );
             } }));

    cudaDeviceSynchronize();

    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);

    return {gradX, gradY, gradRadial};
}

#define NEIGHBOUR_NEDGES_PER_BLOCK 512

/*
This function takes a sorted input sender_list, which maps each edge to a node by index, and finds the positions of first occurences

This is required by the CUDA code so we can send all calculations per-node to a single block.

the function loads NEIGHBOUR_NEDGES_PER_BLOCK + 1 elements into shared memory, and then loops through the buffer twice. Once for even boundaries, once for odd boundaries.
*/

__global__ void calculate_neighbours_kernel(const torch::PackedTensorAccessor64<int32_t, 1, torch::RestrictPtrTraits> sender_list,
                                            torch::PackedTensorAccessor64<int32_t, 1, torch::RestrictPtrTraits> edge_indices)
{
    extern __shared__ char buffer[];
    size_t offset = 0;
    int32_t *smem = reinterpret_cast<int32_t *>(buffer + offset);

    int32_t block_start = blockIdx.x * NEIGHBOUR_NEDGES_PER_BLOCK;

    int32_t nedges = sender_list.size(0);

    // load all elements of senderlist needed by block into shared memory
    for (int32_t i = threadIdx.x; i < NEIGHBOUR_NEDGES_PER_BLOCK + 1; i += blockDim.x)
    {
        int32_t idx = block_start + i;

        if (idx < nedges)
        {
            smem[i] = sender_list[idx];
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
                edge_indices[loc2] = idx + 1;
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
                edge_indices[loc2] = idx + 1;
            }
        }
    }

    // deal with 0th element specifically, so we dont need to use torch::zeros
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        edge_indices[0] = 0;
    }
}

torch::Tensor calculate_neighbours_gpu(torch::Tensor sender_list, int64_t natoms, int64_t nthreadx)
{
    torch::Tensor output_indices = torch::empty(natoms,
                                                torch::TensorOptions()
                                                    .dtype(sender_list.dtype())
                                                    .device(sender_list.device()));

    int32_t nbx = find_integer_divisor(sender_list.size(0), NEIGHBOUR_NEDGES_PER_BLOCK);

    dim3 block_dim(nbx);

    dim3 grid_dim(nthreadx, 1, 1);

    size_t total_buff_size = 0;

    total_buff_size += (NEIGHBOUR_NEDGES_PER_BLOCK + 1) * sizeof(int32_t);

    calculate_neighbours_kernel<<<block_dim, grid_dim, total_buff_size>>>(

        sender_list.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>(),
        output_indices.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>());

    cudaDeviceSynchronize();

    return output_indices;
}

class InvariantMessagePassingTPAutograd : public Function<InvariantMessagePassingTPAutograd>
{
public:
    static torch::Tensor forward(
        AutogradContext *ctx,
        torch::Tensor X,
        torch::Tensor Y,
        torch::Tensor radial,
        torch::Tensor lm_to_L,
        torch::Tensor receiver_list,
        int64_t natoms)
    {
        torch::Tensor neighbours = calculate_neighbours_gpu(receiver_list, natoms, 64);

        if (X.requires_grad() || Y.requires_grad())
        {
            ctx->saved_data["natoms"] = natoms;

            ctx->save_for_backward({X, Y, radial, lm_to_L, receiver_list, neighbours});
        }

        torch::Tensor result = forward_gpu(X, Y, radial, lm_to_L, receiver_list, neighbours, natoms, 32, 4, 1);

        return result;
    }

    static variable_list backward(AutogradContext *ctx, variable_list grad_outputs)
    {
        auto saved_variables = ctx->get_saved_variables();

        auto X = saved_variables[0];
        auto Y = saved_variables[1];
        auto radial = saved_variables[2];
        auto lm_to_L = saved_variables[3];
        auto receiver_list = saved_variables[4];
        auto neighbours = saved_variables[5];

        int64_t natoms = ctx->saved_data["natoms"].toInt();

        auto result = backward_gpu(X, Y, radial, lm_to_L, grad_outputs[0], receiver_list, neighbours, natoms);

        torch::Tensor undef;

        return {result[0], result[1], result[2], undef, undef, undef};
    }
};

torch::Tensor invariant_message_passing_tensor_product(
    torch::Tensor X,
    torch::Tensor Y,
    torch::Tensor radial,
    torch::Tensor lm_to_L,
    torch::Tensor sender_list,
    int64_t natoms)
{
    return InvariantMessagePassingTPAutograd::apply(X, Y, radial, lm_to_L, sender_list, natoms);
}

TORCH_LIBRARY(invariant_tp, m)
{
    m.def("forward", &invariant_message_passing_tensor_product);
    m.def("calculate_neighbours", &calculate_neighbours_gpu);
    m.def("forward_test", &forward_gpu);
    m.def("forward_test2", &forward_gpu2);
    m.def("backward_test", &backward_gpu);
}
