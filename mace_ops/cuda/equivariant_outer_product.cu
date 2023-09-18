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

__host__ __device__ int32_t find_integer_divisor(int32_t x, int32_t bdim)
{
    return (x + bdim - 1) / bdim;
}

#define WARP_SIZE 32
#define NEDGE_PER_BLOCK 8

template <typename scalar_t>
__global__ void edge_equivariant_outer_product_kernel(const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> X,
                                                      const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> Y,
                                                      const torch::PackedTensorAccessor64<int32_t, 1, torch::RestrictPtrTraits> mu_1,
                                                      const torch::PackedTensorAccessor64<int32_t, 1, torch::RestrictPtrTraits> mu_2,
                                                      const torch::PackedTensorAccessor64<int32_t, 1, torch::RestrictPtrTraits> mu_3,
                                                      const torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> cg_coefficients,
                                                      torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> output)
{

    extern __shared__ char buffer[];

    void *sptr = buffer;
    size_t space = 0;

    scalar_t *buffer_cg_coefficients = shared_array<scalar_t>(cg_coefficients.size(0), sptr, &space);

    int32_t *buffer_mu1 = shared_array<int32_t>(mu_1.size(0), sptr, &space);
    int32_t *buffer_mu2 = shared_array<int32_t>(mu_2.size(0), sptr, &space);
    int32_t *buffer_mu3 = shared_array<int32_t>(mu_3.size(0), sptr, &space);

    int edge_start = blockIdx.x * NEDGE_PER_BLOCK;
    int feat_start = blockIdx.y * blockDim.x;

    int feat = feat_start + threadIdx.x;

    bool valid = feat < X.size(2);

    for (int id = threadIdx.y * blockDim.x + threadIdx.x; id < mu_1.size(0); id += blockDim.x * blockDim.y)
    {
        buffer_mu1[id] = mu_1[id];
        buffer_mu2[id] = mu_2[id];
        buffer_mu3[id] = mu_3[id];

        buffer_cg_coefficients[id] = cg_coefficients[id];
    }

    __syncthreads();

    for (int edge_id = threadIdx.y; edge_id < NEDGE_PER_BLOCK; edge_id += blockDim.y)
    {

        scalar_t tmp_output = 0.0;
        int prev_idx = 0;
        int x_idx = 0;
        int y_idx = 0;
        int out_idx = 0;

        for (int instruction = 0; instruction < mu_1.size(0); instruction++)
        {

            x_idx = buffer_mu1[instruction];
            y_idx = buffer_mu2[instruction];
            out_idx = buffer_mu3[instruction];
            scalar_t cg_coeff = buffer_cg_coefficients[instruction];

            scalar_t x = 0.0;
            scalar_t y = 0.0;

            if (valid)
            {
                x = X[x_idx][edge_start + edge_id][feat];
                y = Y[y_idx][edge_start + edge_id][feat];
            }

            if (prev_idx != out_idx)
            {
                if (valid)
                {
                    output[prev_idx][edge_start + edge_id][feat] = tmp_output;
                }

                prev_idx = out_idx;
                tmp_output = 0.0;
            }
            tmp_output += cg_coeff * x * y;
        }

        output[out_idx][edge_start + edge_id][feat] = tmp_output;
    }
}

torch::Tensor edge_equivariant_outer_product_forward_gpu(torch::Tensor X,
                                                         torch::Tensor Y,
                                                         torch::Tensor mu_1,
                                                         torch::Tensor mu_2,
                                                         torch::Tensor mu_3,
                                                         torch::Tensor cg_coefficients,
                                                         int64_t output_nl_size,
                                                         int64_t nthreadx,
                                                         int64_t nthready,
                                                         int64_t nthreadz)
{

    torch::Tensor output = torch::empty({output_nl_size, X.size(1), X.size(2)},
                                        torch::TensorOptions()
                                            .dtype(X.dtype())
                                            .device(X.device()));

    int32_t nbx = find_integer_divisor(X.size(1), NEDGE_PER_BLOCK);
    int32_t nby = find_integer_divisor(X.size(2), nthreadx);

    dim3 block_dim(nbx, nby);

    dim3 grid_dim(nthreadx, nthready, 1);

    AT_DISPATCH_FLOATING_TYPES(
        X.type(), "edge_equivariant_outer_product_forward_gpu", ([&]
                                                                 {
                    size_t shared_size = 0;
                    void* sptr = nullptr;
                    
                        shared_array<scalar_t>(cg_coefficients.size(0), sptr, &shared_size);

                        shared_array<int32_t>(mu_1.size(0), sptr, &shared_size);
                        shared_array<int32_t>(mu_2.size(0), sptr, &shared_size);
                        shared_array<int32_t>(mu_3.size(0), sptr, &shared_size);

                    edge_equivariant_outer_product_kernel<scalar_t><<<block_dim, grid_dim, shared_size>>>(
                        X.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                        Y.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                        mu_1.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>(),
                        mu_2.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>(),
                        mu_3.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>(),
                        cg_coefficients.packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>(),
                        output.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>()); }));

    cudaDeviceSynchronize();

    return output;
}

template <typename scalar_t>
__global__ void equivariant_outer_product_forward_kernel(const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> X,
                                                         const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> Y,
                                                         const torch::PackedTensorAccessor64<int32_t, 1, torch::RestrictPtrTraits> receiver_list,
                                                         const torch::PackedTensorAccessor64<int32_t, 1, torch::RestrictPtrTraits> neighbour_indices,
                                                         const torch::PackedTensorAccessor64<int32_t, 1, torch::RestrictPtrTraits> mu_1,
                                                         const torch::PackedTensorAccessor64<int32_t, 1, torch::RestrictPtrTraits> mu_2,
                                                         const torch::PackedTensorAccessor64<int32_t, 1, torch::RestrictPtrTraits> mu_3,
                                                         const torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> cg_coefficients,
                                                         const torch::PackedTensorAccessor64<int32_t, 1, torch::RestrictPtrTraits> warp_indices,
                                                         const torch::PackedTensorAccessor64<int32_t, 1, torch::RestrictPtrTraits> warp_nwork,
                                                         const torch::PackedTensorAccessor64<int32_t, 1, torch::RestrictPtrTraits> output_ordering,
                                                         torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> output)
{
    extern __shared__ char buffer[];

    void *sptr = buffer;
    size_t space = 0;

    /* SHARED BUFFERS */
    // scalar_t *buffer_x = shared_array<scalar_t>(blockDim.x * X.size(1), sptr, &space);
    scalar_t *buffer_out = shared_array<scalar_t>(blockDim.x * output.size(1), sptr, &space);
    // scalar_t *buffer_y = shared_array<scalar_t>(Y.size(1), sptr, &space);
    scalar_t *buffer_cg_coefficients = shared_array<scalar_t>(mu_1.size(0), sptr, &space);

    int32_t *buffer_mu_1 = shared_array<int32_t>(mu_1.size(0), sptr, &space);
    int32_t *buffer_mu_2 = shared_array<int32_t>(mu_1.size(0), sptr, &space);
    int32_t *buffer_mu_3 = shared_array<int32_t>(mu_1.size(0), sptr, &space);

    int32_t *buffer_index_start = shared_array<int32_t>(warp_nwork.size(0), sptr, &space);
    int32_t *buffer_nwork = shared_array<int32_t>(warp_nwork.size(0), sptr, &space);

    int32_t *buffer_output_ordering = shared_array<int32_t>(output.size(1), sptr, &space);
    /* END SHARED BUFFERS */

    int32_t edge_start = neighbour_indices[blockIdx.x];
    int32_t edge_end = 0;

    int32_t node_index = receiver_list[edge_start]; // get the idnex of the node we need to sum into.

    if (blockIdx.x == neighbour_indices.size(0) - 1) // nnodes -1
    {
        edge_end = Y.size(0); // nedges -1
    }
    else
    {
        edge_end = neighbour_indices[blockIdx.x + 1];
    }

    int32_t nedges = edge_end - edge_start;

    // check if this node has neighbours
    if (nedges == 0)
    {
        return;
    }

    int32_t feat = blockIdx.y * blockDim.x + threadIdx.x;

    bool valid = feat < X.size(2);

    // load shared memory buffers...
    if (threadIdx.y == 0)
    {
        for (int32_t i = threadIdx.x; i < mu_1.size(0); i += blockDim.x)
        {
            buffer_mu_1[i] = mu_1[i];
            buffer_mu_2[i] = mu_2[i];
            buffer_mu_3[i] = mu_3[i];
            buffer_cg_coefficients[i] = cg_coefficients[i];
        }

        for (int32_t i = threadIdx.x; i < warp_indices.size(0); i += blockDim.x)
        {
            buffer_index_start[i] = warp_indices[i];
            buffer_nwork[i] = warp_nwork[i];
        }

        for (int32_t i = threadIdx.x; i < output.size(1); i += blockDim.x)
        {
            buffer_output_ordering[i] = output_ordering[i];
        }
    }

    __syncthreads();

    // zero out shared memory
    for (int32_t i = threadIdx.y; i < output.size(1); i += blockDim.y)
    {
        buffer_out[i * blockDim.x + threadIdx.x] = 0.0;
    }

    __syncthreads();

    for (int32_t edge_idx = threadIdx.z; edge_idx < nedges; edge_idx += blockDim.z)
    {
        int edge = edge_start + edge_idx;

        for (int32_t instruction = threadIdx.y; instruction < warp_indices.size(0); instruction += blockDim.y)
        {
            int32_t index_start = buffer_index_start[instruction];
            int32_t nwork = buffer_nwork[instruction];

            for (int32_t j = 0; j < nwork; j++)
            {
                int32_t instruction_idx = index_start + j;

                int32_t x_idx = buffer_mu_1[instruction_idx];
                int32_t y_idx = buffer_mu_2[instruction_idx];
                int32_t out_idx = buffer_mu_3[instruction_idx];

                scalar_t x = 0.0;

                if (valid)
                    x = X[edge][x_idx][feat];

                scalar_t y = Y[edge][y_idx];

                scalar_t cg_coeff = buffer_cg_coefficients[instruction_idx];

                buffer_out[out_idx * blockDim.x + threadIdx.x] += cg_coeff * x * y;
            }
        }
    }

    __syncthreads();

    for (int i = threadIdx.y; i < output.size(1); i += blockDim.y)
    {
        if (valid)
        {
            int order = buffer_output_ordering[i];
            // output[node_index][order][feat] = buffer_out[i * blockDim.x + threadIdx.x];
            output[node_index][i][feat] = buffer_out[order * blockDim.x + threadIdx.x];
        }
    }
}

/*

template <typename scalar_t>
__global__ void equivariant_outer_product_forward2_kernel(const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> X,
                                                          const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> Y,
                                                          const torch::PackedTensorAccessor64<int32_t, 1, torch::RestrictPtrTraits> receiver_list,
                                                          const torch::PackedTensorAccessor64<int32_t, 1, torch::RestrictPtrTraits> neighbour_indices,
                                                          const torch::PackedTensorAccessor64<int32_t, 1, torch::RestrictPtrTraits> mu_1,
                                                          const torch::PackedTensorAccessor64<int32_t, 1, torch::RestrictPtrTraits> mu_2,
                                                          const torch::PackedTensorAccessor64<int32_t, 1, torch::RestrictPtrTraits> mu_3,
                                                          const torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> cg_coefficients,
                                                          const torch::PackedTensorAccessor64<int32_t, 1, torch::RestrictPtrTraits> warp_indices,
                                                          const torch::PackedTensorAccessor64<int32_t, 1, torch::RestrictPtrTraits> warp_nwork,
                                                          const torch::PackedTensorAccessor64<int32_t, 1, torch::RestrictPtrTraits> output_ordering,
                                                          torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> output)
{
    extern __shared__ char buffer[];

    void *sptr = buffer;
    size_t space = 0;


    // scalar_t *buffer_x = shared_array<scalar_t>(blockDim.x * X.size(1), sptr, &space);
    scalar_t *buffer_out = shared_array<scalar_t>(blockDim.x * output.size(1), sptr, &space);
    // scalar_t *buffer_y = shared_array<scalar_t>(Y.size(1), sptr, &space);
    scalar_t *buffer_cg_coefficients = shared_array<scalar_t>(mu_1.size(0), sptr, &space);

    int32_t *buffer_mu_1 = shared_array<int32_t>(mu_1.size(0), sptr, &space);
    int32_t *buffer_mu_2 = shared_array<int32_t>(mu_1.size(0), sptr, &space);
    int32_t *buffer_mu_3 = shared_array<int32_t>(mu_1.size(0), sptr, &space);

    int32_t *buffer_index_start = shared_array<int32_t>(warp_nwork.size(0), sptr, &space);
    int32_t *buffer_nwork = shared_array<int32_t>(warp_nwork.size(0), sptr, &space);

    int32_t *buffer_output_ordering = shared_array<int32_t>(output.size(1), sptr, &space);


    int32_t edge_start = neighbour_indices[blockIdx.x];
    int32_t edge_end = 0;

    int32_t node_index = receiver_list[edge_start]; // get the idnex of the node we need to sum into.

    if (blockIdx.x == neighbour_indices.size(0) - 1) // nnodes -1
    {
        edge_end = Y.size(0); // nedges -1
    }
    else
    {
        edge_end = neighbour_indices[blockIdx.x + 1];
    }

    int32_t nedges = edge_end - edge_start;

    // check if this node has neighbours
    if (nedges == 0)
    {
        return;
    }

    int32_t feat = blockIdx.y * blockDim.x + threadIdx.x;

    bool valid = feat < X.size(2);

    // load shared memory buffers...
    if (threadIdx.y == 0)
    {
        for (int32_t i = threadIdx.x; i < mu_1.size(0); i += blockDim.x)
        {
            buffer_mu_1[i] = mu_1[i];
            buffer_mu_2[i] = mu_2[i];
            buffer_mu_3[i] = mu_3[i];
            buffer_cg_coefficients[i] = cg_coefficients[i];
        }

        for (int32_t i = threadIdx.x; i < warp_indices.size(0); i += blockDim.x)
        {
            buffer_index_start[i] = warp_indices[i];
            buffer_nwork[i] = warp_nwork[i];
        }

        for (int32_t i = threadIdx.x; i < output.size(1); i += blockDim.x)
        {
            buffer_output_ordering[i] = output_ordering[i];
        }
    }

    __syncthreads();

    // zero out shared memory
    for (int32_t i = threadIdx.y; i < output.size(1); i += blockDim.y)
    {
        buffer_out[i * blockDim.x + threadIdx.x] = 0.0;
    }

    __syncthreads();

    for (int32_t instruction = 0; instruction < warp_indices.size(0); instruction++)
    {
        int32_t index_start = buffer_index_start[instruction];
        int32_t nwork = buffer_nwork[instruction];

        for (int32_t j = 0; j < nwork; j++)
        {
            int32_t instruction_idx = index_start + j;

            int32_t x_idx = buffer_mu_1[instruction_idx];
            int32_t y_idx = buffer_mu_2[instruction_idx];
            int32_t out_idx = buffer_mu_3[instruction_idx];

            scalar_t cg_coeff = buffer_cg_coefficients[instruction_idx];

            for (int i = 0; i < n; i++)
            {
                int edge = i * blockDim.x + threadIdx.x;

                scalar_t y = 0.0;
                scalar_t x = 0.0;

                if (edge < nedges)
                {
                    x =  X[edge_start + edge][x_idx][feat];
                }
            }
            for (int32_t edge_idx = threadIdx.x; edge_idx < nedges; edge_idx += blockDim.x)
            {
                // int edge = edge_start + edge_idx;

                scalar_t x = 0.0;

                if (valid)
                    x = X[edge][x_idx][feat];

                scalar_t y = Y[edge][y_idx];

                atomicAdd(&buffer_out[out_idx * blockDim.x + threadIdx.x], cg_coeff * x * y);
            }
        }
    }

    __syncthreads();

    for (int i = threadIdx.y; i < output.size(1); i += blockDim.y)
    {
        if (valid)
        {
            int order = buffer_output_ordering[i];
            // output[node_index][order][feat] = buffer_out[i * blockDim.x + threadIdx.x];
            output[node_index][i][feat] = buffer_out[order * blockDim.x + threadIdx.x];
        }
    }
} */

torch::Tensor equivariant_outer_product_forward_gpu(torch::Tensor X,
                                                    torch::Tensor Y,
                                                    torch::Tensor receiver_list,
                                                    torch::Tensor neighbour_indices,
                                                    torch::Tensor mu_1,
                                                    torch::Tensor mu_2,
                                                    torch::Tensor mu_3,
                                                    torch::Tensor cg_coefficients,
                                                    torch::Tensor warp_indices,
                                                    torch::Tensor warp_nwork,
                                                    int64_t output_nl_size,
                                                    int64_t natoms,
                                                    torch::Tensor output_ordering,
                                                    int64_t nthreadx,
                                                    int64_t nthready,
                                                    int64_t nthreadz)
{

    torch::Tensor output = torch::empty({natoms, output_nl_size, X.size(2)},
                                        torch::TensorOptions()
                                            .dtype(X.dtype())
                                            .device(X.device()));

    int32_t nby = find_integer_divisor(X.size(2), nthreadx);

    dim3 block_dim(natoms, nby);

    dim3 grid_dim(nthreadx, nthready, nthreadz);

    AT_DISPATCH_FLOATING_TYPES(
        X.type(), "equivariant_outer_product_forward_gpu", ([&]
                                                            {
                    size_t shared_size = 0;
                    void* sptr = nullptr;
                    
                    shared_array<scalar_t>(nthreadx * output.size(1), sptr, &shared_size);
                    shared_array<scalar_t>(mu_1.size(0), sptr, &shared_size);

                    shared_array<int32_t>(mu_1.size(0), sptr, &shared_size);
                    shared_array<int32_t>(mu_1.size(0), sptr, &shared_size);
                    shared_array<int32_t>(mu_1.size(0), sptr, &shared_size);

                    shared_array<int32_t>(warp_nwork.size(0), sptr, &shared_size);
                    shared_array<int32_t>(warp_nwork.size(0), sptr, &shared_size);
                    shared_array<int32_t>(output.size(1), sptr, &shared_size);

                    equivariant_outer_product_forward_kernel<scalar_t><<<block_dim, grid_dim, shared_size>>>(
                        X.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                        Y.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                        receiver_list.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>(),
                        neighbour_indices.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>(),
                        mu_1.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>(),
                        mu_2.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>(),
                        mu_3.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>(),
                        cg_coefficients.packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>(),
                        warp_indices.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>(),
                        warp_nwork.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>(),
                        output_ordering.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>(),
                        output.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>()); }));

    cudaDeviceSynchronize();

    return output;
}

template <typename scalar_t>
__global__ void weighted_equivariant_outer_product_forward_kernel(const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> X,
                                                                  const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> Y,
                                                                  const torch::PackedTensorAccessor64<int32_t, 1, torch::RestrictPtrTraits> receiver_list,
                                                                  const torch::PackedTensorAccessor64<int32_t, 1, torch::RestrictPtrTraits> neighbour_indices,
                                                                  const torch::PackedTensorAccessor64<int32_t, 1, torch::RestrictPtrTraits> mu_1,
                                                                  const torch::PackedTensorAccessor64<int32_t, 1, torch::RestrictPtrTraits> mu_2,
                                                                  const torch::PackedTensorAccessor64<int32_t, 1, torch::RestrictPtrTraits> mu_3,
                                                                  const torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> cg_coefficients,
                                                                  const torch::PackedTensorAccessor64<int32_t, 1, torch::RestrictPtrTraits> warp_indices,
                                                                  const torch::PackedTensorAccessor64<int32_t, 1, torch::RestrictPtrTraits> warp_nwork,
                                                                  const torch::PackedTensorAccessor64<int32_t, 1, torch::RestrictPtrTraits> weight_indices,
                                                                  const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> weights,
                                                                  torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> output)
{
    extern __shared__ char buffer[];

    void *sptr = buffer;
    size_t space = 0;

    /* SHARED BUFFERS */
    // scalar_t *buffer_x = shared_array<scalar_t>(blockDim.x * X.size(1), sptr, &space);
    scalar_t *buffer_out = shared_array<scalar_t>(blockDim.x * output.size(1), sptr, &space);
    // scalar_t *buffer_y = shared_array<scalar_t>(Y.size(1), sptr, &space);
    scalar_t *buffer_cg_coefficients = shared_array<scalar_t>(mu_1.size(0), sptr, &space);

    int32_t *buffer_mu_1 = shared_array<int32_t>(mu_1.size(0), sptr, &space);
    int32_t *buffer_mu_2 = shared_array<int32_t>(mu_1.size(0), sptr, &space);
    int32_t *buffer_mu_3 = shared_array<int32_t>(mu_1.size(0), sptr, &space);

    int32_t *buffer_index_start = shared_array<int32_t>(warp_nwork.size(0), sptr, &space);
    int32_t *buffer_nwork = shared_array<int32_t>(warp_nwork.size(0), sptr, &space);
    int32_t *buffer_weight_index = shared_array<int32_t>(weight_indices.size(0), sptr, &space);
    /* END SHARED BUFFERS */

    int32_t edge_start = neighbour_indices[blockIdx.x];
    int32_t edge_end = 0;

    int32_t node_index = receiver_list[edge_start]; // get the idnex of the node we need to sum into.

    if (blockIdx.x == neighbour_indices.size(0) - 1) // nnodes -1
    {
        edge_end = Y.size(0); // nedges -1
    }
    else
    {
        edge_end = neighbour_indices[blockIdx.x + 1];
    }

    int32_t nedges = edge_end - edge_start;

    // check if this node has neighbours
    if (nedges == 0)
    {
        return;
    }

    int32_t feat = blockIdx.y * blockDim.x + threadIdx.x;

    bool valid = feat < X.size(2);

    // load shared memory buffers...
    if (threadIdx.y == 0)
    {
        for (int32_t i = threadIdx.x; i < mu_1.size(0); i += blockDim.x)
        {
            buffer_mu_1[i] = mu_1[i];
            buffer_mu_2[i] = mu_2[i];
            buffer_mu_3[i] = mu_3[i];
            buffer_cg_coefficients[i] = cg_coefficients[i];
            buffer_weight_index[i] = weight_indices[i];
        }

        for (int32_t i = threadIdx.x; i < warp_indices.size(0); i += blockDim.x)
        {
            buffer_index_start[i] = warp_indices[i];
            buffer_nwork[i] = warp_nwork[i];
        }
    }

    __syncthreads();

    // zero out shared memory
    for (int32_t i = threadIdx.y; i < output.size(1); i += blockDim.y)
    {
        buffer_out[i * blockDim.x + threadIdx.x] = 0.0;
    }

    __syncthreads();

    for (int32_t edge_idx = 0; edge_idx < nedges; edge_idx++)
    {
        int edge = edge_start + edge_idx;

        for (int32_t instruction = threadIdx.y; instruction < warp_indices.size(0); instruction += blockDim.y)
        {
            int32_t index_start = buffer_index_start[instruction];
            int32_t nwork = buffer_nwork[instruction];

            for (int32_t j = 0; j < nwork; j++)
            {
                int32_t instruction_idx = index_start + j;

                int32_t x_idx = buffer_mu_1[instruction_idx];
                int32_t y_idx = buffer_mu_2[instruction_idx];
                int32_t out_idx = buffer_mu_3[instruction_idx];
                int32_t weight_index = buffer_weight_index[instruction_idx];

                scalar_t x = 0.0;
                scalar_t weight = 0.0;

                if (valid)
                {
                    x = X[edge][x_idx][feat];
                    weight = weights[edge][weight_index * X.size(2) + feat];
                }

                scalar_t y = Y[edge][y_idx];

                scalar_t cg_coeff = buffer_cg_coefficients[instruction_idx];

                atomicAdd(&buffer_out[out_idx * blockDim.x + threadIdx.x], cg_coeff * weight * x * y);
            }
        }
    }

    __syncthreads();

    for (int i = threadIdx.y; i < output.size(1); i += blockDim.y)
    {
        if (valid)
        {
            output[node_index][i][feat] = buffer_out[i * blockDim.x + threadIdx.x];
        }
    }
}

torch::Tensor weighted_equivariant_outer_product_forward_gpu(torch::Tensor X,
                                                             torch::Tensor Y,
                                                             torch::Tensor receiver_list,
                                                             torch::Tensor neighbour_indices,
                                                             torch::Tensor mu_1,
                                                             torch::Tensor mu_2,
                                                             torch::Tensor mu_3,
                                                             torch::Tensor cg_coefficients,
                                                             torch::Tensor warp_indices,
                                                             torch::Tensor warp_nwork,
                                                             torch::Tensor weight_indices,
                                                             torch::Tensor weights,
                                                             int64_t output_nl_size,
                                                             int64_t natoms,
                                                             int64_t nthreadx,
                                                             int64_t nthready,
                                                             int64_t nthreadz)
{

    torch::Tensor output = torch::empty({natoms, output_nl_size, X.size(2)},
                                        torch::TensorOptions()
                                            .dtype(X.dtype())
                                            .device(X.device()));

    int32_t nby = find_integer_divisor(X.size(2), nthreadx);

    dim3 block_dim(natoms, nby);

    dim3 grid_dim(nthreadx, nthready, 1);

    AT_DISPATCH_FLOATING_TYPES(
        X.type(), "weighted_equivariant_outer_product_forward_gpu", ([&]
                                                                     {
                    size_t shared_size = 0;
                    void* sptr = nullptr;
                    
                    shared_array<scalar_t>(nthreadx * output.size(1), sptr, &shared_size);
                    shared_array<scalar_t>(mu_1.size(0), sptr, &shared_size);

                    shared_array<int32_t>(mu_1.size(0), sptr, &shared_size);
                    shared_array<int32_t>(mu_1.size(0), sptr, &shared_size);
                    shared_array<int32_t>(mu_1.size(0), sptr, &shared_size);

                    shared_array<int32_t>(warp_nwork.size(0), sptr, &shared_size);
                    shared_array<int32_t>(warp_nwork.size(0), sptr, &shared_size);
                    shared_array<int32_t>(weight_indices.size(0), sptr, &shared_size);

                    weighted_equivariant_outer_product_forward_kernel<scalar_t><<<block_dim, grid_dim, shared_size>>>(
                        X.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                        Y.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                        receiver_list.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>(),
                        neighbour_indices.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>(),
                        mu_1.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>(),
                        mu_2.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>(),
                        mu_3.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>(),
                        cg_coefficients.packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>(),
                        warp_indices.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>(),
                        warp_nwork.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>(),
                        weight_indices.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>(),
                        weights.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                        output.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>()); }));

    cudaDeviceSynchronize();

    return output;
}

#define NEIGHBOUR_NEDGES_PER_BLOCK 512

/*
This function takes a sorted input sender_list, which maps each edge to a node by index, and finds the positions of first occurences

This is required by the CUDA code so we can send all calculations per-node to a single block.

the function loads NEIGHBOUR_NEDGES_PER_BLOCK + 1 elements into shared memory, and then loops through the buffer twice. Once for even boundaries, once for odd boundaries.
*/

__global__ void calculate_neighbours_kernel(const torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits> sender_list,
                                            torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits> edge_indices)
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

        sender_list.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(),
        output_indices.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>());

    cudaDeviceSynchronize();

    return output_indices;
}

TORCH_LIBRARY(mace_ops_equivariant_tp, m)
{
    m.def("edge_equivariant_outer_product_forward", &edge_equivariant_outer_product_forward_gpu);
    m.def("equivariant_outer_product_forward", &equivariant_outer_product_forward_gpu);
    m.def("weighted_equivariant_outer_product_forward", &weighted_equivariant_outer_product_forward_gpu);
    m.def("calculate_neighbours", &calculate_neighbours_gpu);
}
