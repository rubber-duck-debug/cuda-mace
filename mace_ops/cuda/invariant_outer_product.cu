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

__host__ __device__ int32_t find_integer_divisor(int32_t x, int32_t bdim)
{
    return (x + bdim - 1) / bdim;
}

template <typename scalar_t>
__global__ void forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> X,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> Y,
    const torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits> receiver_list,
    const torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits> neighbour_indices,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> output)
{
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

    // check if this node has neighbours
    if (edge_end - edge_start == 0)
    {
        return;
    }

    int32_t feat_start = blockIdx.y * blockDim.x;

    bool valid = feat_start + threadIdx.x < X.size(1);

    __syncthreads();

    for (int32_t m = threadIdx.y; m < Y.size(1); m += blockDim.y)
    {
        scalar_t tmp_output = 0.0;

        for (int32_t i = edge_start; i < edge_end; i++)
        {

            scalar_t y = Y[i][m];
            scalar_t x = 0.0;

            if (valid)
            {
                x = X[i][feat_start + threadIdx.x];
            }

            tmp_output += x * y;
        }

        if (valid)
            output[node_index][m][feat_start + threadIdx.x] = tmp_output;
    }
}

torch::Tensor forward_gpu(torch::Tensor X,
                          torch::Tensor Y,
                          torch::Tensor receiver_list,
                          torch::Tensor neighbour_indices,
                          int64_t natoms,
                          int64_t nthreadx,
                          int64_t nthready,
                          int64_t nthreadz)
{

    torch::Tensor output = torch::empty({natoms, Y.size(1), X.size(1)},
                                        torch::TensorOptions()
                                            .dtype(X.dtype())
                                            .device(X.device()));

    int32_t nby = find_integer_divisor(X.size(1), nthreadx);

    dim3 block_dim(natoms, nby);

    dim3 grid_dim(nthreadx, nthready, 1);

    AT_DISPATCH_FLOATING_TYPES(
        X.type(), "forward_gpu", ([&]
                                  {
                    size_t total_buff_size = 0;

                    //total_buff_size += nthreadx * Y.size(1) * sizeof(scalar_t);

                    forward_kernel<scalar_t><<<block_dim, grid_dim, total_buff_size>>>(
                        X.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                        Y.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                        receiver_list.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(),
                        neighbour_indices.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(),
                        output.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>()); }));

    cudaDeviceSynchronize();

    return output;
}

#define FEATS_PER_BLOCK_Y 32
#define M_PER_BLOCK_X 4

// nx = 4, ny = 32
template <typename scalar_t>
__global__ void backward_dX_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> Y,
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> grad_in, // [nnodes, m, feat]
    const torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits> receiver_list,
    const torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits> neighbour_indices,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> grad_X)
{

    extern __shared__ char buffer[];
    size_t offset = 0;
    scalar_t *buffer_grad_in = reinterpret_cast<scalar_t *>(buffer + offset);
    offset += FEATS_PER_BLOCK_Y * Y.size(1) * sizeof(scalar_t);

    int32_t edge_start = neighbour_indices[blockIdx.x];
    int32_t edge_end = 0;

    int32_t node_index = receiver_list[edge_start];

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
    int global_tid = threadIdx.y * blockDim.x + threadIdx.x; // [0 - 128], ny = 4, nx = 32

    int32_t feat_start = blockIdx.y * FEATS_PER_BLOCK_Y; // each block handles FEATS_PER_BLOCK_Y features

    {
        int nx = 32;
        int tidx = global_tid % nx;
        int tidy = global_tid / nx;
        int ny = (blockDim.y * blockDim.x) / nx;

        for (int m = tidy; m < Y.size(1); m += ny)
        {
            scalar_t val = 0.0;

            if (feat_start + tidx < grad_in.size(2))
            {
                val = grad_in[node_index][m][feat_start + tidx];
            }

            buffer_grad_in[m * FEATS_PER_BLOCK_Y + tidx] = val;
        }
    }

    __syncthreads();
    int niter_m = find_integer_divisor(Y.size(1), blockDim.x);             // 16 / 4
    int niter_gradx = find_integer_divisor(FEATS_PER_BLOCK_Y, blockDim.y); // 32 / 32

    for (int32_t i = edge_start; i < edge_end; i++)
    {
        for (int x_idx = 0; x_idx < niter_gradx; x_idx++)
        {
            int feat = feat_start + x_idx * blockDim.y + threadIdx.y;

            scalar_t tmp_output = 0.0;

            // need to reduce along the m dimension, so we divide threads into groups of 4 or 8 and then do warp reductions across those subgroups.
            for (int y_idx = 0; y_idx < niter_m; y_idx++)
            {
                int m = y_idx * blockDim.x + threadIdx.x;

                scalar_t y = 0.0;
                scalar_t tmp_grad_in = 0.0;

                if (m < Y.size(1) && feat < grad_in.size(2))
                {
                    tmp_grad_in = buffer_grad_in[m * FEATS_PER_BLOCK_Y + threadIdx.y];
                }

                if (m < Y.size(1))
                {
                    y = Y[i][m];
                }

                scalar_t tmp_grad = tmp_grad_in * y;

                for (int offset = blockDim.x / 2; offset > 0; offset /= 2)
                {
                    tmp_grad += __shfl_down_sync(FULL_MASK, tmp_grad, offset);
                }

                tmp_output += tmp_grad;
            }

            if (threadIdx.x == 0 && feat < grad_in.size(2))
            {
                grad_X[i][feat] = tmp_output;
            }
        }
    }
}

// ny = 4, nx = 32
template <typename scalar_t>
__global__ void backward_dY_kernel(const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> X,
                                   const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> grad_in, // [nnodes, m, feat]
                                   const torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits> receiver_list,
                                   const torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits> neighbour_indices,
                                   torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> grad_Y)
{

    extern __shared__ char buffer[];
    size_t offset = 0;
    scalar_t *buffer_grad_in = reinterpret_cast<scalar_t *>(buffer + offset);
    offset += X.size(1) * M_PER_BLOCK_X * sizeof(scalar_t); // e.g 128 x 4

    int32_t edge_start = neighbour_indices[blockIdx.x];
    int32_t edge_end = 0;

    int32_t node_index = receiver_list[edge_start];

    if (blockIdx.x == neighbour_indices.size(0) - 1) // nnodes -1
    {
        edge_end = X.size(0); // nedges -1
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

    int32_t m_start = blockIdx.y * M_PER_BLOCK_X; // each block handles M_PER_BLOCK_X m indices

    int niter_m = find_integer_divisor(M_PER_BLOCK_X, blockDim.y);
    int niter_x = find_integer_divisor(X.size(1), blockDim.x);

    for (int m_idx = 0; m_idx < niter_m; m_idx++)
    {
        int local_m = m_idx * blockDim.y + threadIdx.y;
        int global_m = m_start + local_m;

        for (int feat = threadIdx.x; feat < X.size(1); feat += blockDim.x)
        {
            scalar_t val = 0.0;

            if (global_m < grad_in.size(1))
            {
                val = grad_in[node_index][global_m][feat];
            }

            buffer_grad_in[local_m * X.size(1) + feat] = val;
        }
    }

    __syncthreads();

    for (int32_t i = edge_start; i < edge_end; i++)
    {

        // need to reduce along the channel dimension

        for (int m_idx = 0; m_idx < niter_m; m_idx++)
        {
            int local_m = m_idx * blockDim.y + threadIdx.y;
            int global_m = m_start + m_idx * blockDim.y + threadIdx.y;

            scalar_t tmp_output = 0.0;

            for (int x_idx = 0; x_idx < niter_x; x_idx++)
            {
                int feat = x_idx * blockDim.x + threadIdx.x;

                scalar_t tmp_grad_in = 0.0;

                if (global_m < grad_in.size(1) && feat < X.size(1))
                {
                    tmp_grad_in = buffer_grad_in[local_m * X.size(1) + feat];
                }

                scalar_t x = 0.0;

                if (feat < X.size(1))
                {
                    x = X[i][feat];
                }

                scalar_t tmp_grad = tmp_grad_in * x;

                for (int offset = blockDim.x / 2; offset > 0; offset /= 2)
                {
                    tmp_grad += __shfl_down_sync(FULL_MASK, tmp_grad, offset);
                }

                tmp_output += tmp_grad;
            }

            if (threadIdx.x == 0 && global_m < grad_in.size(1))
            {
                grad_Y[i][global_m] = tmp_output;
            }
        }
    }
}

std::vector<torch::Tensor> backward_gpu(torch::Tensor X,
                                        torch::Tensor Y,
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
        gradX = torch::Tensor();
    }

    dim3 grid_dim_x(M_PER_BLOCK_X, FEATS_PER_BLOCK_Y, 1);
    int32_t nbx = find_integer_divisor(X.size(1), FEATS_PER_BLOCK_Y);
    dim3 block_dim_x(natoms, nbx);

    dim3 grid_dim_y(FEATS_PER_BLOCK_Y, M_PER_BLOCK_X, 1);
    int32_t nby = find_integer_divisor(Y.size(1), M_PER_BLOCK_X);
    dim3 block_dim_y(natoms, nby);

    AT_DISPATCH_FLOATING_TYPES(
        X.type(), "backward_gpu",
        ([&]
         {
            if (X.requires_grad())
            {
                size_t buff_size_x = 0;
                buff_size_x += Y.size(1) * FEATS_PER_BLOCK_Y * sizeof(scalar_t);

                backward_dX_kernel<scalar_t><<<block_dim_x, grid_dim_x, buff_size_x>>>(
                    Y.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                    grad_in.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                    receiver_list.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(),
                    neighbour_indices.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(),
                    gradX.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>());
            }

            if (Y.requires_grad())
            {
                size_t buff_size_y = 0;
                buff_size_y += X.size(1) * M_PER_BLOCK_X * sizeof(scalar_t);

                backward_dY_kernel<scalar_t><<<block_dim_y, grid_dim_y, buff_size_y>>>(
                    X.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                    grad_in.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                    receiver_list.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(),
                    neighbour_indices.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(),
                    gradY.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>());
            } }));

    cudaDeviceSynchronize();

    return {gradX, gradY};
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

/*
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("calculate_neighbours", &calculate_neighbours_gpu, "computes neighbourlist starts from sender list.");
    m.def("forward", &forward_gpu, "ops forward GPU.");
    m.def("backward", &backward_gpu, "ops backward GPU.");
}
*/

class OuterProductAutograd : public Function<OuterProductAutograd>
{
public:
    static torch::Tensor forward(
        AutogradContext *ctx,
        torch::Tensor X,
        torch::Tensor Y,
        torch::Tensor receiver_list,
        int64_t natoms)
    {
        torch::Tensor neighbours = calculate_neighbours_gpu(receiver_list, natoms, 64);

        if (X.requires_grad() || Y.requires_grad())
        {
            ctx->saved_data["natoms"] = natoms;

            ctx->save_for_backward({X, Y, receiver_list, neighbours});
        }

        torch::Tensor result = forward_gpu(X, Y, receiver_list, neighbours, natoms, 32, 4, 1);

        return result;
    }

    static variable_list backward(AutogradContext *ctx, variable_list grad_outputs)
    {
        auto saved_variables = ctx->get_saved_variables();

        auto X = saved_variables[0];
        auto Y = saved_variables[1];
        auto receiver_list = saved_variables[2];
        auto neighbours = saved_variables[3];

        int64_t natoms = ctx->saved_data["natoms"].toInt();

        // cout << "grad_outputs shape: " << grad_outputs[0].sizes() <<  endl;

        auto result = backward_gpu(X, Y, grad_outputs[0], receiver_list, neighbours, natoms);

        torch::Tensor undef;

        return {result[0], result[1], undef, undef};
    }
};

torch::Tensor invariant_outer_product_scattersum(torch::Tensor X, torch::Tensor Y, torch::Tensor sender_list, int64_t natoms)
{
    return OuterProductAutograd::apply(X, Y, sender_list, natoms);
}

TORCH_LIBRARY(ops_cu, m)
{
    m.def("invariant_outer_product_scattersum", &invariant_outer_product_scattersum);
    m.def("calculate_neighbours", &calculate_neighbours_gpu);
    m.def("forward", &forward_gpu);
    m.def("backward", &backward_gpu);
}
