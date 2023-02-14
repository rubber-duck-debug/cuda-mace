#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>

using namespace std;
using namespace torch::indexing;

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

__global__ void sparse_tensor_product_cuda_forward_kernel(const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> X1,
                                                          const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> mu_1,
                                                          const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> X2,
                                                          const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> mu_2,
                                                          const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> cg_coefficients,
                                                          torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> output,
                                                          const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> mu_3)
{

    extern __shared__ char buffer[];
    int offset = 0;

    float *buffer_cg_coefficients = reinterpret_cast<float *>(buffer);
    offset += cg_coefficients.size(0) * sizeof(float);

    int *buffer_mu_1 = reinterpret_cast<int *>(buffer + offset);
    offset += mu_1.size(0) * sizeof(int);
    int *buffer_mu_2 = reinterpret_cast<int *>(buffer + offset);
    offset += mu_2.size(0) * sizeof(int);
    int *buffer_mu_3 = reinterpret_cast<int *>(buffer + offset);
    offset += mu_3.size(0) * sizeof(int);

    float *buffer_x1 = reinterpret_cast<float *>(buffer + offset);
    offset += blockDim.y * X1.size(2) * sizeof(float);
    float *buffer_x2 = reinterpret_cast<float *>(buffer + offset);
    offset += X2.size(1) * X2.size(2) * sizeof(float);
    float *buffer_x3 = reinterpret_cast<float *>(buffer + offset);
    offset += blockDim.y * output.size(2) * sizeof(float);

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;

    int n_threads_x = blockDim.x;
    int n_threads_y = blockDim.y;
    int n_threads_z = blockDim.z;

    int n_blocks_x = gridDim.x;
    int n_blocks_y = gridDim.y;
    int n_blocks_z = gridDim.z;

    int X1_size_0 = X1.size(0);
    int X1_size_1 = X1.size(1);
    int X1_size_2 = X1.size(2);

    int X2_size_0 = X2.size(0);
    int X2_size_1 = X2.size(1);
    int X2_size_2 = X2.size(2);

    int X3_size_0 = output.size(0);
    int X3_size_1 = output.size(1);
    int X3_size_2 = output.size(2);

    // load all cg coefficients, mu indices into shared memory

    /// y: 0 0 0 0 1 1 1 1 2 2 2 2 3 3 3 3
    /// x: 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3
    for (int i = ty * n_threads_x + tx; i < cg_coefficients.size(0); i += n_threads_y * n_threads_x)
    {
        buffer_cg_coefficients[i] = cg_coefficients[i];
    }

    for (int i = ty * n_threads_x + tx; i < mu_1.size(0); i += n_threads_y * n_threads_x)
    {
        buffer_mu_1[i] = mu_1[i];
        buffer_mu_2[i] = mu_2[i];
        buffer_mu_3[i] = mu_3[i];
    }

    __syncthreads();

    for (int atom_idx = bx; atom_idx < X1.size(0); atom_idx += n_blocks_x)
    { // loop over edges

        if (ty == 0)
        {
            for (int ix = tx; ix < X2.size(2); ix += n_threads_x)
            {
                buffer_x2[ix] = X2[atom_idx][0][ix];
            }
        }
        __syncthreads();

        for (int idx_y = ty; idx_y < X1.size(1); idx_y += n_threads_y)
        {

            // loop over channels with threadIdx.y

            // zero out or load shared memory for subset of X1[ix, :], X2[ix, :] and X3[ix, :]
            for (int ix = tx; ix < X1.size(2); ix += n_threads_x)
            {
                buffer_x1[ty * X1_size_2 + ix] = X1[atom_idx][idx_y][ix];
            }

            for (int ix = tx; ix < output.size(2); ix += n_threads_x)
            {
                buffer_x3[ty * X3_size_2 + ix] = 0.0;
            }

            __syncthreads();

            for (int instruction_idx = tx; instruction_idx < mu_1.size(0); instruction_idx += n_threads_x)
            {

                int X1_index = buffer_mu_1[instruction_idx];
                int X2_index = buffer_mu_2[instruction_idx];
                int X3_index = buffer_mu_3[instruction_idx];

                float x1 = buffer_x1[ty * X1_size_2 + X1_index];
                float x2 = buffer_x2[X2_index];

                float cg_coeff = buffer_cg_coefficients[instruction_idx];

                // lots of memory bank conflicts here, need to think of better implementation
                atomicAdd(&buffer_x3[ty * X3_size_2 + X3_index], cg_coeff * x1 * x2);
            }

            // write out to global memory

            for (int ix = tx; ix < output.size(2); ix += n_threads_x)
            {
                output[atom_idx][idx_y][ix] = buffer_x3[ty * X3_size_2 + ix];
            }
        }
    }
}

__global__ void sparse_weighted_tensor_product_cuda_forward_kernel(const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> X1,
                                                                   const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> mu_1,
                                                                   const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> X2,
                                                                   const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> mu_2,
                                                                   const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> cg_coefficients,
                                                                   const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> weights,
                                                                   const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> weight_indices,
                                                                   torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> output)
{

    extern __shared__ char buffer[];
    int offset = 0;

    float *buffer_cg_coefficients = reinterpret_cast<float *>(buffer);
    offset += cg_coefficients.size(0) * sizeof(float);

    
    int *buffer_mu_1 = reinterpret_cast<int *>(buffer + offset);
    offset += mu_1.size(0) * sizeof(int);
    int *buffer_mu_2 = reinterpret_cast<int *>(buffer + offset);
    offset += mu_2.size(0) * sizeof(int);

    float *buffer_x1 = reinterpret_cast<float *>(buffer + offset);
    offset += blockDim.y * X1.size(2) * sizeof(float);
    float *buffer_x2 = reinterpret_cast<float *>(buffer + offset);
    offset += X2.size(1) * X2.size(2) * sizeof(float);

    float *buffer_weights = reinterpret_cast<float *>(buffer + offset);
    offset += blockDim.y * weights.size(1) * sizeof(float);
    int *buffer_weight_indices = reinterpret_cast<int *>(buffer + offset);
    offset += weight_indices.size(0) * sizeof(int);

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;

    int n_threads_x = blockDim.x;
    int n_threads_y = blockDim.y;
    int n_threads_z = blockDim.z;

    int n_blocks_x = gridDim.x;
    int n_blocks_y = gridDim.y;
    int n_blocks_z = gridDim.z;

    int X1_size_0 = X1.size(0);
    int X1_size_1 = X1.size(1);
    int X1_size_2 = X1.size(2);

    int X2_size_0 = X2.size(0);
    int X2_size_1 = X2.size(1);
    int X2_size_2 = X2.size(2);

    int weights_size_0 = weights.size(0);
    int weights_size_1 = weights.size(1);

    unsigned int mask = 0xffffffff;
    // load all cg coefficients, mu indices into shared memory

    for (int i = ty * n_threads_x + tx; i < cg_coefficients.size(0); i += n_threads_y * n_threads_x)
    {
        buffer_cg_coefficients[i] = cg_coefficients[i];
    }

    for (int i = ty * n_threads_x + tx; i < mu_1.size(0); i += n_threads_y * n_threads_x)
    {
        buffer_mu_1[i] = mu_1[i];
        buffer_mu_2[i] = mu_2[i];
    }

    for (int i = ty * n_threads_x + tx; i < weight_indices.size(0); i += n_threads_y * n_threads_x)
    {
        buffer_weight_indices[i] = weight_indices[i];
    }

    __syncthreads();

    for (int atom_idx = bx; atom_idx < X1.size(0); atom_idx += n_blocks_x)
    { // loop over edges

        if (ty == 0)
        {
            for (int ix = tx; ix < X2.size(2); ix += n_threads_x)
            {
                buffer_x2[ix] = X2[atom_idx][0][ix];
            }
        }

        __syncthreads();

        for (int idx_y = ty; idx_y < X1.size(1); idx_y += n_threads_y)
        {

            // buffer_out[ty] = 0.0;

            // loop over channels with threadIdx.y

            // zero out or load shared memory for subset of X1[ix, :], X2[ix, :] and X3[ix, :]
            for (int ix = tx; ix < X1.size(2); ix += n_threads_x)
            {
                buffer_x1[ty * X1_size_2 + ix] = X1[atom_idx][idx_y][ix];
            }

            for (int ix = tx; ix < weights.size(1); ix += n_threads_x)
            {
                buffer_weights[ty * weights_size_1 + ix] = weights[idx_y][ix];
            }

            __syncthreads();

            float sum = 0.0;

            for (int instruction_idx = tx; instruction_idx < mu_1.size(0); instruction_idx += n_threads_x)
            {

                int X1_index = buffer_mu_1[instruction_idx];
                int X2_index = buffer_mu_2[instruction_idx];

                float x1 = buffer_x1[ty * X1_size_2 + X1_index];
                float x2 = buffer_x2[X2_index];

                int weight_index = buffer_weight_indices[instruction_idx];

                float weight = buffer_weights[ty * weights_size_1 + weight_index];

                float cg_coeff = buffer_cg_coefficients[instruction_idx];

                sum += weight * cg_coeff * x1 * x2;
            }

            // reduce sum with warp primitive

            // y: 0 0 0 0 1 1 1 1 2 2 2 2 3 3 3 3
            // x: 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3; nthreads_x = 4

            for (int offset = n_threads_x / 2; offset > 0; offset /= 2)
            {
                sum += __shfl_down_sync(mask, sum, offset);
            }

            if (tx == 0)
            {
                output[atom_idx][idx_y] = sum; 
            }
        }
    }
}

__global__ void sparse_weighted_tensor_product_cuda_backward_dX1_kernel(const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> X1,
                                                                                torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> d_X1,
                                                                                const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> mu_1,
                                                                                const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> X2,
                                                                                const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> mu_2,
                                                                                const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> cg_coefficients,
                                                                                const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> weights,
                                                                                const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> weight_indices,
                                                                                const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> grad_input)
{

    extern __shared__ char buffer[];
    int offset = 0;

    float *buffer_cg_coefficients = reinterpret_cast<float *>(buffer);
    offset += cg_coefficients.size(0) * sizeof(float);

    int *buffer_mu_1 = reinterpret_cast<int *>(buffer + offset);
    offset += mu_1.size(0) * sizeof(int);
    int *buffer_mu_2 = reinterpret_cast<int *>(buffer + offset);
    offset += mu_2.size(0) * sizeof(int);

    float *buffer_x1 = reinterpret_cast<float *>(buffer + offset);
    offset += blockDim.y * X1.size(2) * sizeof(float);
    float *buffer_x2 = reinterpret_cast<float *>(buffer + offset);
    offset += X2.size(1) * X2.size(2) * sizeof(float);

    float *buffer_weights = reinterpret_cast<float *>(buffer + offset);
    offset += blockDim.y * weights.size(1) * sizeof(float);
    int *buffer_weight_indices = reinterpret_cast<int *>(buffer + offset);
    offset += weight_indices.size(0) * sizeof(int);

    float *buffer_d_X1 = reinterpret_cast<float *>(buffer + offset); 
    offset += blockDim.y * d_X1.size(2) * sizeof(float);

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;

    int n_threads_x = blockDim.x;
    int n_threads_y = blockDim.y;
    int n_threads_z = blockDim.z;

    int n_blocks_x = gridDim.x;
    int n_blocks_y = gridDim.y;
    int n_blocks_z = gridDim.z;

    int X1_size_0 = X1.size(0);
    int X1_size_1 = X1.size(1);
    int X1_size_2 = X1.size(2);

    int X2_size_0 = X2.size(0);
    int X2_size_1 = X2.size(1);
    int X2_size_2 = X2.size(2);

    int weights_size_0 = weights.size(0);
    int weights_size_1 = weights.size(1);

    unsigned int mask = 0xffffffff;
    // load all cg coefficients, mu indices into shared memory

    for (int i = ty * n_threads_x + tx; i < cg_coefficients.size(0); i += n_threads_y * n_threads_x)
    {
        buffer_cg_coefficients[i] = cg_coefficients[i];
    }

    for (int i = ty * n_threads_x + tx; i < mu_1.size(0); i += n_threads_y * n_threads_x)
    {
        buffer_mu_1[i] = mu_1[i];
        buffer_mu_2[i] = mu_2[i];
    }

    for (int i = ty * n_threads_x + tx; i < weight_indices.size(0); i += n_threads_y * n_threads_x)
    {
        buffer_weight_indices[i] = weight_indices[i];
    }

    __syncthreads();

    for (int atom_idx = bx; atom_idx < X1.size(0); atom_idx += n_blocks_x)
    { // loop over edges

        if (ty == 0)
        {
            for (int ix = tx; ix < X2.size(2); ix += n_threads_x)
            {
                buffer_x2[ix] = X2[atom_idx][0][ix];
            }
        }

        __syncthreads();

        for (int idx_y = ty; idx_y < X1.size(1); idx_y += n_threads_y)
        {

            // buffer_out[ty] = 0.0;

            // loop over channels with threadIdx.y

            // zero out or load shared memory for subset of X1[ix, :], X2[ix, :] and X3[ix, :]
            for (int ix = tx; ix < X1.size(2); ix += n_threads_x)
            {
                buffer_x1[ty * X1_size_2 + ix] = X1[atom_idx][idx_y][ix];
            }

            for (int ix = tx; ix < weights.size(1); ix += n_threads_x)
            {
                buffer_weights[ty * weights_size_1 + ix] = weights[idx_y][ix];
            }

            for (int ix = tx; ix < d_X1.size(2); ix += n_threads_x)
            {
                buffer_d_X1[ty * X1_size_2 + ix] = 0.0;
            }

            __syncthreads();


            for (int instruction_idx = tx; instruction_idx < mu_1.size(0); instruction_idx += n_threads_x)
            {

                int X1_index = buffer_mu_1[instruction_idx];
                int X2_index = buffer_mu_2[instruction_idx];

                float x1 = buffer_x1[ty * X1_size_2 + X1_index];
                float x2 = buffer_x2[X2_index];

                int weight_index = buffer_weight_indices[instruction_idx];

                float weight = buffer_weights[ty * weights_size_1 + weight_index];

                float cg_coeff = buffer_cg_coefficients[instruction_idx];

                atomicAdd(&buffer_d_X1[ty * X1_size_2 + X1_index], weight * cg_coeff * x2);
            }

            for (int sph_idx = tx; sph_idx < d_X1.size(2); sph_idx += n_threads_x)
            {
                d_X1[atom_idx][idx_y][sph_idx] = grad_input[atom_idx][idx_y] * buffer_d_X1[ty * X1_size_2 + sph_idx];
            }
        }
    }
}

std::vector<torch::Tensor> sparse_tensor_product_cuda_forward(
    torch::Tensor X1,
    torch::Tensor mu_1,
    torch::Tensor X2,
    torch::Tensor mu_2,
    torch::Tensor cg_coefficients,
    torch::Tensor mu_3,
    int64_t output_size)
{

    auto output = torch::zeros({X1.size(0), X1.size(1), output_size},
                               torch::TensorOptions()
                                   .dtype(X1.dtype())
                                   .device(X1.device()));

    const auto batch_sizex = X1.size(0);

    auto find_num_blocks = [](int x, int bdim)
    { return (x + bdim - 1) / bdim; };

    dim3 block_dim(batch_sizex);
    // int nbx = find_num_blocks(nx, block_dim.x);

    dim3 grid_dim(8, 64);

    size_t total_buff_size = (3 * mu_1.size(0)) * sizeof(int) + (cg_coefficients.size(0) * sizeof(float)) +
                             (grid_dim.y * output_size * sizeof(float)) + (grid_dim.y * X1.size(2) * sizeof(float)) + (X2.size(1) * X2.size(2) * sizeof(float));

    sparse_tensor_product_cuda_forward_kernel<<<block_dim, grid_dim, total_buff_size>>>(X1.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
                                                                                        mu_1.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                        X2.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
                                                                                        mu_2.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                        cg_coefficients.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
                                                                                        output.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
                                                                                        mu_3.packed_accessor32<int, 1, torch::RestrictPtrTraits>());

    cudaDeviceSynchronize();

    return {output};
}

std::vector<torch::Tensor> sparse_weighted_tensor_product_cuda_forward(
    torch::Tensor X1,
    torch::Tensor mu_1,
    torch::Tensor X2,
    torch::Tensor mu_2,
    torch::Tensor cg_coefficients,
    torch::Tensor weights,
    torch::Tensor weight_indices)
{

    auto output = torch::zeros({X1.size(0), X1.size(1)},
                               torch::TensorOptions()
                                   .dtype(X1.dtype())
                                   .device(X1.device()));

    const auto batch_sizex = X1.size(0);

    auto find_num_blocks = [](int x, int bdim)
    { return (x + bdim - 1) / bdim; };

    dim3 block_dim(batch_sizex);
    dim3 grid_dim(8, 64);

    size_t total_buff_size = (2 * mu_1.size(0)) * sizeof(int) + (cg_coefficients.size(0) * sizeof(float)) +
                             (grid_dim.y * weights.size(1) * sizeof(float)) + (weight_indices.size(0) * sizeof(int)) + (grid_dim.y * X1.size(2) * sizeof(float)) + (X2.size(1) * X2.size(2) * sizeof(float)) + (grid_dim.y * sizeof(float));

    sparse_weighted_tensor_product_cuda_forward_kernel<<<block_dim, grid_dim, total_buff_size>>>(X1.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
                                                                                                 mu_1.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                                 X2.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
                                                                                                 mu_2.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                                 cg_coefficients.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
                                                                                                 weights.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                                                                                                 weight_indices.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                                 output.packed_accessor32<float, 2, torch::RestrictPtrTraits>());

    cudaDeviceSynchronize();

    return {output};
}


std::vector<torch::Tensor> sparse_weighted_tensor_product_cuda_backward_dX1(
    torch::Tensor X1,
    torch::Tensor mu_1,
    torch::Tensor X2,
    torch::Tensor mu_2,
    torch::Tensor cg_coefficients,
    torch::Tensor weights,
    torch::Tensor weight_indices,
    torch::Tensor grad_output)
{

    // __global__ void sparse_weighted_tensor_product_cuda_forward_and_backward_dX1_kernel(const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> X1,
    //                                                                             torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> d_X1,
    //                                                                             const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> mu_1,
    //                                                                             const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> X2,
    //                                                                             const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> mu_2,
    //                                                                             const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> cg_coefficients,
    //                                                                             const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> weights,
    //                                                                             const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> weight_indices,
    //                                                                             const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> grad_input,
    //                                                                             torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> output)

    auto dX1 = torch::zeros_like(X1,
                               torch::TensorOptions()
                                   .dtype(X1.dtype())
                                   .device(X1.device()));

    const auto batch_sizex = X1.size(0);

    auto find_num_blocks = [](int x, int bdim)
    { return (x + bdim - 1) / bdim; };

    dim3 block_dim(batch_sizex);
    dim3 grid_dim(8, 64);

    size_t total_buff_size = (2 * mu_1.size(0)) * sizeof(int) + (cg_coefficients.size(0) * sizeof(float)) +
                             (grid_dim.y * weights.size(1) * sizeof(float)) + (weight_indices.size(0) * sizeof(int)) + (grid_dim.y * X1.size(2) * sizeof(float)) + (X2.size(1) * X2.size(2) * sizeof(float)) + (grid_dim.y * dX1.size(2)* sizeof(float));

    sparse_weighted_tensor_product_cuda_backward_dX1_kernel<<<block_dim, grid_dim, total_buff_size>>>(X1.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
                                                                                                dX1.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
                                                                                                 mu_1.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                                 X2.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
                                                                                                 mu_2.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                                 cg_coefficients.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
                                                                                                 weights.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                                                                                                 weight_indices.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                                 grad_output.packed_accessor32<float, 2, torch::RestrictPtrTraits>()
                                                                                                 );

    cudaDeviceSynchronize();

    return {dX1};
}

std::vector<torch::Tensor> sparse_tensor_product_gpu_forward(
    torch::Tensor X1,
    torch::Tensor mu_1,
    torch::Tensor X2,
    torch::Tensor mu_2,
    torch::Tensor cg_coefficients,
    torch::Tensor mu_3,
    int64_t output_size)
{

    CHECK_INPUT(X1);
    CHECK_INPUT(mu_1);
    CHECK_INPUT(X2);
    CHECK_INPUT(mu_2);
    CHECK_INPUT(cg_coefficients);
    CHECK_INPUT(mu_3);

    return sparse_tensor_product_cuda_forward(X1, mu_1, X2, mu_2, cg_coefficients, mu_3, output_size);
}

std::vector<torch::Tensor> sparse_weighted_tensor_product_gpu_forward(
    torch::Tensor X1,
    torch::Tensor mu_1,
    torch::Tensor X2,
    torch::Tensor mu_2,
    torch::Tensor cg_coefficients,
    torch::Tensor weights,
    torch::Tensor weight_indices)
{

    CHECK_INPUT(X1);
    CHECK_INPUT(mu_1);
    CHECK_INPUT(X2);
    CHECK_INPUT(mu_2);
    CHECK_INPUT(cg_coefficients);
    CHECK_INPUT(weight_indices);
    CHECK_INPUT(weights);
 

    return sparse_weighted_tensor_product_cuda_forward(X1, mu_1, X2, mu_2, cg_coefficients, weights, weight_indices);
}


std::vector<torch::Tensor> sparse_weighted_tensor_product_gpu_backward_dX1(
    torch::Tensor X1,
    torch::Tensor mu_1,
    torch::Tensor X2,
    torch::Tensor mu_2,
    torch::Tensor cg_coefficients,
    torch::Tensor weights,
    torch::Tensor weight_indices,
    torch::Tensor grad_output)
{

    CHECK_INPUT(X1);
    CHECK_INPUT(mu_1);
    CHECK_INPUT(X2);
    CHECK_INPUT(mu_2);
    CHECK_INPUT(cg_coefficients);
    CHECK_INPUT(weight_indices);
    CHECK_INPUT(weights);
    CHECK_INPUT(grad_output);

    return sparse_weighted_tensor_product_cuda_backward_dX1(X1, mu_1, X2, mu_2, cg_coefficients, weights, weight_indices, grad_output);
}

//

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &sparse_tensor_product_gpu_forward, "Sparse Tensor Product Forward (CUDA)");
    m.def("weighted_forward", &sparse_weighted_tensor_product_gpu_forward, "Sparse Weighted Tensor Product Forward (CUDA)");
    m.def("weighted_backward_dX1", &sparse_weighted_tensor_product_gpu_backward_dX1, "Sparse Weighted Tensor Product Forward and backward wrt. X1 (CUDA)");
}
