#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>

using namespace std;
using namespace torch::indexing;

#define BLOCK_SIZE 8


__global__ void sparse_tensor_product_cuda_forward_kernel_1(const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> X1,
                                                          const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> mu_1,
                                                          const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> mu_1_offsets,
                                                          const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> X2,
                                                          const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> mu_2,
                                                          const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> mu_2_offsets,
                                                          const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> cg_coefficients,
                                                          const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> n_cg_coefficients,
                                                          const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> cg_offsets,
                                                          torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> output,
                                                          const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> mu_3,
                                                          const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> mu_3_offsets)
{

    extern __shared__ char buffer[];
    int offset = 0;

    float *buffer_cg_coefficients = reinterpret_cast<float *>(buffer);
    offset += cg_coefficients.size(0) * sizeof(float);
    int *buffer_n_cg_elements = reinterpret_cast<int *>(buffer + offset);
    offset += n_cg_coefficients.size(0) * sizeof(int);
    int *buffer_cg_offsets = reinterpret_cast<int *>(buffer + offset);
    offset += cg_offsets.size(0) * sizeof(int);

    int *buffer_mu_1 = reinterpret_cast<int *>(buffer + offset);
    offset += mu_1.size(0) * sizeof(int);
    int *buffer_mu_2 = reinterpret_cast<int *>(buffer + offset);
    offset += mu_2.size(0) * sizeof(int);
    int *buffer_mu_3 = reinterpret_cast<int *>(buffer + offset);
    offset += mu_3.size(0) * sizeof(int);

    int *buffer_mu_1_offsets = reinterpret_cast<int *>(buffer + offset);
    offset += mu_1_offsets.size(0) * sizeof(int);
    int *buffer_mu_2_offsets = reinterpret_cast<int *>(buffer + offset);
    offset += mu_2_offsets.size(0) * sizeof(int);
    int *buffer_mu_3_offsets = reinterpret_cast<int *>(buffer + offset);
    offset += mu_3_offsets.size(0) * sizeof(int);

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

    for (int i = ty * n_threads_x + tx; i < cg_coefficients.size(0); i += n_threads_y * n_threads_x)
    {
        buffer_cg_coefficients[i] = cg_coefficients[i];
    }

    for (int i = ty * n_threads_x + tx; i < n_cg_coefficients.size(0); i += n_threads_y * n_threads_x)
    {
        buffer_n_cg_elements[i] = n_cg_coefficients[i];
        buffer_cg_offsets[i] = cg_offsets[i];
    }
    __syncthreads();

    for (int i = ty * n_threads_x + tx; i < mu_1.size(0); i += n_threads_y * n_threads_x)
    {
        buffer_mu_1[i] = mu_1[i];
        buffer_mu_2[i] = mu_2[i];
        buffer_mu_3[i] = mu_3[i];
    }
    __syncthreads();

    for (int i = ty * n_threads_x + tx; i < mu_1_offsets.size(0); i += n_threads_y * n_threads_x)
    {
        buffer_mu_1_offsets[i] = mu_1_offsets[i];
        buffer_mu_2_offsets[i] = mu_2_offsets[i];
        buffer_mu_3_offsets[i] = mu_3_offsets[i];
    }
    __syncthreads();

    int n_iter_y = (int)ceil((float)X1.size(1) / n_threads_y);

    for (int ix = bx; ix < X1.size(0); ix += n_blocks_x)
    { // loop over edges

        for (int iy = 0; iy < n_iter_y; iy++)
        {
            int idx_y = iy * n_threads_y + ty;

            if (idx_y < X1.size(1))
            {

                // loop over channels with threadIdx.y

                // zero out or load shared memory for subset of X1[ix, :], X2[ix, :] and X3[ix, :]
                for (int iz = tx; iz < X1.size(2); iz += n_threads_x)
                {
                    buffer_x1[ty * X1_size_2 + iz] = X1[ix][idx_y][iz];
                }

                for (int iz = tx; iz < X2.size(2); iz += n_threads_x)
                {
                    buffer_x2[iz] = X1[ix][0][iz];
                }

                for (int iz = tx; iz < output.size(2); iz += n_threads_x)
                {
                    buffer_x3[ty * X3_size_2 + iz] = 0.0;
                }

                for (int instruction_idx = tz; instruction_idx < mu_1_offsets.size(0); instruction_idx+= n_threads_z)
                {

                    int X1_start = buffer_mu_1_offsets[instruction_idx];
                    int X2_start = buffer_mu_2_offsets[instruction_idx];
                    int X3_start = buffer_mu_3_offsets[instruction_idx];

                    int n_elements = buffer_n_cg_elements[instruction_idx];
                    int cg_start = buffer_cg_offsets[instruction_idx];

                    for (int iz = tx; iz < n_elements; iz += n_threads_x)
                    {

                        float x1 = buffer_x1[ty * X1_size_2 + X1_start + buffer_mu_1[iz]];
                        float x2 = buffer_x2[X2_start + buffer_mu_2[iz]];

                        float cg_coeff = buffer_cg_coefficients[cg_start + iz];

                        atomicAdd(&buffer_x3[ty * X3_size_2 + X3_start + buffer_mu_3[iz]], cg_coeff * x1 * x2);
                    }
                }

                // write out to global memory

                for (int iz = tx; iz < output.size(2); iz += n_threads_x)
                {
                    output[ix][idx_y][iz] = buffer_x3[ty * X3_size_2 + iz];
                }
            }
        }
    }
}


__global__ void sparse_tensor_product_cuda_forward_kernel_2(const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> X1,
                                                          const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> mu_1,
                                                          const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> mu_1_offsets,
                                                          const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> X2,
                                                          const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> mu_2,
                                                          const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> mu_2_offsets,
                                                          const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> cg_coefficients,
                                                          const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> n_cg_coefficients,
                                                          const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> cg_offsets,
                                                          torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> output,
                                                          const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> mu_3,
                                                          const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> mu_3_offsets)
{

    extern __shared__ char buffer[];
    int offset = 0;

    float *buffer_cg_coefficients = reinterpret_cast<float *>(buffer);
    offset += cg_coefficients.size(0) * sizeof(float);
    int *buffer_n_cg_elements = reinterpret_cast<int *>(buffer + offset);
    offset += n_cg_coefficients.size(0) * sizeof(int);
    int *buffer_cg_offsets = reinterpret_cast<int *>(buffer + offset);
    offset += cg_offsets.size(0) * sizeof(int);

    int *buffer_mu_1 = reinterpret_cast<int *>(buffer + offset);
    offset += mu_1.size(0) * sizeof(int);
    int *buffer_mu_2 = reinterpret_cast<int *>(buffer + offset);
    offset += mu_2.size(0) * sizeof(int);
    int *buffer_mu_3 = reinterpret_cast<int *>(buffer + offset);
    offset += mu_3.size(0) * sizeof(int);

    int *buffer_mu_1_offsets = reinterpret_cast<int *>(buffer + offset);
    offset += mu_1_offsets.size(0) * sizeof(int);
    int *buffer_mu_2_offsets = reinterpret_cast<int *>(buffer + offset);
    offset += mu_2_offsets.size(0) * sizeof(int);
    int *buffer_mu_3_offsets = reinterpret_cast<int *>(buffer + offset);
    offset += mu_3_offsets.size(0) * sizeof(int);

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

    for (int i = ty * n_threads_x + tx; i < cg_coefficients.size(0); i += n_threads_y * n_threads_x)
    {
        buffer_cg_coefficients[i] = cg_coefficients[i];
    }

    for (int i = ty * n_threads_x + tx; i < n_cg_coefficients.size(0); i += n_threads_y * n_threads_x)
    {
        buffer_n_cg_elements[i] = n_cg_coefficients[i];
        buffer_cg_offsets[i] = cg_offsets[i];
    }
    __syncthreads();

    for (int i = ty * n_threads_x + tx; i < mu_1.size(0); i += n_threads_y * n_threads_x)
    {
        buffer_mu_1[i] = mu_1[i];
        buffer_mu_2[i] = mu_2[i];
        buffer_mu_3[i] = mu_3[i];
    }
    __syncthreads();

    for (int i = ty * n_threads_x + tx; i < mu_1_offsets.size(0); i += n_threads_y * n_threads_x)
    {
        buffer_mu_1_offsets[i] = mu_1_offsets[i];
        buffer_mu_2_offsets[i] = mu_2_offsets[i];
        buffer_mu_3_offsets[i] = mu_3_offsets[i];
    }
    __syncthreads();

    int n_iter_y = (int)ceil((float)X1.size(1) / n_threads_y);

    for (int ix = bx; ix < X1.size(0); ix += n_blocks_x)
    { // loop over edges

        for (int iy = 0; iy < n_iter_y; iy++)
        {
            int idx_y = iy * n_threads_y + ty;

            if (idx_y < X1.size(1))
            {

                // loop over channels with threadIdx.y

                // zero out or load shared memory for subset of X1[ix, :], X2[ix, :] and X3[ix, :]
                for (int iz = tx; iz < X1.size(2); iz += n_threads_x)
                {
                    buffer_x1[ty * X1_size_2 + iz] = X1[ix][idx_y][iz];
                }

                for (int iz = tx; iz < X2.size(2); iz += n_threads_x)
                {
                    buffer_x2[iz] = X1[ix][0][iz];
                }

                for (int iz = tx; iz < output.size(2); iz += n_threads_x)
                {
                    buffer_x3[ty * X3_size_2 + iz] = 0.0;
                }

                for (int instruction_idx = tx; instruction_idx < mu_1_offsets.size(0); instruction_idx+= n_threads_x)
                {

                    int X1_start = buffer_mu_1_offsets[instruction_idx];
                    int X2_start = buffer_mu_2_offsets[instruction_idx];
                    int X3_start = buffer_mu_3_offsets[instruction_idx];

                    int n_elements = buffer_n_cg_elements[instruction_idx];
                    int cg_start = buffer_cg_offsets[instruction_idx];

                    float sum = 0.0;

                    int last_idx = 0;

                    for (int iz = 0; iz < n_elements; iz ++) {

                        float x1 = buffer_x1[ty * X1_size_2 + X1_start + buffer_mu_1[iz]];
                        float x2 = buffer_x2[X2_start + buffer_mu_2[iz]];
                        float cg_coeff = buffer_cg_coefficients[cg_start + iz];

                        sum += cg_coeff * x1 * x2;

                        last_idx = iz;
                    }

                    buffer_x3[ty * X3_size_2 + X3_start + buffer_mu_3[last_idx]] = sum;
                }

                // write out to global memory

                for (int iz = tx; iz < output.size(2); iz += n_threads_x)
                {
                    output[ix][idx_y][iz] = buffer_x3[ty * X3_size_2 + iz];
                }
            }
        }
    }
}

std::vector<torch::Tensor> sparse_tensor_product_cuda_forward_1(
    torch::Tensor X1,
    torch::Tensor mu_1,
    torch::Tensor mu_1_offsets,
    torch::Tensor X2,
    torch::Tensor mu_2,
    torch::Tensor mu_2_offsets,
    torch::Tensor cg_coefficients,
    torch::Tensor n_cg_coefficients,
    torch::Tensor cg_offsets,
    torch::Tensor mu_3,
    torch::Tensor mu_3_offsets,
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

    dim3 grid_dim(4, 32, 2);

    size_t total_buff_size = (3 * mu_1.size(0) + 3 * mu_1_offsets.size(0)) * sizeof(int) + (cg_coefficients.size(0) * sizeof(float)) + ((n_cg_coefficients.size(0) + cg_offsets.size(0))  * sizeof(int)) +
                             (grid_dim.y * output_size * sizeof(float)) + (grid_dim.y * X1.size(2) * sizeof(float)) + (X2.size(1) * X2.size(2) * sizeof(float));

    // printf("total_buff_size: %d\n", total_buff_size);

    sparse_tensor_product_cuda_forward_kernel_1<<<block_dim, grid_dim, total_buff_size>>>(X1.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
                                                                                        mu_1.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                        mu_1_offsets.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                        X2.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
                                                                                        mu_2.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                        mu_2_offsets.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                        cg_coefficients.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
                                                                                        n_cg_coefficients.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                        cg_offsets.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                        output.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
                                                                                        mu_3.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                        mu_3_offsets.packed_accessor32<int, 1, torch::RestrictPtrTraits>());

    cudaDeviceSynchronize();

    return {output};
}

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> sparse_tensor_product_gpu_forward_1(
    torch::Tensor X1,
    torch::Tensor mu_1,
    torch::Tensor mu_1_offsets,
    torch::Tensor X2,
    torch::Tensor mu_2,
    torch::Tensor mu_2_offsets,
    torch::Tensor cg_coefficients,
    torch::Tensor n_cg_coefficients,
    torch::Tensor cg_offsets,
    torch::Tensor mu_3,
    torch::Tensor mu_3_offsets,
    int64_t output_size)
{

    CHECK_INPUT(X1);
    CHECK_INPUT(mu_1);
    CHECK_INPUT(mu_1_offsets);
    CHECK_INPUT(X2);
    CHECK_INPUT(mu_2);
    CHECK_INPUT(mu_2_offsets);
    CHECK_INPUT(cg_coefficients);
    CHECK_INPUT(n_cg_coefficients);
    CHECK_INPUT(cg_offsets);
    CHECK_INPUT(mu_3);
    CHECK_INPUT(mu_3_offsets);

    return sparse_tensor_product_cuda_forward_1(X1, mu_1, mu_1_offsets, X2, mu_2, mu_2_offsets, cg_coefficients, n_cg_coefficients,cg_offsets, mu_3, mu_3_offsets, output_size);
}


std::vector<torch::Tensor> sparse_tensor_product_cuda_forward_2(
    torch::Tensor X1,
    torch::Tensor mu_1,
    torch::Tensor mu_1_offsets,
    torch::Tensor X2,
    torch::Tensor mu_2,
    torch::Tensor mu_2_offsets,
    torch::Tensor cg_coefficients,
    torch::Tensor n_cg_coefficients,
    torch::Tensor cg_offsets,
    torch::Tensor mu_3,
    torch::Tensor mu_3_offsets,
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

    dim3 grid_dim(4, 64);

    size_t total_buff_size = (3 * mu_1.size(0) + 3 * mu_1_offsets.size(0)) * sizeof(int) + (cg_coefficients.size(0) * sizeof(float)) + ((n_cg_coefficients.size(0) + cg_offsets.size(0))  * sizeof(int)) +
                             (grid_dim.y * output_size * sizeof(float)) + (grid_dim.y * X1.size(2) * sizeof(float)) + (X2.size(1) * X2.size(2) * sizeof(float));

    // printf("total_buff_size: %d\n", total_buff_size);

    sparse_tensor_product_cuda_forward_kernel_2<<<block_dim, grid_dim, total_buff_size>>>(X1.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
                                                                                        mu_1.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                        mu_1_offsets.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                        X2.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
                                                                                        mu_2.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                        mu_2_offsets.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                        cg_coefficients.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
                                                                                        n_cg_coefficients.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                        cg_offsets.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                        output.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
                                                                                        mu_3.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                        mu_3_offsets.packed_accessor32<int, 1, torch::RestrictPtrTraits>());

    cudaDeviceSynchronize();

    return {output};
}

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> sparse_tensor_product_gpu_forward_2(
    torch::Tensor X1,
    torch::Tensor mu_1,
    torch::Tensor mu_1_offsets,
    torch::Tensor X2,
    torch::Tensor mu_2,
    torch::Tensor mu_2_offsets,
    torch::Tensor cg_coefficients,
    torch::Tensor n_cg_coefficients,
    torch::Tensor cg_offsets,
    torch::Tensor mu_3,
    torch::Tensor mu_3_offsets,
    int64_t output_size)
{

    CHECK_INPUT(X1);
    CHECK_INPUT(mu_1);
    CHECK_INPUT(mu_1_offsets);
    CHECK_INPUT(X2);
    CHECK_INPUT(mu_2);
    CHECK_INPUT(mu_2_offsets);
    CHECK_INPUT(cg_coefficients);
    CHECK_INPUT(n_cg_coefficients);
    CHECK_INPUT(cg_offsets);
    CHECK_INPUT(mu_3);
    CHECK_INPUT(mu_3_offsets);

    return sparse_tensor_product_cuda_forward_2(X1, mu_1, mu_1_offsets, X2, mu_2, mu_2_offsets, cg_coefficients, n_cg_coefficients,cg_offsets, mu_3, mu_3_offsets, output_size);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward_1", &sparse_tensor_product_gpu_forward_1, "Sparse Tensor Product Forward (CUDA)");
    m.def("forward_2", &sparse_tensor_product_gpu_forward_2, "Sparse Tensor Product Forward (CUDA)");
}
