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

template <typename scalar_t>
__global__ void sparse_tensor_product_cuda_forward_kernel(const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> X1,
                                                          const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> X2,
                                                          const torch::PackedTensorAccessor32<int16_t, 1, torch::RestrictPtrTraits> mu_1,
                                                          const torch::PackedTensorAccessor32<int16_t, 1, torch::RestrictPtrTraits> mu_2,
                                                          const torch::PackedTensorAccessor32<int16_t, 1, torch::RestrictPtrTraits> mu_3,
                                                          const torch::PackedTensorAccessor32<int16_t, 1, torch::RestrictPtrTraits> X3_ordering,
                                                          const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> cg_coefficients,
                                                          torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> output)
{

    extern __shared__ char buffer[];
    size_t offset = 0;

    scalar_t *buffer_x1 = reinterpret_cast<scalar_t *>(buffer + offset);
    offset += blockDim.x * X1.size(1) * sizeof(scalar_t);
    scalar_t *buffer_x2 = reinterpret_cast<scalar_t *>(buffer + offset);
    offset += X2.size(1) * X2.size(2) * sizeof(scalar_t);

    scalar_t *buffer_cg_coefficients = reinterpret_cast<scalar_t *>(buffer + offset);
    offset += cg_coefficients.size(0) * sizeof(scalar_t);
    scalar_t *buffer_cg_coefficients_X3 = reinterpret_cast<scalar_t *>(buffer + offset);
    offset += cg_coefficients.size(0) * sizeof(scalar_t);

    int16_t *buffer_mu_1 = reinterpret_cast<int16_t *>(buffer + offset);
    offset += mu_1.size(0) * sizeof(int16_t);
    int16_t *buffer_mu_2 = reinterpret_cast<int16_t *>(buffer + offset);
    offset += mu_2.size(0) * sizeof(int16_t);
    int16_t *buffer_mu_3 = reinterpret_cast<int16_t *>(buffer + offset);
    offset += mu_3.size(0) * sizeof(int16_t);

    int16_t *buffer_X3_ordering = reinterpret_cast<int16_t *>(buffer + offset);
    offset += X3_ordering.size(0) * sizeof(int16_t);

    int16_t *buffer_mu_1_X3 = reinterpret_cast<int16_t *>(buffer + offset);
    offset += mu_1.size(0) * sizeof(int16_t);
    int16_t *buffer_mu_2_X3 = reinterpret_cast<int16_t *>(buffer + offset);
    offset += mu_2.size(0) * sizeof(int16_t);
    int16_t *buffer_mu_3_X3 = reinterpret_cast<int16_t *>(buffer + offset);
    offset += mu_3.size(0) * sizeof(int16_t);

    // load all cg coefficients, mu indices into shared memory

    if (threadIdx.y == 0)
    {
        for (int i = threadIdx.x; i < cg_coefficients.size(0); i += blockDim.x)
        {
            buffer_cg_coefficients[i] = cg_coefficients[i];
        }

        for (int i = threadIdx.x; i < mu_1.size(0); i += blockDim.x)
        {
            buffer_mu_1[i] = mu_1[i];
            buffer_mu_2[i] = mu_2[i];
            buffer_mu_3[i] = mu_3[i];
        }

        for (int i = threadIdx.x; i < X3_ordering.size(0); i += blockDim.x)
        {
            buffer_X3_ordering[i] = X3_ordering[i];
        }
    }
    __syncthreads();

    // create re-ordered buffers

    if (threadIdx.y == 0)
    {
        for (int i = threadIdx.x; i < X3_ordering.size(0); i += blockDim.x)
        {
            int16_t x3_i = X3_ordering[i];

            buffer_mu_1_X3[i] = buffer_mu_1[x3_i];
            buffer_mu_2_X3[i] = buffer_mu_2[x3_i];
            buffer_mu_3_X3[i] = buffer_mu_3[x3_i];

            buffer_cg_coefficients_X3[i] = buffer_cg_coefficients[x3_i];
        }
    }

    __syncthreads();

    int atom_idx = blockIdx.x;

    for (int l_id = threadIdx.x; l_id < X2.size(1); l_id += blockDim.x)
    {
        buffer_x2[l_id] = X2[atom_idx][l_id][0];
    }

    __syncthreads();

    for (int channel_id = threadIdx.x; channel_id < X1.size(2); channel_id += blockDim.x) // loop over nchannels
    {
        // zero out or load shared memory for subset of X1[ix, :], X2[ix, :] and X3[ix, :]
        for (int l_id = 0; l_id < X1.size(1); l_id += 1)
        {
            buffer_x1[l_id * blockDim.x + threadIdx.x] = X1[atom_idx][l_id][channel_id];
        }

        __syncthreads();

        int16_t prev_X3_index = 0;
        int16_t X3_index = 0;

        scalar_t sum = 0.0;

        for (int instruction_idx = 0; instruction_idx < mu_1.size(0); instruction_idx += 1)
        {

            int16_t X1_index = buffer_mu_1_X3[instruction_idx];
            int16_t X2_index = buffer_mu_2_X3[instruction_idx];
            X3_index = buffer_mu_3_X3[instruction_idx];

            scalar_t x1 = buffer_x1[X1_index * blockDim.x + threadIdx.x];
            scalar_t x2 = buffer_x2[X2_index];

            scalar_t cg_coeff = buffer_cg_coefficients_X3[instruction_idx];

            // new X3_index so writeout
            if (prev_X3_index != X3_index)
            {
                output[atom_idx][prev_X3_index][channel_id] = sum;
                sum = 0.0;
                prev_X3_index = X3_index;
            }

            sum += cg_coeff * x1 * x2;
        }
        // write out last element
        output[atom_idx][X3_index][channel_id] = sum;
    }
}

torch::Tensor sparse_tensor_product_gpu_forward(
    const torch::Tensor X1,
    const torch::Tensor X2,
    const torch::Tensor mu_1,
    const torch::Tensor mu_2,
    const torch::Tensor mu_3,
    const torch::Tensor X3_ordering,
    const torch::Tensor cg_coefficients,
    const int output_size,
    const int nthreadx,
    const int nthready,
    const int nthreadz)
{

    CHECK_INPUT(X1);
    CHECK_INPUT(mu_1);
    CHECK_INPUT(X2);
    CHECK_INPUT(mu_2);
    CHECK_INPUT(cg_coefficients);
    CHECK_INPUT(mu_3);

    torch::Tensor output = torch::empty({X1.size(0), output_size, X1.size(2)},
                                        torch::TensorOptions()
                                            .dtype(X1.dtype())
                                            .device(X1.device()));

    const auto batch_sizex = X1.size(0);

    auto find_num_blocks = [](int x, int bdim)
    { return (x + bdim - 1) / bdim; };

    dim3 block_dim(batch_sizex);

    dim3 grid_dim(nthreadx, nthready, nthreadz);

    AT_DISPATCH_FLOATING_TYPES(
        X1.type(), "sparse_tensor_product_gpu_forward", ([&]
                                                         {
            size_t total_buff_size = 0;

            total_buff_size += 7 * mu_1.size(0) * sizeof(int16_t);
            total_buff_size += 2 * cg_coefficients.size(0) * sizeof(scalar_t);
            total_buff_size += grid_dim.x * X1.size(1) * sizeof(scalar_t);
            total_buff_size += X2.size(1) * X2.size(2) * sizeof(scalar_t);

            sparse_tensor_product_cuda_forward_kernel<scalar_t><<<block_dim, grid_dim, total_buff_size>>>(X1.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                                                                                    X2.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                                                                                    mu_1.packed_accessor32<int16_t, 1, torch::RestrictPtrTraits>(),
                                                                                    mu_2.packed_accessor32<int16_t, 1, torch::RestrictPtrTraits>(),
                                                                                    mu_3.packed_accessor32<int16_t, 1, torch::RestrictPtrTraits>(),
                                                                                    X3_ordering.packed_accessor32<int16_t, 1, torch::RestrictPtrTraits>(),
                                                                                    cg_coefficients.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                                                                    output.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>()
                                                                                    ); }));

    cudaDeviceSynchronize();

    return output;
}

template <typename scalar_t>
__global__ void sparse_tensor_product_backward_kernel(const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> X1,
                                                      const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> X2,
                                                      const torch::PackedTensorAccessor32<int16_t, 1, torch::RestrictPtrTraits> mu1,
                                                      const torch::PackedTensorAccessor32<int16_t, 1, torch::RestrictPtrTraits> mu2,
                                                      const torch::PackedTensorAccessor32<int16_t, 1, torch::RestrictPtrTraits> mu3,
                                                      const torch::PackedTensorAccessor32<int16_t, 1, torch::RestrictPtrTraits> X1_ordering,
                                                      const torch::PackedTensorAccessor32<int16_t, 1, torch::RestrictPtrTraits> X2_ordering,
                                                      const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> cg_coefficients,
                                                      const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> grad_out,
                                                      torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> grad_X1,
                                                      torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> grad_X2,
                                                      const bool requires_grad_X1,
                                                      const bool requires_grad_X2)
{

    extern __shared__ char buffer[];

    size_t offset = 0;

    scalar_t *buffer_x1 = reinterpret_cast<scalar_t *>(buffer + offset);
    offset += blockDim.x * X1.size(1) * sizeof(scalar_t);
    scalar_t *buffer_x2 = reinterpret_cast<scalar_t *>(buffer + offset);
    offset += X2.size(1) * X2.size(2) * sizeof(scalar_t);
    scalar_t *buffer_grad_out = reinterpret_cast<scalar_t *>(buffer + offset);
    offset += blockDim.x * grad_out.size(1) * sizeof(scalar_t);

    scalar_t *buffer_cg_coefficients = reinterpret_cast<scalar_t *>(buffer + offset);
    offset += cg_coefficients.size(0) * sizeof(scalar_t);
    scalar_t *buffer_cg_coefficients_X1 = reinterpret_cast<scalar_t *>(buffer + offset);
    offset += cg_coefficients.size(0) * sizeof(scalar_t);
    scalar_t *buffer_cg_coefficients_X2 = reinterpret_cast<scalar_t *>(buffer + offset);
    offset += cg_coefficients.size(0) * sizeof(scalar_t);

    scalar_t *buffer_grad_X2;
    if (requires_grad_X2)
    { // assume X2 has 1 channel
        buffer_grad_X2 = reinterpret_cast<scalar_t *>(buffer + offset);
        offset += X2.size(1) * sizeof(scalar_t);
    }

    int16_t *buffer_X1_ordering = reinterpret_cast<int16_t *>(buffer + offset);
    offset += X1_ordering.size(0) * sizeof(int16_t);
    int16_t *buffer_X2_ordering = reinterpret_cast<int16_t *>(buffer + offset);
    offset += X2_ordering.size(0) * sizeof(int16_t);

    int16_t *buffer_mu1 = reinterpret_cast<int16_t *>(buffer + offset);
    offset += mu1.size(0) * sizeof(int16_t);
    int16_t *buffer_mu2 = reinterpret_cast<int16_t *>(buffer + offset);
    offset += mu2.size(0) * sizeof(int16_t);
    int16_t *buffer_mu3 = reinterpret_cast<int16_t *>(buffer + offset);
    offset += mu3.size(0) * sizeof(int16_t);

    int16_t *buffer_mu1_X1 = reinterpret_cast<int16_t *>(buffer + offset);
    offset += mu1.size(0) * sizeof(int16_t);
    int16_t *buffer_mu1_X2 = reinterpret_cast<int16_t *>(buffer + offset);
    offset += mu1.size(0) * sizeof(int16_t);
    int16_t *buffer_mu2_X1 = reinterpret_cast<int16_t *>(buffer + offset);
    offset += mu2.size(0) * sizeof(int16_t);
    int16_t *buffer_mu2_X2 = reinterpret_cast<int16_t *>(buffer + offset);
    offset += mu2.size(0) * sizeof(int16_t);
    int16_t *buffer_mu3_X1 = reinterpret_cast<int16_t *>(buffer + offset);
    offset += mu3.size(0) * sizeof(int16_t);
    int16_t *buffer_mu3_X2 = reinterpret_cast<int16_t *>(buffer + offset);
    offset += mu3.size(0) * sizeof(int16_t);

    int atom_idx = blockIdx.x;

    // load all cg coefficients, mu indices into shared memory
    __syncthreads();

    for (int i = threadIdx.x; i < X1_ordering.size(0); i += blockDim.x)
    {
        buffer_X1_ordering[i] = X1_ordering[i];
        buffer_X2_ordering[i] = X2_ordering[i];
    }

    __syncthreads();

    for (int i = threadIdx.x; i < cg_coefficients.size(0); i += blockDim.x)
    {
        buffer_mu1[i] = mu1[i];
        buffer_mu2[i] = mu2[i];
        buffer_mu3[i] = mu3[i];
        buffer_cg_coefficients[i] = cg_coefficients[i];
    }
    __syncthreads();

    for (int i = threadIdx.x; i < X1_ordering.size(0); i += blockDim.x)
    {
        int16_t mu_1_i = X1_ordering[i];
        int16_t mu_2_i = X2_ordering[i];

        buffer_cg_coefficients_X1[i] = buffer_cg_coefficients[mu_1_i];
        buffer_cg_coefficients_X2[i] = buffer_cg_coefficients[mu_2_i];

        buffer_mu1_X1[i] = buffer_mu1[mu_1_i];
        buffer_mu1_X2[i] = buffer_mu1[mu_2_i];

        buffer_mu2_X1[i] = buffer_mu2[mu_1_i];
        buffer_mu2_X2[i] = buffer_mu2[mu_2_i];

        buffer_mu3_X1[i] = buffer_mu3[mu_1_i];
        buffer_mu3_X2[i] = buffer_mu3[mu_2_i];
    }

    __syncthreads();

    for (int X2_sph_id = threadIdx.x; X2_sph_id < X2.size(1); X2_sph_id += blockDim.x)
    {

        buffer_x2[X2_sph_id] = X2[atom_idx][X2_sph_id][0];

        if (requires_grad_X2)
        {
            buffer_grad_X2[X2_sph_id] = 0.0;
        }
    }

    __syncthreads();

    for (int channel_id = threadIdx.x; channel_id < X1.size(2); channel_id += blockDim.x)
    {
        __syncthreads();

        for (int X1_sph_id = 0; X1_sph_id < X1.size(1); X1_sph_id++)
        {
            buffer_x1[X1_sph_id * blockDim.x + threadIdx.x] = X1[atom_idx][X1_sph_id][channel_id];
        }

        for (int grad_sph_id = 0; grad_sph_id < grad_out.size(1); grad_sph_id++)
        {
            buffer_grad_out[grad_sph_id * blockDim.x + threadIdx.x] = grad_out[atom_idx][grad_sph_id][channel_id];
        }

        __syncthreads();

        int16_t prev_X1_index_mu1 = 0;
        int16_t X1_index_mu1 = 0;
        scalar_t sum_X1 = 0.0;

        int16_t prev_X2_index_mu2 = 0;
        int16_t X2_index_mu2 = 0;
        scalar_t sum_X2 = 0.0;

        for (int instruction_idx = 0; instruction_idx < mu1.size(0); instruction_idx++)
        {
            __syncthreads();

            X1_index_mu1 = buffer_mu1_X1[instruction_idx];
            int16_t X2_index_mu1 = buffer_mu2_X1[instruction_idx];
            int16_t X3_index_mu1 = buffer_mu3_X1[instruction_idx];
            scalar_t cg_coeff_mu1 = buffer_cg_coefficients_X1[instruction_idx];
            scalar_t grad_out_mu1 = buffer_grad_out[X3_index_mu1 * blockDim.x + threadIdx.x];
            scalar_t x2_mu1 = buffer_x2[X2_index_mu1];

            int16_t X1_index_mu2 = buffer_mu1_X2[instruction_idx];
            X2_index_mu2 = buffer_mu2_X2[instruction_idx];
            int16_t X3_index_mu2 = buffer_mu3_X2[instruction_idx];
            scalar_t cg_coeff_mu2 = buffer_cg_coefficients_X2[instruction_idx];
            scalar_t grad_out_mu2 = buffer_grad_out[X3_index_mu2 * blockDim.x + threadIdx.x];
            scalar_t x1_mu2 = buffer_x1[X1_index_mu2 * blockDim.x + threadIdx.x];

            __syncthreads();

            if (requires_grad_X1 && prev_X1_index_mu1 != X1_index_mu1)
            {
                grad_X1[atom_idx][prev_X1_index_mu1][channel_id] = sum_X1;
                sum_X1 = 0.0;
                prev_X1_index_mu1 = X1_index_mu1;
            }

            if (requires_grad_X2 && prev_X2_index_mu2 != X2_index_mu2)
            {
                for (int offset = 16; offset > 0; offset /= 2)
                    sum_X2 += __shfl_down_sync(FULL_MASK, sum_X2, offset);

                if (threadIdx.x % 32 == 0)
                {
                    atomicAdd(&buffer_grad_X2[prev_X2_index_mu2], sum_X2);
                }

                sum_X2 = 0.0;
                prev_X2_index_mu2 = X2_index_mu2;
            }

            if (requires_grad_X1)
                sum_X1 += cg_coeff_mu1 * x2_mu1 * grad_out_mu1;

            if (requires_grad_X2)
                sum_X2 += cg_coeff_mu2 * x1_mu2 * grad_out_mu2;
        }

        __syncthreads();

        if (requires_grad_X1)
            grad_X1[atom_idx][X1_index_mu1][channel_id] = sum_X1;

        if (requires_grad_X2)
        {
            for (int offset = 16; offset > 0; offset /= 2)
                sum_X2 += __shfl_down_sync(FULL_MASK, sum_X2, offset);

            if (threadIdx.x % 32 == 0)
            {
                atomicAdd(&buffer_grad_X2[X2_index_mu2], sum_X2);
            }
        }

        __syncthreads();
    }

    if (requires_grad_X2)
    {
        for (int X2_sph_id = threadIdx.x; X2_sph_id < grad_X2.size(1); X2_sph_id += blockDim.x)
        {
            grad_X2[atom_idx][X2_sph_id][0] = buffer_grad_X2[X2_sph_id];
        }
    }
}

std::vector<torch::Tensor> sparse_tensor_product_gpu_backward(
    const torch::Tensor X1,
    const torch::Tensor X2,
    const torch::Tensor mu1,
    const torch::Tensor mu2,
    const torch::Tensor mu3,
    const torch::Tensor X1_ordering,
    const torch::Tensor X2_ordering,
    const torch::Tensor cg_coefficients,
    const torch::Tensor grad_output,
    const int nthreadx,
    const int nthready,
    const int nthreadz)
{

    torch::Tensor grad_X1;
    torch::Tensor grad_X2;

    if (X1.requires_grad())
    {
        grad_X1 = torch::empty_like(X1,
                                    torch::TensorOptions()
                                        .dtype(X1.dtype())
                                        .device(X1.device()));
    }
    else
    {
        grad_X1 = torch::empty({1, 1, 1},
                               torch::TensorOptions()
                                   .dtype(X1.dtype())
                                   .device(X1.device()));
    }

    if (X2.requires_grad())
    {
        // can reduce over natoms in the backwards pass
        grad_X2 = torch::empty_like(X2,
                                    torch::TensorOptions()
                                        .dtype(X2.dtype())
                                        .device(X2.device()));
    }
    else
    {
        grad_X2 = torch::empty({1, 1, 1},
                               torch::TensorOptions()
                                   .dtype(X2.dtype())
                                   .device(X2.device()));
    }

    const auto batch_sizex = X1.size(0);

    auto find_num_blocks = [](int x, int bdim)
    { return (x + bdim - 1) / bdim; };

    // int nbx = find_num_blocks(batch_sizex, nthreadz);
    dim3 block_dim(X1.size(0));

    dim3 grid_dim(nthreadx, nthready, nthreadz);

    AT_DISPATCH_FLOATING_TYPES(
        X1.type(), "sparse_tensor_product_gpu_backward", ([&]
                                                          {

            size_t total_buff_size = 0;

            total_buff_size += 11 * mu1.size(0) * sizeof(int16_t);
            total_buff_size += 3 * cg_coefficients.size(0) * sizeof(scalar_t);

            total_buff_size += grid_dim.x * X1.size(1) * sizeof(scalar_t);
            total_buff_size += X2.size(1) * X2.size(2) * sizeof(scalar_t);
            total_buff_size += grid_dim.x * grad_output.size(1) * sizeof(scalar_t);

            if (X2.requires_grad()) { // assume X2 has 1 channel
                total_buff_size += X2.size(1) * sizeof(scalar_t);
            }

            sparse_tensor_product_backward_kernel<scalar_t><<<block_dim, grid_dim, total_buff_size>>>(
                                                                                    X1.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                                                                                    X2.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                                                                                    mu1.packed_accessor32<int16_t, 1, torch::RestrictPtrTraits>(),
                                                                                    mu2.packed_accessor32<int16_t, 1, torch::RestrictPtrTraits>(),
                                                                                    mu3.packed_accessor32<int16_t, 1, torch::RestrictPtrTraits>(),
                                                                                    X1_ordering.packed_accessor32<int16_t, 1, torch::RestrictPtrTraits>(),
                                                                                    X2_ordering.packed_accessor32<int16_t, 1, torch::RestrictPtrTraits>(),
                                                                                    cg_coefficients.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                                                                    grad_output.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                                                                                    grad_X1.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                                                                                    grad_X2.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                                                                                    X1.requires_grad(),
                                                                                    X2.requires_grad()
                                                                                    ); }));

    cudaDeviceSynchronize();

    return {grad_X1, grad_X2};
}

class TensorProductAutograd : public Function<TensorProductAutograd>
{
public:
    static torch::Tensor forward(
        AutogradContext *ctx,
        torch::Tensor X1,
        torch::Tensor X2,
        const torch::Tensor mu1,
        const torch::Tensor mu2,
        const torch::Tensor mu3,
        const torch::Tensor X1_ordering,
        const torch::Tensor X2_ordering,
        const torch::Tensor X3_ordering,
        const torch::Tensor cg_coefficients,
        const int64_t output_size,
        const int64_t nthreadx,
        const int64_t nthready,
        const int64_t nthreadz)
    {

        auto result = sparse_tensor_product_gpu_forward(X1, X2, mu1, mu2, mu3, X3_ordering, cg_coefficients, output_size, nthreadx, nthready, nthreadz);

        if (X1.requires_grad() || X2.requires_grad())
        {
            ctx->saved_data["nthreadx"] = nthreadx;
            ctx->saved_data["nthready"] = nthready;
            ctx->saved_data["nthreadz"] = nthreadz;

            ctx->save_for_backward({X1, X2, mu1, mu2, mu3, X1_ordering, X2_ordering, cg_coefficients});
        }

        return result;
    }

    static variable_list backward(AutogradContext *ctx, variable_list grad_outputs)
    {

        auto saved_variables = ctx->get_saved_variables();

        auto X1 = saved_variables[0];
        auto X2 = saved_variables[1];
        auto mu1 = saved_variables[2];
        auto mu2 = saved_variables[3];
        auto mu3 = saved_variables[4];
        auto X1_ordering = saved_variables[5];
        auto X2_ordering = saved_variables[6];
        auto cg_coefficients = saved_variables[7];
        int nthreadx = ctx->saved_data["nthreadx"].toInt();
        int nthready = ctx->saved_data["nthready"].toInt();
        int nthreadz = ctx->saved_data["nthreadz"].toInt();

        auto result = sparse_tensor_product_gpu_backward(X1, X2,
                                                         mu1, mu2, mu3, X1_ordering, X2_ordering, cg_coefficients, grad_outputs[0], nthreadx, nthready, nthreadz);

        torch::Tensor undef;

        return {result[0], result[1], undef, undef, undef, undef, undef, undef, undef, undef, undef, undef, undef};
    }
};

torch::Tensor tensor_product(torch::Tensor X1, torch::Tensor X2,
                             torch::Tensor mu1, torch::Tensor mu2, torch::Tensor mu3,
                             torch::Tensor X1_ordering, torch::Tensor X2_ordering, torch::Tensor X3_ordering,
                             torch::Tensor cg_coefficients,
                             int64_t output_size, int64_t nthreadx, int64_t nthready, int64_t nthreadz)
{

    return TensorProductAutograd::apply(X1, X2,
                                        mu1, mu2, mu3,
                                        X1_ordering, X2_ordering, X3_ordering,
                                        cg_coefficients,
                                        output_size, nthreadx, nthready, nthreadz);
}

TORCH_LIBRARY(mace_cuda_tp, m)
{
    m.def("tensor_product", &tensor_product);
}
