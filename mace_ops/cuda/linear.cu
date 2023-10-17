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

template <typename scalar_t>
__global__ void forward_kernel(
    const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> X,
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> weights,
    const torch::PackedTensorAccessor64<int32_t, 1, torch::RestrictPtrTraits> weight_indices,
    const torch::PackedTensorAccessor64<int32_t, 1, torch::RestrictPtrTraits> output_indices,
    const int64_t noutputs,
    torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> output)
{

    extern __shared__ char buffer[];

    void *sptr = buffer;
    size_t space = 0;

    scalar_t *buffer_output = shared_array<scalar_t>(noutputs * blockDim.x, sptr, &space);

    int feat = blockIdx.y * blockDim.x + threadIdx.x;

    for (int lm = threadIdx.y; lm < noutputs; lm += blockDim.y)
    {
        buffer_output[lm * blockDim.x + threadIdx.x] = 0.0;
    }

    __syncthreads();

    for (int lm = threadIdx.y; lm < X.size(1); lm += blockDim.y)
    {
        int weight_index = weight_indices[lm];
        int output_index = output_indices[lm];

        atomicAdd(&buffer_output[output_index * blockDim.x + threadIdx.x], X[blockIdx.x][lm][feat] * weights[weight_index][feat]);
    }

    __syncthreads();

    for (int lm = threadIdx.y; lm < noutputs; lm += blockDim.y)
    {
        output[blockIdx.x][lm][feat] = buffer_output[lm * blockDim.x + threadIdx.x];
    }
}

torch::Tensor forward_gpu(
    torch::Tensor X,
    torch::Tensor weights,
    torch::Tensor weight_indices,
    torch::Tensor output_indices,
    int64_t noutputs,
    int64_t nthreadx,
    int64_t nthready,
    int64_t nthreadz)
{

    torch::Tensor output = torch::empty({X.size(0), noutputs, X.size(2)},
                                        torch::TensorOptions()
                                            .dtype(X.dtype())
                                            .device(X.device()));

    int32_t nby = find_integer_divisor(X.size(2), nthreadx);

    dim3 block_dim(X.size(0), nby);

    dim3 grid_dim(nthreadx, nthready, 1);

    AT_DISPATCH_FLOATING_TYPES(
        X.type(), "forward_gpu", ([&]
                                  {

                    size_t space = 0;
                    void* sptr = nullptr;

                     shared_array<scalar_t>(noutputs * grid_dim.x, sptr, &space);

                    forward_kernel<scalar_t><<<block_dim, grid_dim, space>>>(
                        X.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                        weights.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                        weight_indices.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>(),
                        output_indices.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>(),
                        noutputs,
                        output.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>()); }));

    cudaDeviceSynchronize();

    return output;
}

template <typename scalar_t>
__global__ void backward_kernel(
    const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> X,
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> weights,
    const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> grad_in,
    const torch::PackedTensorAccessor64<int32_t, 1, torch::RestrictPtrTraits> weight_indices,
    const torch::PackedTensorAccessor64<int32_t, 1, torch::RestrictPtrTraits> output_indices,
    torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> gradX)
{
    extern __shared__ char buffer[];

    void *sptr = buffer;
    size_t space = 0;

    scalar_t *buffer_gradX = shared_array<scalar_t>(X.size(1) * blockDim.x, sptr, &space);

    int feat = blockIdx.y * blockDim.x + threadIdx.x;

    for (int lm = threadIdx.y; lm < X.size(1); lm += blockDim.y)
    {
        buffer_gradX[lm * blockDim.x + threadIdx.x] = 0.0;
    }

    __syncthreads();

    for (int lm = threadIdx.y; lm < X.size(1); lm += blockDim.y)
    {
        int weight_index = weight_indices[lm];
        int output_index = output_indices[lm];

        atomicAdd(&buffer_gradX[lm * blockDim.x + threadIdx.x], grad_in[blockIdx.x][output_index][feat] * weights[weight_index][feat]);
    }

    __syncthreads();

    for (int lm = threadIdx.y; lm < X.size(1); lm += blockDim.y)
    {
        gradX[blockIdx.x][lm][feat] = buffer_gradX[lm * blockDim.x + threadIdx.x];
    }
}

torch::Tensor backward_gpu(
    torch::Tensor X,
    torch::Tensor weights,
    torch::Tensor grad_in,
    torch::Tensor weight_indices,
    torch::Tensor output_indices,
    int64_t nthreadx,
    int64_t nthready,
    int64_t nthreadz)
{

    torch::Tensor gradX = torch::empty_like(X, torch::TensorOptions()
                                                   .dtype(X.dtype())
                                                   .device(X.device()));

    int32_t nby = find_integer_divisor(X.size(2), nthreadx);

    dim3 block_dim(X.size(0), nby);

    dim3 grid_dim(nthreadx, nthready, 1);

    AT_DISPATCH_FLOATING_TYPES(
        X.type(), "backward_gpu", ([&]
                                   {
                                       size_t space = 0;
                                       void *sptr = nullptr;

                                       shared_array<scalar_t>(X.size(1) * grid_dim.x, sptr, &space);

                                       /*backward_kernel<scalar_t><<<block_dim, grid_dim, space>>>(
                                           X.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                                           weights.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                                           grad_in.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                                           weight_indices.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>(),
                                           output_indices.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>(),
                                           gradX.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>());  */

                                       backward_kernel<scalar_t><<<block_dim, grid_dim, space>>>(
                                           X.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                                           weights.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                                           grad_in.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                                           weight_indices.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>(),
                                           output_indices.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>(),
                                           gradX.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>()); }));

    cudaDeviceSynchronize();

    return gradX;
}

class LinearAutograd : public Function<LinearAutograd>
{
public:
    static torch::Tensor forward(
        AutogradContext *ctx,
        torch::Tensor X,
        torch::Tensor weights,
        torch::Tensor weight_indices,
        torch::Tensor output_indices,
        int64_t noutputs,
        int64_t nthreadx,
        int64_t nthready,
        int64_t nthreadz)
    {

        if (X.requires_grad())
        {
            ctx->save_for_backward({X, weights, weight_indices, output_indices});

            ctx->saved_data["nthreadx"] = nthreadx;
            ctx->saved_data["nthready"] = nthready;
            ctx->saved_data["nthreadz"] = nthreadz;
        }

        torch::Tensor result = forward_gpu(X, weights, weight_indices, output_indices, noutputs, nthreadx, nthready, nthreadz);

        return result;
    }

    static variable_list backward(AutogradContext *ctx, variable_list grad_outputs)
    {
        auto saved_variables = ctx->get_saved_variables();

        auto X = saved_variables[0];
        auto weights = saved_variables[1];
        auto weight_indices = saved_variables[2];
        auto output_indices = saved_variables[3];

        int64_t nthreadx = ctx->saved_data["nthreadx"].toInt();
        int64_t nthready = ctx->saved_data["nthready"].toInt();
        int64_t nthreadz = ctx->saved_data["nthreadz"].toInt();

        torch::Tensor result = backward_gpu(X, weights, grad_outputs[0], weight_indices, output_indices, nthreadx, nthready, nthreadz);

        torch::Tensor undef;

        return {result, undef, undef, undef, undef, undef, undef, undef};
    }
};

torch::Tensor linear(
    torch::Tensor X,
    torch::Tensor weights,
    torch::Tensor weight_indices,
    torch::Tensor output_indices,
    int64_t noutputs,
    int64_t nthreadx,
    int64_t nthready,
    int64_t nthreadz)
{
    return LinearAutograd::apply(X, weights, weight_indices, output_indices, noutputs, nthreadx, nthready, nthreadz);
}

TORCH_LIBRARY(linear, m)
{
    m.def("forward", &linear);
    m.def("forward_only", &forward_gpu);
}
