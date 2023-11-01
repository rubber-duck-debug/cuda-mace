#include <mma.h>
#include <cuda.h>
#include <torch/script.h>

#include <stdio.h>
#include <stdlib.h>
#include <cuda/barrier>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_fp16.h>

using namespace nvcuda;
using namespace std;
using namespace torch::autograd;

#define WARP_SIZE 32

// MMA matrix tile dimensions.
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

__host__ __device__ int32_t find_integer_divisor(int32_t x, int32_t bdim)
{
    return (x + bdim - 1) / bdim;
}

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

template <class T>
__host__ __device__ T *align_array(std::size_t n_elements, void *&ptr, const std::size_t alignment,
                                   std::size_t *space = nullptr) noexcept
{
    // const std::size_t alignment = alignof(T);
    const std::uintptr_t intptr = reinterpret_cast<uintptr_t>(ptr);
    const std::uintptr_t aligned = (intptr + alignment - 1) & -alignment;
    const std::uintptr_t end = aligned + n_elements * sizeof(T);
    if (space)
        *space += static_cast<std::size_t>(end - intptr);
    ptr = reinterpret_cast<void *>(end);
    return reinterpret_cast<T *>(aligned);
}

__global__ void very_dumb_matmul(const float *X, const float *W, float *OUT, const int NNODES, const int M_TOTAL, const int N_TOTAL, const int K_TOTAL)
{

    // load X into shared memory, but column-major
    // load 32 by 32 chunk of W into shared memory

    extern __shared__ char buffer[];

    void *sptr = buffer;
    size_t space = 0;

    float *buffer_X = shared_array<float>(K_TOTAL * M_TOTAL, sptr, &space);
    float *buffer_out = shared_array<float>(N_TOTAL * M_TOTAL, sptr, &space);

    for (int i = 0; i < find_integer_divisor(M_TOTAL, blockDim.y); i++)
    {
        for (int j = 0; j < find_integer_divisor(K_TOTAL, blockDim.x); j++)
        {
            buffer_X[(j * blockDim.x + threadIdx.x) * M_TOTAL + (i * blockDim.y + threadIdx.y)] = X[blockIdx.x * M_TOTAL * K_TOTAL + (i * blockDim.y + threadIdx.y) * K_TOTAL + (j * blockDim.x + threadIdx.x)];
        }
    }
    __syncthreads();

    for (int i = threadIdx.y; i < M_TOTAL; i += blockDim.y)
    {
        for (int j = threadIdx.x; j < N_TOTAL; j += blockDim.x)
        {

            float output = 0.0;

            for (int k = 0; k < K_TOTAL; k++)
            {
                output += buffer_X[k * M_TOTAL + i] * W[k * N_TOTAL + j];
            }

            buffer_out[i * N_TOTAL + j] = output;
        }
    }

    __syncthreads();

    for (int tid = threadIdx.y * blockDim.x + threadIdx.x; tid < M_TOTAL * N_TOTAL; tid += blockDim.x * blockDim.y)
        OUT[blockIdx.x * M_TOTAL * N_TOTAL + tid] = buffer_out[tid];
}

__global__ void matmul_wmma_kernel(float *X, float *W, float *OUT, const int NNODES, const int M_TOTAL, const int N_TOTAL, const int K_TOTAL)
{

    extern __shared__ char buffer[];

    void *sptr = buffer;
    size_t space = 0;

    half *buffer_X = shared_array<half>(K_TOTAL * M_TOTAL, sptr, &space);
    half *buffer_delta_X = shared_array<half>(K_TOTAL * M_TOTAL, sptr, &space);
    half *buffer_W = shared_array<half>(WMMA_K * N_TOTAL, sptr, &space);
    half *buffer_delta_W = shared_array<half>(WMMA_K * N_TOTAL, sptr, &space);

    for (int i = 0; i < find_integer_divisor(M_TOTAL, blockDim.y); i++)
    {
        for (int j = 0; j < find_integer_divisor(K_TOTAL, blockDim.x); j++)
        {
            float x = X[blockIdx.x * M_TOTAL * K_TOTAL + (i * blockDim.y + threadIdx.y) * K_TOTAL + (j * blockDim.x + threadIdx.x)];

            half x_h = __float2half(x);

            buffer_X[(j * blockDim.x + threadIdx.x) * M_TOTAL + (i * blockDim.y + threadIdx.y)] = x_h;
            buffer_delta_X[(j * blockDim.x + threadIdx.x) * M_TOTAL + (i * blockDim.y + threadIdx.y)] = __float2half(x - __half2float(x_h));
        }
    }

    __syncthreads();

    int lda = K_TOTAL;
    int ldb = N_TOTAL;
    int ldc = N_TOTAL;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> a_frag, delta_a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag, delta_b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> ab_frag;

    wmma::fill_fragment(ab_frag, 0.0f);

    int warpM = threadIdx.x / warpSize;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

    for (int i = 0; i < K_TOTAL; i += WMMA_K)
    {

        // need to load tiles of W in here... need threadIdx.y * WMMA_N x WMMA_K tiles...
        for (int j = threadIdx.y; j < WMMA_K; j += blockDim.y)
        {
            for (int k = threadIdx.x; k < N_TOTAL; k += blockDim.x)
            {
                float w = W[(i + j) * N_TOTAL + k];
                half w_h = __float2half(w);

                buffer_W[j * N_TOTAL + k] = __float2half(w);
                buffer_delta_W[j * N_TOTAL + k] = __float2half(w - __half2float(w_h));
            }
        }

        // Now need to compute C_{32} = A_{16}B_{16} + \Delta A_{16} B_{16} + A_{16}\Delta B_{16} + \Delta A_{16}\Delta B_{16}
        __syncthreads();

        int aCol = i;
        int aRow = warpM * WMMA_M;
        int bCol = warpN * WMMA_N;
        int bRow = i;

        if (aRow < M_TOTAL && aCol < K_TOTAL && bRow < K_TOTAL && bCol < N_TOTAL)
        {
            wmma::load_matrix_sync(a_frag, buffer_X + aCol * M_TOTAL + aRow, M_TOTAL);
            wmma::load_matrix_sync(delta_a_frag, buffer_delta_X + aCol * M_TOTAL + aRow, M_TOTAL);

            // wmma::load_matrix_sync(b_frag, buffer_W + bCol + bRow * ldb, ldb);
            // wmma::load_matrix_sync(delta_b_frag, buffer_W + bCol + bRow * ldb, ldb);

            wmma::load_matrix_sync(b_frag, buffer_W + bCol, ldb);
            wmma::load_matrix_sync(delta_b_frag, buffer_delta_W + bCol, ldb);

            // Perform the matrix multiplication
            wmma::mma_sync(ab_frag, a_frag, b_frag, ab_frag);
            wmma::mma_sync(ab_frag, delta_a_frag, b_frag, ab_frag);
            wmma::mma_sync(ab_frag, a_frag, delta_b_frag, ab_frag);
            wmma::mma_sync(ab_frag, delta_a_frag, delta_b_frag, ab_frag);
        }
    }

    __syncthreads();

    int cCol = warpN * WMMA_N;
    int cRow = warpM * WMMA_M;

    if (cRow < M_TOTAL && cCol < N_TOTAL)
    {
        wmma::store_matrix_sync(OUT + blockIdx.x * M_TOTAL * N_TOTAL + cCol + cRow * ldc, ab_frag, ldc, wmma::mem_row_major);
    }
}

void deleter(void *arg){};

torch::Tensor matmul_wmma(torch::Tensor X, torch::Tensor W)
{
    const int NNODES = X.size(0);
    const int M = X.size(1);
    const int N = W.size(1);
    const int K = W.size(0);

    TORCH_CHECK(X.device().is_cuda(), "X must be a CUDA tensor");
    TORCH_CHECK(W.device().is_cuda(), "W must be a CUDA tensor");

    TORCH_CHECK(M == 16, "X dim=1 must have dimension 16 [(lmax +1)**2]");
    TORCH_CHECK(N % 16 == 0, "W dim=2 must be a multiple of 16");
    TORCH_CHECK(K % 16 == 0, "X dim=2 must be a multiple of 16");

    // float *output_ptr = NULL;

    // cudaMalloc(reinterpret_cast<void **>(&output_ptr),          sizeof(float) * M * N);

    torch::Tensor output = torch::empty({NNODES, M, N},
                                        torch::TensorOptions()
                                            .dtype(X.dtype())
                                            .device(X.device()));

    dim3 gridDim, blockDim;

    blockDim.x = WARP_SIZE;
    blockDim.y = 8;

    gridDim.x = NNODES;
    gridDim.y = 1; // find_integer_divisor(N, blockDim.y * WMMA_N);

    size_t shared_size = 0;
    void *sptr = nullptr;

    shared_array<half>(K * M, sptr, &shared_size);
    shared_array<half>(K * M, sptr, &shared_size);
    shared_array<half>(WMMA_K * N, sptr, &shared_size);
    shared_array<half>(WMMA_K * N, sptr, &shared_size);

    assert(((unsigned long long)X.data_ptr<float>()) % 128 == 0);
    assert(((unsigned long long)W.data_ptr<float>()) % 128 == 0);
    assert(((unsigned long long)output.data_ptr<float>()) % 128 == 0);

    // printf("launching wmma kernel...\n");
    // printf("grid dim.x: %d grid dim.y: %d grid dim.z: %d\n", gridDim.x, gridDim.y, gridDim.z);
    // printf("block dim.x: %d block dim.y: %d block dim.z: %d\n", blockDim.x, blockDim.y, blockDim.z);

    matmul_wmma_kernel<<<gridDim, blockDim, shared_size>>>(X.data_ptr<float>(), W.data_ptr<float>(), output.data_ptr<float>(),
                                                           NNODES, M, N, K);

    cudaDeviceSynchronize();

    // torch::Tensor output = torch::from_blob(output_ptr, {NNODES, M, N}, deleter, torch::TensorOptions().dtype(X.dtype()).device(X.device()));

    return output;
}

class MatmulAutograd : public Function<MatmulAutograd>
{
public:
    static torch::Tensor forward(
        AutogradContext *ctx,
        torch::Tensor X,
        torch::Tensor W,
        torch::Tensor W_transposed)
    {
        if (X.requires_grad())
        {
            ctx->save_for_backward({W_transposed});
        }

        torch::Tensor result = matmul_wmma(X, W);

        return result;
    }

    static variable_list backward(AutogradContext *ctx, variable_list grad_outputs)
    {
        auto saved_variables = ctx->get_saved_variables();

        auto W_T = saved_variables[0];

        torch::Tensor dX = matmul_wmma(grad_outputs[0].contiguous(), W_T);

        torch::Tensor undef;

        return {dX, undef, undef};
    }
};

torch::Tensor matmul(
    torch::Tensor X,
    torch::Tensor W,
    torch::Tensor W_T)
{
    return MatmulAutograd::apply(X, W, W_T);
}

__global__ void linear_wmma_kernel(
    const float *__restrict__ X,
    const float *__restrict__ W,
    float *OUT,
    const int *__restrict__ l_start,
    const int *__restrict__ l_end,
    const float *__restrict__ path_weights,
    const int ninstructions,
    const int NNODES,
    const int M_TOTAL,
    const int N_TOTAL,
    const int K_TOTAL)
{

    extern __shared__ char buffer[];

    void *sptr = buffer;
    size_t space = 0;

    float *buffer_X = shared_array<float>(K_TOTAL * M_TOTAL, sptr, &space);
    float *buffer_tmp_output = shared_array<float>(M_TOTAL * N_TOTAL, sptr, &space);

    // const float *X_i = X + blockIdx.x * M_TOTAL * K_TOTAL;
    // float *OUT_i = OUT + blockIdx.x * M_TOTAL * N_TOTAL;

    for (int i = 0; i < M_TOTAL / blockDim.y; i++)
    {
        for (int j = 0; j < K_TOTAL / blockDim.x; j++)
        {

            buffer_X[(j * blockDim.x + threadIdx.x) * M_TOTAL + (i * blockDim.y + threadIdx.y)] = X[blockIdx.x * M_TOTAL * K_TOTAL + (i * blockDim.y + threadIdx.y) * K_TOTAL + (j * blockDim.x + threadIdx.x)];
        }
    }

    __syncthreads();

    int a_row = 0;
    int b_col = (blockIdx.y * blockDim.y + threadIdx.y) * WMMA_N;

    for (int instruction = 0; instruction < ninstructions; instruction++)
    {
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, 8, wmma::precision::tf32, wmma::col_major> a_frag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, 8, wmma::precision::tf32, wmma::row_major> b_frag;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, 8, float> ab_frag;

        int lstart = l_start[instruction];
        int lend = l_end[instruction];
        float pathw = path_weights[instruction];

        wmma::fill_fragment(ab_frag, 0.0f);

        for (int k = 0; k < K_TOTAL; k += 8)
        {

            wmma::load_matrix_sync(a_frag, buffer_X + k * M_TOTAL + a_row, M_TOTAL);
            wmma::load_matrix_sync(b_frag, W + (instruction * K_TOTAL * N_TOTAL) + b_col + k * N_TOTAL, N_TOTAL);

            for (int i = 0; i < a_frag.num_elements; i++)
            {
                // a_frag.x[i] = wmma::__float_to_tf32(a_frag.x[i]);
            }

            for (int i = 0; i < b_frag.num_elements; i++)
            {
                // b_frag.x[i] = wmma::__float_to_tf32(b_frag.x[i]);
            }

            // Perform the matrix multiplication
            wmma::mma_sync(ab_frag, a_frag, b_frag, ab_frag);
        }

        // apply path weight
        for (int i = 0; i < ab_frag.num_elements; i++)
        {
            ab_frag.x[i] = ab_frag.x[i] * pathw;
        }

        wmma::store_matrix_sync(buffer_tmp_output + b_col + a_row * N_TOTAL, ab_frag, N_TOTAL, wmma::mem_row_major);

        // wait for output to be fully populated...
        __syncthreads();

        // write out the part of the matmul that we need.
        for (int lm = lstart + threadIdx.y; lm < lend; lm += blockDim.y)
        {
            for (int channel = threadIdx.x; channel < N_TOTAL; channel += blockDim.x)
            {
                OUT[blockIdx.x * M_TOTAL * N_TOTAL + lm * N_TOTAL + channel] = buffer_tmp_output[lm * N_TOTAL + channel];
            }
        }
    }
}

torch::Tensor linear_(
    torch::Tensor X,
    torch::Tensor W,
    torch::Tensor l_start,
    torch::Tensor l_end,
    torch::Tensor path_weights,
    bool print_debug = false)
{

    const int NNODES = X.size(0);
    const int M = X.size(1);
    const int ninstructions = W.size(0);
    const int N = W.size(2);
    const int K = W.size(1);

    TORCH_CHECK(X.device().is_cuda(), "X must be a CUDA tensor");
    TORCH_CHECK(W.device().is_cuda(), "W must be a CUDA tensor");
    TORCH_CHECK(l_start.device().is_cuda(), "l_start must be a CUDA tensor");
    TORCH_CHECK(l_end.device().is_cuda(), "l_end must be a CUDA tensor");

    TORCH_CHECK(l_start.size(0) == l_end.size(0) && l_start.size(0) == W.size(0), "l_start/end must be same size as first dimension of W");

    TORCH_CHECK(M == 16, "X dim=1 must have dimension 16 [(lmax +1)**2]");
    TORCH_CHECK(N % 16 == 0, "W dim=2 must be a multiple of 16");
    TORCH_CHECK(K % 16 == 0, "X dim=2 must be a multiple of 16");

    torch::Tensor output = torch::empty({NNODES, M, N},
                                        torch::TensorOptions()
                                            .dtype(X.dtype())
                                            .device(X.device()));

    dim3 gridDim, blockDim;
    blockDim.x = WARP_SIZE;
    blockDim.y = 4; // 8 * WMMA_N = 64

    gridDim.x = NNODES;
    gridDim.y = find_integer_divisor(N, blockDim.y * WMMA_N);

    std::cout << "grid dim: " << gridDim.x << " " << gridDim.y << " " << gridDim.z << std::endl;
    std::cout << "block dim: " << blockDim.x << " " << blockDim.y << std::endl;

    size_t shared_size = 0;
    void *sptr = nullptr;

    shared_array<float>(M * K, sptr, &shared_size); // X
    shared_array<float>(M * N, sptr, &shared_size); // tmp_output

    linear_wmma_kernel<<<gridDim, blockDim, shared_size>>>(X.data_ptr<float>(), W.data_ptr<float>(), output.data_ptr<float>(),
                                                           l_start.data_ptr<int>(),
                                                           l_end.data_ptr<int>(),
                                                           path_weights.data_ptr<float>(),
                                                           ninstructions,
                                                           NNODES, M, N, K);

    cudaDeviceSynchronize();

    return output;
}

class LinearAutograd : public Function<LinearAutograd>
{
public:
    static torch::Tensor forward(
        AutogradContext *ctx,
        torch::Tensor X,
        torch::Tensor W,
        torch::Tensor W_transposed, // needed for backwards pass dL/dX
        torch::Tensor l_start,
        torch::Tensor l_end,
        torch::Tensor path_weights)
    {

        if (X.requires_grad())
        {
            ctx->save_for_backward({W_transposed, l_start, l_end, path_weights});
        }

        torch::Tensor result = linear_(X, W, l_start, l_end, path_weights);

        return result;
    }

    static variable_list backward(AutogradContext *ctx, variable_list grad_outputs)
    {
        auto saved_variables = ctx->get_saved_variables();

        auto W_T = saved_variables[0];
        auto l_start = saved_variables[1];
        auto l_end = saved_variables[2];
        auto path_weights = saved_variables[3];

        torch::Tensor dX = linear_(grad_outputs[0], W_T, l_start, l_end, path_weights);

        torch::Tensor dW;

        // if (W.requires_grad())
        //{
        //  for i  in range(x.shape[0]) : grad_w += torch.matmul(x[i].transpose(-1, -2), grad_output[i])
        // dW = torch::bmm(X.transpose(-1, -2).contiguous(), grad_outputs[0]).sum(0);
        //}

        torch::Tensor undef;

        return {dX, undef, undef, undef, undef, undef};
    }
};

torch::Tensor linear(
    torch::Tensor X,
    torch::Tensor W,
    torch::Tensor W_T,
    torch::Tensor l_start,
    torch::Tensor l_end,
    torch::Tensor path_weights)
{
    return LinearAutograd::apply(X, W, W_T, l_start, l_end, path_weights);
}

TORCH_LIBRARY(linear_wmma, m)
{
    m.def("linear", &linear);
    m.def("matmul", &matmul);
}
