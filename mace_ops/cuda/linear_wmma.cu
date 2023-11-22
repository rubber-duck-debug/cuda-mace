#include <mma.h>
#include <cuda.h>
#include <torch/script.h>

#include <stdio.h>
#include <stdlib.h>
#include <cuda/barrier>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_fp16.h>
#include <cuda_pipeline_primitives.h>
#include <cooperative_groups.h>
#include <cuda/pipeline>

using namespace cooperative_groups;
using namespace nvcuda;
using namespace std;
using namespace torch::autograd;
using barrier = cuda::barrier<cuda::thread_scope_block>;

#define WARP_SIZE 32

// MMA matrix tile dimensions.
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 8

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
#define M_BATCH 16
#define N_BATCH 32
#define K_BATCH 32

__global__ void matmul_kernel(float *X, float *W, float *OUT, const int NNODES, const int M, const int N, const int K)
{

    extern __shared__ char buffer[];

    void *sptr = buffer;
    size_t space = 0;

    float *buffer_X = shared_array<float>((K_BATCH + 1) * K_BATCH, sptr, &space); // use double the space for X for now just to deal with bank conflicts
    float *buffer_W = shared_array<float>(K_BATCH * N_BATCH, sptr, &space);

    // define some registers to help increase arithmetic intensity
    float output_reg[4] = {0.0, 0.0, 0.0, 0.0};

    int nk_iter = find_integer_divisor(K, K_BATCH);

    for (int k_id = 0; k_id < nk_iter; k_id++)
    {

        int kstart = k_id * K_BATCH;

        // 0 * (16+1)   = 0         |  0 * (32 + 1)     = 0  % 32 = 0
        // 1            = 17        |  1 * (32 + 1)     = 33 % 32 = 1
        // 2            = 34        |  2 * (32 + 1)     = 66 % 32 = 2
        // 3
        // load 16x32 tile of X
        for (int m = threadIdx.y; m < M_BATCH; m += blockDim.y)
        {
            buffer_X[threadIdx.x * (K_BATCH + 1) + m] = X[blockIdx.x * M * K + m * K + kstart + threadIdx.x];
        }

        // load 32x32 tile of W
        for (int k = threadIdx.y; k < K_BATCH; k += blockDim.y)
        {
            buffer_W[k * N_BATCH + threadIdx.x] = W[(kstart + k) * N + (blockIdx.y * N_BATCH) + threadIdx.x];
        }

        __syncthreads();

        // now we're ready to do the matmul of [16, 32] x [32,32] -> [16, 32]
        // M: 16, blockDim.y = 8, so 2 passes to do per thread...need register of size 2...
        for (int i = threadIdx.y; i < M; i += blockDim.y)
        {
            float tmp = 0.0;

            for (int k = 0; k < K_BATCH; k++)
            {
                tmp += buffer_X[k * (K_BATCH + 1) + i] * buffer_W[k * N_BATCH + threadIdx.x];
            }

            output_reg[i / blockDim.y] += tmp;
        }
    }
    // write sub matrix to output

    for (int i = threadIdx.y; i < M; i += blockDim.y)
    {
        OUT[blockIdx.x * M * N + i * N + (blockIdx.y * N_BATCH) + threadIdx.x] = output_reg[i / blockDim.y];
    }
}

torch::Tensor matmul(torch::Tensor X, torch::Tensor W)
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

    torch::Tensor output = torch::empty({NNODES, M, N},
                                        torch::TensorOptions()
                                            .dtype(X.dtype())
                                            .device(X.device()));

    dim3 gridDim, blockDim;

    blockDim.x = WARP_SIZE;
    blockDim.y = 4;

    gridDim.x = NNODES;
    gridDim.y = find_integer_divisor(N, N_BATCH);

    size_t shared_size = 0;
    void *sptr = nullptr;

    shared_array<float>((K_BATCH + 1) * K_BATCH, sptr, &shared_size); // use double the space for X for now just to deal with bank conflicts
    shared_array<float>(K_BATCH * N_BATCH, sptr, &shared_size);
    // shared_array<float>(M_BATCH * N_BATCH, sptr, &shared_size);

    matmul_kernel<<<gridDim, blockDim, shared_size>>>(X.data_ptr<float>(), W.data_ptr<float>(), output.data_ptr<float>(),
                                                      NNODES, M, N, K);

    return output;
}

#define N_CONSUMER_Y 2

__device__ inline void producer_wmma(
    barrier ready[],
    barrier filled[],
    float *bufferX,
    float *bufferW,
    const float *__restrict__ X,
    const float *__restrict__ W,
    const int M,
    const int N,
    const int K)
{
    // assume here that X is in column major wrt. channels, i.e X[nnodes, channels, lm]

    // consumer threads:
    //  threadIdx.x = 32, threadIdx.y = 8, nthreads = 256
    //  8 WMMA ops possible simultaneously: 16x8 .x. [8x16, 8x16, 8x16, 8x16, 8x16, 8x16, 8x16, 8x16 ]

    // 16x8 and 8x128 consumed per WMMA_K

    // double buffering, so load 16x16 and 16x128 into 2 buffers, results in (16**2 + 16*128) * 4 = 9216B of shared memory space.

    /*
    reorganize producer threads into this shape for reading in X
    */

    // assume that warps 0 and 1 are used for producing
    int tidy = threadIdx.y * blockDim.x + threadIdx.x / 16; // 4
    int tidx = threadIdx.y * blockDim.x + threadIdx.x % 16; // 16

    // nstages = K / 8
    for (int stage = 0; stage < K / WMMA_K; stage++)
    {
        ready[stage % 2].arrive_and_wait(); /* wait for buffer_(i%2) to be ready to be filled */

        // load 8x16 tile from X : [channels, lm]
        for (int i = tidy; i < WMMA_K; i += 4)
        {
            bufferX[((stage % 2) + i) * M_BATCH + tidx] = __ldg(X + blockIdx.x * M * N + (stage * WMMA_K + i) * M + tidx);
        }

        // load 8x128 from W
        for (int i = threadIdx.y; i < WMMA_K; i += 2)
        {
            for (int j = threadIdx.x; j < N; j += blockDim.x)
            {
                bufferW[(stage % 2 + i) * N + j] = __ldg(W + (stage * WMMA_K + i) * N + j);
            }
        }

        barrier::arrival_token token = filled[stage % 2].arrive(); /* buffer_(i) is filled */
    }
}

__device__ inline void consumer_wmma(
    barrier ready[],
    barrier filled[],
    float *buffer_X,
    float *buffer_W,
    float *__restrict__ OUTPUT,
    const int M,
    const int N,
    const int K)
{
    barrier::arrival_token token1 = ready[0].arrive(); /* buffer_0 is ready for initial fill */
    barrier::arrival_token token2 = ready[1].arrive(); /* buffer_1 is ready for initial fill */

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::col_major> a_frag, delta_a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> b_frag, delta_b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> ab_frag;

    int aRow = 0;
    int bCol = (blockIdx.y * (blockDim.y - N_CONSUMER_Y) + (threadIdx.y - N_CONSUMER_Y)) * WMMA_N;

    for (int stage = 0; stage < K / WMMA_K; stage++)
    {
        filled[stage % 2].arrive_and_wait(); /* wait for buffer_(i%2) to be filled */

        /* consume buffer_(i%2) */
        // buffer_X is stored as [k, M]: [16, 16], buffer_W is stored as [k, N]: [16, 128]
        wmma::load_matrix_sync(a_frag, buffer_X + ((stage % 2) * WMMA_M), WMMA_M);
        wmma::load_matrix_sync(b_frag, buffer_W + ((stage % 2) * N) + bCol, N);
        // wmma::load_matrix_sync(b_frag, W + bCol + k * N_TOTAL, N_TOTAL);

        for (int l = 0; l < a_frag.num_elements; l++)
        {
            float curr = a_frag.x[l];
            float tf32 = wmma::__float_to_tf32(curr);
            delta_a_frag.x[l] = wmma::__float_to_tf32(curr - tf32);
            a_frag.x[l] = tf32;
        }

        for (int l = 0; l < b_frag.num_elements; l++)
        {
            float curr = b_frag.x[l];
            float tf32 = wmma::__float_to_tf32(curr);
            delta_b_frag.x[l] = wmma::__float_to_tf32(curr - tf32);
            b_frag.x[l] = tf32;
        }

        wmma::mma_sync(ab_frag, a_frag, b_frag, ab_frag);
        wmma::mma_sync(ab_frag, a_frag, delta_b_frag, ab_frag);
        wmma::mma_sync(ab_frag, delta_a_frag, b_frag, ab_frag);

        barrier::arrival_token token = ready[stage % 2].arrive(); /* buffer_(i%2) is ready to be re-filled */
    }

    __syncthreads();

    // now lets fill output

    wmma::store_matrix_sync(OUTPUT + blockIdx.x * M * N + bCol + aRow * N, ab_frag, N, wmma::mem_row_major);
}

// N is the total number of float elements in arrays in and out
__global__ void producer_consumer_wmma_matmul(
    const float *__restrict__ X,
    const float *__restrict__ W,
    float *__restrict__ OUT, const int M, const int N, const int K)
{

    extern __shared__ char buffer[];

    void *sptr = buffer;
    size_t space = 0;

    float *buffer_X = shared_array<float>(2 * WMMA_K * M, sptr, &space);
    float *buffer_W = shared_array<float>(2 * WMMA_K * N, sptr, &space);

    // bar[0] and bar[1] track if buffers buffer_0 and buffer_1 are ready to be filled,
    // while bar[2] and bar[3] track if buffers buffer_0 and buffer_1 are filled-in respectively
    __shared__ barrier bar[4];

    if (threadIdx.y * blockDim.x + threadIdx.x < 4)
        init(bar + threadIdx.y * blockDim.x + threadIdx.x, blockDim.x * blockDim.y);

    __syncthreads();

    if (threadIdx.y < N_CONSUMER_Y)
        producer_wmma(bar, bar + 2, buffer_X, buffer_W, X, W, M, N, K);
    else
        consumer_wmma(bar, bar + 2, buffer_X, buffer_W, OUT, M, N, K);
}

torch::Tensor producer_consumer_matmul(torch::Tensor X, torch::Tensor W)
{
    const int NNODES = X.size(0);
    const int M = X.size(2);
    const int N = W.size(1);
    const int K = W.size(0);

    TORCH_CHECK(X.device().is_cuda(), "X must be a CUDA tensor");
    TORCH_CHECK(W.device().is_cuda(), "W must be a CUDA tensor");

    TORCH_CHECK(M == 16, "X dim=2 must have dimension 16 [(lmax +1)**2]");
    TORCH_CHECK(N % 16 == 0, "W dim=1 must be a multiple of 16");
    TORCH_CHECK(K % 16 == 0, "X dim=1 and W dim=0 must be a multiple of 16");

    torch::Tensor output = torch::empty({NNODES, M, N},
                                        torch::TensorOptions()
                                            .dtype(X.dtype())
                                            .device(X.device()));

    dim3 gridDim, blockDim;

    blockDim.x = WARP_SIZE;
    blockDim.y = min(8, find_integer_divisor(N, WMMA_N)) + N_CONSUMER_Y;

    gridDim.x = NNODES;
    gridDim.y = find_integer_divisor(N, blockDim.y * WMMA_N);

    size_t shared_size = 0;
    void *sptr = nullptr;

    assert(((unsigned long long)X.data_ptr<float>()) % 128 == 0);
    assert(((unsigned long long)W.data_ptr<float>()) % 128 == 0);
    assert(((unsigned long long)output.data_ptr<float>()) % 128 == 0);

    shared_array<float>(2 * WMMA_K * M, sptr, &shared_size);
    shared_array<float>(2 * WMMA_K * N, sptr, &shared_size);

    producer_consumer_wmma_matmul<<<gridDim, blockDim, shared_size>>>(X.data_ptr<float>(), W.data_ptr<float>(), output.data_ptr<float>(),
                                                                      M, N, K);

    return output;
}

/*
This is a test kernel that implements C = XW, where X = [16, nchannels_in] and W = [nchannels_in, nchannels_out]

this matmul is further decomposed into C = dXW + XdW + XW, where d represents a variable containng the loss in precision on going from F32 -> TF32

*/

template <int nodes_per_block>
__global__ void matmul_wmma_kernel(const float *__restrict__ X, const float *__restrict__ W, float *__restrict__ OUT, const int NNODES, const int M_TOTAL, const int N_TOTAL, const int K_TOTAL)
{

    extern __shared__ char buffer[];

    void *sptr = buffer;
    size_t space = 0;

    // X is always [16, n_channels], but we load this into 32x33 buffer so we remove bank conflicts
    float *buffer_X = shared_array<float>(K_BATCH * (nodes_per_block * WMMA_M + 1), sptr, &space);

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::col_major> a_frag[nodes_per_block], delta_a_frag[nodes_per_block];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> b_frag, delta_b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> ab_frag[nodes_per_block];

#pragma unroll
    for (int i = 0; i < nodes_per_block; i++)
    {
        wmma::fill_fragment(ab_frag[i], 0.0f);
    }

    for (int tid = threadIdx.y * blockDim.x + threadIdx.x; tid < K_BATCH * (nodes_per_block * WMMA_M + 1); tid += blockDim.x * blockDim.y)
    {
        buffer_X[tid] = 0.0;
    }

    __syncthreads();

    int aRow = 0;
    int bCol = (blockIdx.y * blockDim.y + threadIdx.y) * WMMA_N;

    for (int k_batch = 0; k_batch < K_TOTAL / K_BATCH; k_batch++)
    {

        int k_start = k_batch * K_BATCH;

        cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

// pipeline this load to make use of VRAM -> L2 -> L1 -> shared instructions. We consume 16x8 every WMMA, so we pre-load 16x32
#pragma unroll
        for (int node_id = 0; node_id < nodes_per_block; node_id++)
        {
            for (int j = threadIdx.y; j < M_TOTAL; j += blockDim.y)
            {
                cuda::memcpy_async(buffer_X + threadIdx.x * (nodes_per_block * WMMA_M + 1) + (node_id * 16) + j, X + (blockIdx.x * nodes_per_block + node_id) * M_TOTAL * K_TOTAL + j * K_TOTAL + k_start + threadIdx.x, sizeof(float), pipe);
            }
        }

        pipe.producer_commit();

        cuda::pipeline_consumer_wait_prior<0>(pipe);

        __syncthreads();

        for (int k_sub = 0; k_sub < K_BATCH; k_sub += WMMA_K)
        {
            int k = k_start + k_sub;

#pragma unroll
            for (int node = 0; node < nodes_per_block; node++)
            {
                wmma::load_matrix_sync(a_frag[node], buffer_X + k_sub * (nodes_per_block * WMMA_M + 1) + node * WMMA_M, (nodes_per_block * WMMA_M + 1));
            }

            wmma::load_matrix_sync(b_frag, W + bCol + k * N_TOTAL, N_TOTAL);

            // now lets do some in-register conversions to calculate dX, dW
#pragma unroll
            for (int node = 0; node < nodes_per_block; node++)
            {
                for (int l = 0; l < a_frag[node].num_elements; l++)
                {
                    float curr = a_frag[node].x[l];
                    float tf32 = wmma::__float_to_tf32(curr);
                    delta_a_frag[node].x[l] = wmma::__float_to_tf32(curr - tf32);
                    a_frag[node].x[l] = tf32;
                }
            }

            for (int l = 0; l < b_frag.num_elements; l++)
            {
                float curr = b_frag.x[l];
                float tf32 = wmma::__float_to_tf32(curr);
                delta_b_frag.x[l] = wmma::__float_to_tf32(curr - tf32);
                b_frag.x[l] = tf32;
            }

#pragma unroll
            for (int node = 0; node < nodes_per_block; node++)
            {
                wmma::mma_sync(ab_frag[node], a_frag[node], b_frag, ab_frag[node]);
                wmma::mma_sync(ab_frag[node], a_frag[node], delta_b_frag, ab_frag[node]);
                wmma::mma_sync(ab_frag[node], delta_a_frag[node], b_frag, ab_frag[node]);
            }
        }
    }
#pragma unroll
    for (int node = 0; node < nodes_per_block; node++)
    {
        wmma::store_matrix_sync(OUT + (blockIdx.x * nodes_per_block + node) * M_TOTAL * N_TOTAL + bCol + aRow * N_TOTAL, ab_frag[node], N_TOTAL, wmma::mem_row_major);
    }
}

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

    torch::Tensor output = torch::empty({NNODES, M, N},
                                        torch::TensorOptions()
                                            .dtype(X.dtype())
                                            .device(X.device()));

    dim3 gridDim, blockDim;

    constexpr int nodes_per_block = 2;

    blockDim.x = WARP_SIZE;
    blockDim.y = min(8, find_integer_divisor(N, WMMA_N));

    gridDim.x = NNODES / nodes_per_block;
    gridDim.y = find_integer_divisor(N, blockDim.y * WMMA_N);

    size_t shared_size = 0;
    void *sptr = nullptr;

    assert(((unsigned long long)X.data_ptr<float>()) % 128 == 0);
    assert(((unsigned long long)W.data_ptr<float>()) % 128 == 0);
    assert(((unsigned long long)output.data_ptr<float>()) % 128 == 0);

    shared_array<float>(K_BATCH * (nodes_per_block * WMMA_M + 1), sptr, &shared_size);

    matmul_wmma_kernel<nodes_per_block><<<gridDim, blockDim, shared_size>>>(X.data_ptr<float>(), W.data_ptr<float>(), output.data_ptr<float>(),
                                                                            NNODES, M, N, K);

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

        return matmul_wmma(X, W);
    }

    static variable_list backward(AutogradContext *ctx, variable_list grad_outputs)
    {
        auto saved_variables = ctx->get_saved_variables();

        auto W_T = saved_variables[0];

        torch::Tensor dX = matmul_wmma(grad_outputs[0].contiguous(), W_T);

        torch::Tensor undef;

        return {dX, undef, undef, undef};
    }
};

torch::Tensor matmul_fwd(
    torch::Tensor X,
    torch::Tensor W,
    torch::Tensor W_T)
{
    return MatmulAutograd::apply(X, W, W_T);
}

__global__ void linear_wmma_kernel(float *__restrict__ X,
                                   float *__restrict__ W,
                                   float *__restrict__ OUT,
                                   int *__restrict__ l_start,
                                   int *__restrict__ l_end,
                                   float *__restrict__ path_weights,
                                   const int ninstructions,
                                   const int NNODES,
                                   const int M_TOTAL,
                                   const int N_TOTAL,
                                   const int K_TOTAL)
{

    extern __shared__ char buffer[];

    void *sptr = buffer;
    size_t space = 0;

    // X is always [16, n_channels], but we load this into 32x33 buffer so we remove bank conflicts
    float *buffer_X = shared_array<float>(K_BATCH * (K_BATCH + 1), sptr, &space);
    float *buffer_out = shared_array<float>(M_TOTAL * blockDim.y * WMMA_N, sptr, &space);

    float *buffer_l_start = shared_array<float>(ninstructions, sptr, &space);
    float *buffer_l_end = shared_array<float>(ninstructions, sptr, &space);
    float *buffer_path_weights = shared_array<float>(ninstructions, sptr, &space);

    if (threadIdx.y == 0)
    {
        for (int i = threadIdx.x; i < ninstructions; i += blockDim.x)
        {
            buffer_l_start[i] = l_start[i];
            buffer_l_end[i] = l_end[i];
            buffer_path_weights[i] = path_weights[i];
        }
    }

    float *X_i = X + blockIdx.x * M_TOTAL * K_TOTAL;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::col_major> a_frag, delta_a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> b_frag, delta_b_frag;

    for (int tid = threadIdx.y * blockDim.x + threadIdx.x; tid < K_BATCH * (K_BATCH + 1); tid += blockDim.x * blockDim.y)
    {
        buffer_X[tid] = 0.0;
    }

    __syncthreads();

    int aRow = 0;
    int bCol = (blockIdx.y * blockDim.y + threadIdx.y) * WMMA_N;

    for (int instruction = 0; instruction < ninstructions; instruction++)
    {

        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> ab_frag;
        wmma::fill_fragment(ab_frag, 0.0f);

        int lstart = buffer_l_start[instruction];
        int lend = buffer_l_end[instruction];
        float pathw = buffer_path_weights[instruction];

        for (int k_batch = 0; k_batch < K_TOTAL / K_BATCH; k_batch++)
        {
            int k_start = k_batch * K_BATCH;

            cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

            // pipeline this load to make use of VRAM -> L2 -> L1 -> shared instructions. We consume 16x8 every WMMA, so we pre-load 16x32
            for (int j = threadIdx.y; j < M_TOTAL; j += blockDim.y)
            {
                cuda::memcpy_async(buffer_X + threadIdx.x * (K_BATCH + 1) + j, X_i + j * K_TOTAL + k_start + threadIdx.x, sizeof(float), pipe);
            }

            pipe.producer_commit();

            cuda::pipeline_consumer_wait_prior<0>(pipe);

            __syncthreads();

            for (int k_sub = 0; k_sub < K_BATCH; k_sub += WMMA_K)
            {
                int k = k_start + k_sub;

                wmma::load_matrix_sync(a_frag, buffer_X + k_sub * (K_BATCH + 1), (K_BATCH + 1));

                // wmma::load_matrix_sync(b_frag, W + bCol + k * N_TOTAL, N_TOTAL);
                wmma::load_matrix_sync(b_frag, W + (instruction * K_TOTAL * N_TOTAL) + bCol + k * N_TOTAL, N_TOTAL);

                // now lets do some in-register conversions to calculate dX, dW
                for (int l = 0; l < a_frag.num_elements; l++)
                {
                    float curr = a_frag.x[l];
                    float tf32 = wmma::__float_to_tf32(curr);
                    delta_a_frag.x[l] = wmma::__float_to_tf32(curr - tf32);
                    a_frag.x[l] = tf32;
                }

                for (int l = 0; l < b_frag.num_elements; l++)
                {
                    float curr = b_frag.x[l];
                    float tf32 = wmma::__float_to_tf32(curr);
                    delta_b_frag.x[l] = wmma::__float_to_tf32(curr - tf32);
                    b_frag.x[l] = tf32;
                }

                wmma::mma_sync(ab_frag, a_frag, b_frag, ab_frag);
                wmma::mma_sync(ab_frag, a_frag, delta_b_frag, ab_frag);
                wmma::mma_sync(ab_frag, delta_a_frag, b_frag, ab_frag);
            }
        }

        // apply path weight
        for (int i = 0; i < ab_frag.num_elements; i++)
        {
            ab_frag.x[i] = ab_frag.x[i] * pathw;
        }

        // wmma::store_matrix_sync(buffer_out + bCol + aRow * (blockDim.y * WMMA_N), ab_frag[instruction], (blockDim.y * WMMA_N), wmma::mem_row_major);

        wmma::store_matrix_sync(OUT + blockIdx.x * M_TOTAL * N_TOTAL + bCol + aRow * N_TOTAL, ab_frag, N_TOTAL, wmma::mem_row_major);

        /*for (int lm = lstart + threadIdx.y; lm < lend; lm += blockDim.y)
        {
            for (int channel = threadIdx.x; channel < blockDim.y * WMMA_N; channel += blockDim.x)
            {
                OUT[blockIdx.x * M_TOTAL * N_TOTAL + lm * N_TOTAL + (blockIdx.y * blockDim.y * WMMA_N) + channel] = buffer_out[lm * blockDim.y * WMMA_N + channel];
            }
        }*/
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
    blockDim.y = 8; // 8 * WMMA_N = 64

    gridDim.x = NNODES;
    gridDim.y = find_integer_divisor(N, blockDim.y * WMMA_N);

    // std::cout << "grid dim: " << gridDim.x << " " << gridDim.y << " " << gridDim.z << std::endl;
    // std::cout << "block dim: " << blockDim.x << " " << blockDim.y << std::endl;

    size_t shared_size = 0;
    void *sptr = nullptr;

    shared_array<float>(K_BATCH * (K_BATCH + 1), sptr, &shared_size);
    shared_array<float>(M * blockDim.y * WMMA_N, sptr, &shared_size);

    shared_array<float>(ninstructions, sptr, &shared_size);
    shared_array<float>(ninstructions, sptr, &shared_size);
    shared_array<float>(ninstructions, sptr, &shared_size);

    linear_wmma_kernel<<<gridDim, blockDim, shared_size>>>(X.data_ptr<float>(), W.data_ptr<float>(), output.data_ptr<float>(),
                                                           l_start.data_ptr<int>(),
                                                           l_end.data_ptr<int>(),
                                                           path_weights.data_ptr<float>(),
                                                           ninstructions,
                                                           NNODES, M, N, K);

    // cudaDeviceSynchronize();

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

        torch::Tensor dX = linear_(grad_outputs[0].contiguous(), W_T, l_start, l_end, path_weights);

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
    m.def("matmul_fwd", &matmul_fwd);
    m.def("matmul_wmma", &matmul_wmma);
    m.def("producer_consumer_matmul", &producer_consumer_matmul);
}
