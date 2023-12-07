#include <mma.h>
#include <cuda.h>
#include <torch/script.h>

#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_fp16.h>
#include <cuda_pipeline_primitives.h>

using namespace nvcuda;
using namespace std;
using namespace torch::autograd;
using namespace torch::indexing;

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

#define MATMUL_NUM_THREADS 128
/*test kernel for non-wmma linear */
template <const int BK, const int L>
__global__ void __launch_bounds__(MATMUL_NUM_THREADS) linear_kernel(float *__restrict__ X, float *__restrict__ W, float *__restrict__ OUT, const int NNODES, const int M, const int N, const int K)
{

    const uint node = blockIdx.x;
    const uint cCol = blockIdx.y;

    __shared__ float Xs[33 * 32];  // 16 * 64 BK, BM, need to pad 33 elements and not 32 so we don't have bank conflicts
    __shared__ float Ws[BK * 128]; // 16 * 64, BK, BN

    const float path_weight = 1.0f / sqrt((float)K);

    // X += node * M * K; // move pointer to start of X

    // OUT += node * M * N + cCol * BN; // move pointer to start of OUT

    float threadResults[4 * 4] = {0.0};
    float regM[4] = {0.0};
    float regN[4] = {0.0};

    const uint lstart = L * L;
    const uint nl = 2 * L + 1;

    W += L * 128 * 128 + cCol * 128;

    const uint threadCol = threadIdx.x % 32; // [0-32]
    const uint threadRow = threadIdx.x / 32; //  128 / 32 = [0-4]

    const uint innerRowB = threadIdx.x / 32; // [0-4]
    const uint innerColB = threadIdx.x % 32; // [0-32]
    const uint rowStrideB = 4;

    // L=0: 0, 16, 32, 48
    // gid = blockIdx.x * 16;
    // gid       : 0 1 2 3   4  5  6  7   8  9  10 11... 999
    // gid % 3   : 0 1 2 0   1  2  0  1   2  0  1  2
    // gid / 3   : 0 0 0 1   1  1  2  2   2  3  3  3
    // l=1       : 1 2 3 17, 18 19 33 34, 35 49 50 51

    // gid = blockIdx.x * 16 + offset + threadIdx.y (offset is iter * blockDim.y)
    // l_start + (id /3 * 16) + id % 3 = l_start + (gid / nl * 16) + gid % nl

    uint smem_flip_idx = 0;
    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) // 0, 16, 32
    {
        if (bkIdx % 32 == 0)
        {
            // load the next MxK (16x32) elements of X into shared memory, transposed -> KxM (32x16)
            for (uint offset = 0; offset < 4; offset++)
            {
                uint gid = blockIdx.x * 16 + offset * 4 + threadRow;

                uint lidx = lstart + ((gid / nl) * 16) + gid % nl; // which lm index we need to pick along X to load into shared memory

                if (gid < NNODES * nl)
                    Xs[threadCol * 33 + offset * 4 + threadRow] = X[lidx * K + bkIdx + threadCol];
            }
        }

        // load the next KxM elements of W into shared memory (16x128)
        for (uint k = 0; k < 4; k++)
        {
            for (uint offset = 0; offset + rowStrideB <= BK; offset += rowStrideB)
            {
                Ws[(innerRowB + offset) * N + k * 32 + innerColB] = W[(innerRowB + offset) * N + k * 32 + innerColB];
            }
        }

        __syncthreads();

        // read in shared memory into registers and perform computation
        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx)
        {
            // threadRow = threadIdx.x / 32
            for (uint i = 0; i < 4; ++i)
            {
                // regM[i] = path_weight * Xs[(smem_flip_idx * 16 + dotIdx) * 33 + threadRow * 4 + i];
                regM[i] = path_weight * Xs[(smem_flip_idx * 16 + dotIdx) * 33 + i * 4 + threadRow];
            }

            // threadCol = threadIdx.x % 32
            for (uint i = 0; i < 4; ++i)
            {
                regN[i] = Ws[dotIdx * N + threadCol * 4 + i]; // threadCol: 0-32
            }

            // inner-most loop over registers
            for (int resM = 0; resM < 4; ++resM)
            {
                for (int resN = 0; resN < 4; ++resN)
                {
                    threadResults[resM * 4 + resN] += regM[resM] * regN[resN];
                }
            }
        }

        __syncthreads();

        smem_flip_idx = 1 - smem_flip_idx; // alternates between 0, 1, 0, 1 for each BK slice, since we load 2xBKx32 Xs when bkIdx % 32 == 0
        W += BK * N;                       // move the pointer to W along by BKxN
    }

    // now write output to global memory
    for (uint resM = 0; resM < 4; ++resM)
    {
        uint gid = blockIdx.x * 16 + resM * 4 + threadRow;

        uint lidx = lstart + ((gid / nl) * 16) + gid % nl; // which lm index we need to pick along OUT to write

        for (uint resN = 0; resN < 4; ++resN)
        {
            // const int i = resM * 4;

            // float4 tmp_node1;

            // tmp_node1.x = threadResults[i + 0];
            // tmp_node1.y = threadResults[i + 1];
            // tmp_node1.z = threadResults[i + 2];
            // tmp_node1.w = threadResults[i + 3];

            // if (gid < NNODES * nl)
            //     reinterpret_cast<float4 *>(&OUT[lidx * N + threadCol * 4])[0] = tmp_node1;

            if (gid < NNODES * nl)
            {
                OUT[lidx * N + (threadCol * 4) + resN] = threadResults[resM * 4 + resN];
            }
        }
    }
}

__global__ void linear_wmma_kernel(const float *__restrict__ X, const float *__restrict__ W, float *__restrict__ OUT, const int NNODES, const uint M, const uint N, const uint K, const uint L)
{

    const uint cCol = blockIdx.y;

    extern __shared__ char buffer[];

    void *sptr = buffer;
    size_t space = 0;

    float *Xs = shared_array<float>(K_BATCH * (K_BATCH + 1), sptr, &space);
    float *buffer_out = shared_array<float>(M * blockDim.y * WMMA_N, sptr, &space);

    /*
        for (int tid = threadIdx.y * blockDim.x + threadIdx.x; tid < K_BATCH * (K_BATCH + 1); tid += blockDim.x * blockDim.y)
        {
            Xs[tid] = 0.0;
        }

        for (int tid = threadIdx.y * blockDim.x + threadIdx.x; tid < M * blockDim.y * WMMA_N; tid += blockDim.x * blockDim.y)
        {
            buffer_out[tid] = 0.0;
        }
    */

    // X += node * M * K; // move pointer to start of X

    // OUT += node * M * N + cCol * BN; // move pointer to start of OUT

    const float path_weight = 1.0f / sqrt((float)K);

    const uint lstart = L * L;
    const uint nl = 2 * L + 1;

    W += L * K * N; // move W to the correct weights sub-matrix

    const uint threadCol = threadIdx.x; // [0-32]
    const uint threadRow = threadIdx.y; //  128 / 32 = [0-4]
    const uint rowStrideB = blockDim.y;
    const uint nmiter = find_integer_divisor(M, rowStrideB);
    // wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> a_frag, delta_a_frag;
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::col_major> a_frag, delta_a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> b_frag, delta_b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> ab_frag;

    wmma::fill_fragment(ab_frag, 0.0f);

    // const uint aRow = 0;

    const uint bCol = (blockIdx.y * blockDim.y + threadIdx.y) * WMMA_N;

    for (uint bkIdx = 0; bkIdx < K; bkIdx += WMMA_K) // 0, 16, 32
    {
        __syncthreads();

        if (bkIdx % 32 == 0)
        {
            for (int m = 0; m < nmiter; m++)
            {
                // ((m * rowStrideB + threadRow)/nl) * 16 -> start index for each node
                // ((m * rowStrideB + threadRow) % nl -> index into the current L channel
                if (m * rowStrideB + threadRow < 16)
                {
                    int gid = blockIdx.x * 16 + m * rowStrideB + threadRow;
                    int lidx = lstart + (gid / nl) * 16 + gid % nl;

                    if (lidx < NNODES * 16) // bounds checking
                    {
                        Xs[threadCol * 33 + m * rowStrideB + threadRow] = X[lidx * K + bkIdx + threadCol];
                    }
                    else
                    {
                        Xs[threadCol * 33 + m * rowStrideB + threadRow] = 0.0f;
                    }
                }
            }
        }

        __syncthreads();

        if (bCol < N)
        {
            wmma::load_matrix_sync(a_frag, Xs + (bkIdx % 32) * 33, 33);
            wmma::load_matrix_sync(b_frag, W + bCol, N);

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

            wmma::store_matrix_sync(buffer_out + threadIdx.y * WMMA_N, ab_frag, blockDim.y * WMMA_N, wmma::mem_row_major);
        }

        W += WMMA_K * N; // move the pointer to W along by BKxN
    }

    __syncthreads();

    // N_START = cCol * blockDim.y * WMMA_N

    for (uint n_block = 0; n_block < min(N - cCol * (blockDim.y * WMMA_N), blockDim.y * WMMA_N) / blockDim.x; n_block++)
    {
        for (int m = 0; m < nmiter; m++)
        {
            // ((m * rowStrideB + threadRow)/nl) * 16 -> start index for each node for lm channel
            // ((m * rowStrideB + threadRow) % nl -> index into the current lm index
            if (m * rowStrideB + threadRow < 16)
            {
                int gid = blockIdx.x * 16 + m * rowStrideB + threadRow;

                int lidx = lstart + (gid / nl) * 16 + gid % nl;

                if (lidx < NNODES * 16 && cCol * (blockDim.y * WMMA_N) + n_block * 32 + threadCol < N)
                    OUT[lidx * N + cCol * (blockDim.y * WMMA_N) + n_block * 32 + threadCol] = path_weight * buffer_out[(m * rowStrideB + threadRow) * (blockDim.y * WMMA_N) + n_block * 32 + threadCol];
            }
        }
    }
}

torch::Tensor linear_wmma(torch::Tensor X, torch::Tensor W)
{

    const int NNODES = X.size(0);
    const int M = X.size(1);
    const int N = W.size(2);
    const int K = W.size(1);

    torch::Tensor output = torch::empty({NNODES, M, N},
                                        torch::TensorOptions()
                                            .dtype(X.dtype())
                                            .device(X.device()));

    dim3 blockDim;

    blockDim.y = min(8, find_integer_divisor(N, WMMA_N));
    blockDim.x = 32;

    size_t shared_size = 0;
    void *sptr = nullptr;

    shared_array<float>(K_BATCH * (K_BATCH + 1), sptr, &shared_size);
    shared_array<float>(M * blockDim.y * WMMA_N, sptr, &shared_size);

    vector<cudaStream_t> streams;

    for (int l = 0; l < 4; l++)
    {
        dim3 griddim;
        griddim.x = find_integer_divisor(NNODES * (2 * l + 1), 16);
        griddim.y = find_integer_divisor(N, blockDim.y * WMMA_N);

        cudaStream_t stream;

        cudaStreamCreate(&stream);

        linear_wmma_kernel<<<griddim, blockDim, shared_size, stream>>>(X.data_ptr<float>(), W.data_ptr<float>(), output.data_ptr<float>(), NNODES, M, N, K, l);

        streams.push_back(stream);
    }

    cudaDeviceSynchronize();

    for (int l = 0; l < streams.size(); l++)
    {
        cudaStreamDestroy(streams[l]);
    }

    return output;
}

__global__ void elemental_linear_wmma_kernel(
    const float *__restrict__ X,
    const float *__restrict__ W,
    const int64_t *__restrict__ node_idx,
    const int64_t nselected,
    const int64_t element_id,
    const int64_t nelements,
    float *__restrict__ OUT,
    const int NNODES,
    const uint M,
    const uint N,
    const uint K,
    const uint L)
{

    const uint cCol = blockIdx.y;

    extern __shared__ char buffer[];

    void *sptr = buffer;
    size_t space = 0;

    float *Xs = shared_array<float>(K_BATCH * (K_BATCH + 1), sptr, &space);
    float *buffer_out = shared_array<float>(M * blockDim.y * WMMA_N, sptr, &space);

    // sqrt((2L+1))/sqrt(n_channels * n_elements)
    // const float path_weight = sqrt((float)(2 * L + 1)) / sqrt((float)K * (float)nelements);
    const float path_weight = 1.0f / sqrt((float)K);

    const uint lstart = L * L;
    const uint nl = 2 * L + 1;

    W += element_id * 4 * K * N + L * K * N; // move W to the correct weights sub-matrix

    const uint threadCol = threadIdx.x; // [0-32]
    const uint threadRow = threadIdx.y; //  128 / 32 = [0-4]
    const uint rowStrideB = blockDim.y;
    const uint nmiter = find_integer_divisor(M, rowStrideB);
    // wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> a_frag, delta_a_frag;
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::col_major> a_frag, delta_a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> b_frag, delta_b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> ab_frag;

    wmma::fill_fragment(ab_frag, 0.0f);

    // const uint aRow = 0;

    const uint bCol = (blockIdx.y * blockDim.y + threadIdx.y) * WMMA_N;

    for (uint bkIdx = 0; bkIdx < K; bkIdx += WMMA_K) // 0, 16, 32
    {
        __syncthreads();

        if (bkIdx % 32 == 0)
        {
            for (int m = 0; m < nmiter; m++)
            {
                int gid = blockIdx.x * 16 + m * rowStrideB + threadRow;

                // ((m * rowStrideB + threadRow)/nl) * 16 -> start index for each node
                // ((m * rowStrideB + threadRow) % nl -> index into the current L channel
                if (m * rowStrideB + threadRow < 16 && gid / nl < nselected)
                {
                    int lidx = lstart + node_idx[(gid / nl)] * 16 + gid % nl;
                    // int lidx = lstart + gid / nl * 16 + gid % nl;
                    if (lidx < NNODES * 16) // bounds checking
                    {
                        Xs[threadCol * 33 + m * rowStrideB + threadRow] = X[lidx * K + bkIdx + threadCol];
                        // Xs[threadCol * 33 + m * rowStrideB + threadRow] = X[lidx * K + bkIdx + threadCol];
                    }
                    else
                    {
                        Xs[threadCol * 33 + m * rowStrideB + threadRow] = 0.0f;
                    }
                }
            }
        }

        __syncthreads();

        if (bCol < N)
        {
            wmma::load_matrix_sync(a_frag, Xs + (bkIdx % 32) * 33, 33);
            wmma::load_matrix_sync(b_frag, W + bCol, N);

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

            wmma::store_matrix_sync(buffer_out + threadIdx.y * WMMA_N, ab_frag, blockDim.y * WMMA_N, wmma::mem_row_major);
        }

        W += WMMA_K * N; // move the pointer to W along by BKxN
    }

    __syncthreads();

    // N_START = cCol * blockDim.y * WMMA_N

    for (uint n_block = 0; n_block < min(N - cCol * (blockDim.y * WMMA_N), blockDim.y * WMMA_N) / blockDim.x; n_block++)
    {
        for (int m = 0; m < nmiter; m++)
        {

            int gid = blockIdx.x * 16 + m * rowStrideB + threadRow;
            // ((m * rowStrideB + threadRow)/nl) * 16 -> start index for each node for lm channel
            // ((m * rowStrideB + threadRow) % nl -> index into the current lm index
            if (m * rowStrideB + threadRow < 16 && gid / nl < nselected)
            {

                /// int lidx = lstart + (gid / nl) * 16 + gid % nl;
                int lidx = lstart + node_idx[(gid / nl)] * 16 + gid % nl;

                if (lidx < NNODES * 16 && cCol * (blockDim.y * WMMA_N) + n_block * 32 + threadCol < N)
                {
                    OUT[lidx * N + cCol * (blockDim.y * WMMA_N) + n_block * 32 + threadCol] = path_weight * buffer_out[(m * rowStrideB + threadRow) * (blockDim.y * WMMA_N) + n_block * 32 + threadCol];
                    // OUT[lidx * N + cCol * (blockDim.y * WMMA_N) + n_block * 32 + threadCol] = path_weight * buffer_out[(m * rowStrideB + threadRow) * (blockDim.y * WMMA_N) + n_block * 32 + threadCol];
                }
            }
        }
    }
}

torch::Tensor elemental_linear_wmma(torch::Tensor X, torch::Tensor W, torch::Tensor elemental_embedding)
{

    const int NNODES = X.size(0);
    const int M = X.size(1);
    const int N = W.size(-1);
    const int K = W.size(2);

    torch::Tensor output = torch::empty({NNODES, M, N},
                                        torch::TensorOptions()
                                            .dtype(X.dtype())
                                            .device(X.device()));

    dim3 blockDim;

    blockDim.y = min(8, find_integer_divisor(N, WMMA_N));
    blockDim.x = 32;

    size_t shared_size = 0;
    void *sptr = nullptr;

    shared_array<float>(K_BATCH * (K_BATCH + 1), sptr, &shared_size);
    shared_array<float>(M * blockDim.y * WMMA_N, sptr, &shared_size);

    vector<cudaStream_t> streams;

    for (int element_id = 0; element_id < elemental_embedding.size(1); element_id++)
    {
        torch::Tensor elemental_embedding_ = elemental_embedding.index({Ellipsis, element_id});
        torch::Tensor node_idx = torch::where(elemental_embedding_ == 1)[0];

        int64_t nselected = node_idx.size(0);

        if (nselected > 0)
        {

            for (int l = 0; l < 4; l++)
            {
                dim3 griddim;
                griddim.x = find_integer_divisor(nselected * (2 * l + 1), 16);
                griddim.y = find_integer_divisor(N, blockDim.y * WMMA_N);

                cudaStream_t stream;

                cudaStreamCreate(&stream);

                elemental_linear_wmma_kernel<<<griddim, blockDim, shared_size, stream>>>(
                    X.data_ptr<float>(),
                    W.data_ptr<float>(),
                    node_idx.data_ptr<int64_t>(),
                    nselected,
                    element_id,
                    elemental_embedding.size(-1),
                    output.data_ptr<float>(),
                    NNODES, M, N, K, l);

                streams.push_back(stream);
            }
        }
    }

    cudaDeviceSynchronize();

    for (int l = 0; l < streams.size(); l++)
    {
        cudaStreamDestroy(streams[l]);
    }

    return output;
}

class LinearAutograd : public Function<LinearAutograd>
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

        torch::Tensor result = linear_wmma(X, W);

        return result;
    }

    static variable_list backward(AutogradContext *ctx, variable_list grad_outputs)
    {
        auto saved_variables = ctx->get_saved_variables();

        auto W_T = saved_variables[0];

        if (!grad_outputs[0].is_contiguous())
        {
            grad_outputs[0] = grad_outputs[0].contiguous();
        }

        torch::Tensor dX = linear_wmma(grad_outputs[0], W_T);

        torch::Tensor undef;

        return {dX, undef, undef};
    }
};

class ElementalLinearAutograd : public Function<ElementalLinearAutograd>
{
public:
    static torch::Tensor forward(
        AutogradContext *ctx,
        torch::Tensor X,
        torch::Tensor W,
        torch::Tensor W_transposed,
        torch::Tensor one_hot_embedding)
    {

        if (X.requires_grad())
        {
            ctx->save_for_backward({one_hot_embedding, W_transposed});
        }

        torch::Tensor result = elemental_linear_wmma(X, W, one_hot_embedding);

        return result;
    }

    static variable_list backward(AutogradContext *ctx, variable_list grad_outputs)
    {
        auto saved_variables = ctx->get_saved_variables();

        auto one_hot_embedding = saved_variables[0];
        auto W_T = saved_variables[1];

        if (!grad_outputs[0].is_contiguous())
        {
            grad_outputs[0] = grad_outputs[0].contiguous();
        }

        torch::Tensor dX = elemental_linear_wmma(grad_outputs[0], W_T, one_hot_embedding);

        torch::Tensor undef;

        return {dX, undef, undef, undef};
    }
};

torch::Tensor linear(
    torch::Tensor X,
    torch::Tensor W,
    torch::Tensor W_T)
{
    return LinearAutograd::apply(X, W, W_T);
}

torch::Tensor elemental_linear(
    torch::Tensor X,
    torch::Tensor W,
    torch::Tensor W_T,
    torch::Tensor one_hot_embedding)
{
    return ElementalLinearAutograd::apply(X, W, W_T, one_hot_embedding);
}

TORCH_LIBRARY(linear_wmma, m)
{
    m.def("linear", &linear);
    m.def("elemental_linear", &elemental_linear);
}
