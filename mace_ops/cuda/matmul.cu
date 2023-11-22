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

const int MATMUL_NUM_THREADS = 128;

template <const int BM, const int BN, const int BK, const int BK_X, const int TM, const int TN>
__global__ void __launch_bounds__(MATMUL_NUM_THREADS) async_matmul_kernel(float *__restrict__ X, float *__restrict__ W, float *__restrict__ OUT, const int NNODES, const int M, const int N, const int K)
{

    const uint node = blockIdx.x;
    const uint cCol = blockIdx.y;

    __shared__ float Xs[128 * 16];     // 16 * 64 BK, BM
    __shared__ float Ws[2 * BK * 128]; // 16 * 64, BK, BN

    __shared__ cuda::barrier<cuda::thread_scope::thread_scope_block> frontBarrier;
    __shared__ cuda::barrier<cuda::thread_scope::thread_scope_block> backBarrier;

    auto frontBarrierPtr = &frontBarrier;
    auto backBarrierPtr = &backBarrier;

    auto block = cooperative_groups::this_thread_block();

    if (block.thread_rank() == 0)
    {
        init(&frontBarrier, block.size());
        init(&backBarrier, block.size());
    }
    __syncthreads();

    // X += node * M * K; // move pointer to start of X
    W += cCol * BN;
    // OUT += node * M * N + cCol * BN; // move pointer to start of OUT

    float threadResults[4 * 4] = {0.0};
    float regM[4] = {0.0};
    float regN[4] = {0.0};

    const int threadCol = threadIdx.x % 32; // [0-32]
    const int threadRow = threadIdx.x / 32; //  128 / 32 = [0-4]

    const uint innerRowA = threadIdx.x / 4; // 32
    const uint innerColA = threadIdx.x % 4; // 4
    const uint rowStrideA = 32;

    const uint innerRowB = threadIdx.x / 32; // [0-4]
    const uint innerColB = threadIdx.x % 32; // [0-32]
    const uint rowStrideB = 4;

    // read in all of X
    for (uint offset = 0; offset + rowStrideA <= K; offset += rowStrideA)
    {
        cuda::memcpy_async(&Xs[(innerRowA + offset) * BM + innerColA * 4],
                           &X[node * BM * K + (innerRowA + offset) * BM + innerColA * 4],
                           cuda::aligned_size_t<sizeof(float4)>(sizeof(float4)), *frontBarrierPtr);
    }

    float *Wd = W;
    int Ws_offset = 0;

    // load first tile of W into double buffer
    for (uint offset = 0; offset + rowStrideB <= BK; offset += rowStrideB)
    {
        cuda::memcpy_async(
            &Ws[(innerRowB + offset) * BN + innerColB * 4],
            &Wd[(innerRowB + offset) * N + innerColB * 4],
            cuda::aligned_size_t<sizeof(float4)>(sizeof(float4)), *frontBarrierPtr);
    }

    for (uint bkIdx = 0; bkIdx < K - BK; bkIdx += BK) // 0, 16, 32
    {
        // load next tile of W in double buffer
        for (uint offset = 0; offset + rowStrideB <= BK; offset += rowStrideB)
        {
            cuda::memcpy_async(
                &Ws[(1 - Ws_offset) * BK * BN + (innerRowB + offset) * BN + innerColB * 4],
                &Wd[BK * N + (innerRowB + offset) * N + innerColB * 4],
                cuda::aligned_size_t<sizeof(float4)>(sizeof(float4)), *backBarrierPtr);
        }
        // wait on first tile to complete
        (*frontBarrierPtr).arrive_and_wait();

        // read in shared memory into registers and perform computation
        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx)
        {
            // threadRow = threadIdx.x / 32
            for (uint i = 0; i < 4; ++i)
            {
                regM[i] = Xs[(bkIdx + dotIdx) * BM + threadRow * 4 + i]; // threadRow: 0-4
            }

            // threadCol = threadIdx.x % 32
            for (uint i = 0; i < 4; ++i)
            {
                regN[i] = Ws[Ws_offset * BK * BN + dotIdx * BN + threadCol * 4 + i]; // threadCol: 0-32
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

        Wd += BK * N;

        Ws_offset = 1 - Ws_offset;

        // swap the front and back barriers
        auto tmp = frontBarrierPtr;
        frontBarrierPtr = backBarrierPtr;
        backBarrierPtr = tmp;

        //__syncthreads();
    }

    // do final tile of computation
    (*frontBarrierPtr).arrive_and_wait();

    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx)
    {
        // threadRow = threadIdx.x / 32
        for (uint i = 0; i < 4; ++i)
        {
            regM[i] = Xs[((K - BK) + dotIdx) * BM + threadRow * 4 + i]; // threadRow: 0-4
        }

        // threadCol = threadIdx.x % 32
        for (uint i = 0; i < 4; ++i)
        {
            regN[i] = Ws[Ws_offset * BK * BN + dotIdx * BN + threadCol * 4 + i]; // threadCol: 0-32
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
    // do final tile of computation

    // now write output to global memory
    for (int resM = 0; resM < 4; ++resM)
    {
        const int i = resM * 4;

        float4 tmp_node1;

        tmp_node1.x = threadResults[i + 0];
        tmp_node1.y = threadResults[i + 1];
        tmp_node1.z = threadResults[i + 2];
        tmp_node1.w = threadResults[i + 3];

        reinterpret_cast<float4 *>(&OUT[node * M * N + cCol * BN + (threadRow * 4 + resM) * N +
                                        threadCol * 4])[0] = tmp_node1;

        // OUT[(threadRow * 4 + resM) * N + (threadCol * 4) + resN] = threadResults[resM * 4 + resN];
    }
}

template <const int BM, const int BN, const int BK, const int BK_X, const int TM, const int TN>
__global__ void __launch_bounds__(MATMUL_NUM_THREADS) matmul_kernel(float *__restrict__ X, float *__restrict__ W, float *__restrict__ OUT, const int NNODES, const int M, const int N, const int K)
{

    const uint node = blockIdx.x;
    const uint cCol = blockIdx.y;

    __shared__ float Xs[128 * 16]; // 16 * 64 BK, BM
    __shared__ float Ws[BK * 128]; // 16 * 64, BK, BN

    // X += node * M * K; // move pointer to start of X
    W += cCol * BN;
    // OUT += node * M * N + cCol * BN; // move pointer to start of OUT

    float threadResults[4 * 4] = {0.0};
    float regM[4] = {0.0};
    float regN[4] = {0.0};

    const int threadCol = threadIdx.x % 32; // [0-32]
    const int threadRow = threadIdx.x / 32; //  128 / 32 = [0-4]

    const uint innerRowA = threadIdx.x / 4; // 32
    const uint innerColA = threadIdx.x % 4; // 4
    const uint rowStrideA = 32;

    const uint innerRowB = threadIdx.x / 32; // [0-4]
    const uint innerColB = threadIdx.x % 32; // [0-32]
    const uint rowStrideB = 4;

    // read in all of X
    for (uint offset = 0; offset + rowStrideA <= K; offset += rowStrideA)
    {
        float4 tmp_node1 = reinterpret_cast<float4 *>(&X[node * BM * K + (innerRowA + offset) * BM + innerColA * 4])[0];
        Xs[(innerRowA + offset) * BM + innerColA * 4 + 0] = tmp_node1.x;
        Xs[(innerRowA + offset) * BM + innerColA * 4 + 1] = tmp_node1.y;
        Xs[(innerRowA + offset) * BM + innerColA * 4 + 2] = tmp_node1.z;
        Xs[(innerRowA + offset) * BM + innerColA * 4 + 3] = tmp_node1.w;
    }

    __syncthreads();

    // read in X once and re-use, since its small

    // for (int ll = 0; ll < 4; ll++)
    //{
    float *Wd = W;

    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) // 0, 16, 32
    {

        /* if (K % 128 == 0) {
            //read in next chunk of X
        } */

        for (uint offset = 0; offset + rowStrideB <= BK; offset += rowStrideB)
        {
            reinterpret_cast<float4 *>(
                &Ws[(innerRowB + offset) * BN + innerColB * 4])[0] =
                reinterpret_cast<float4 *>(
                    &Wd[(innerRowB + offset) * N + innerColB * 4])[0];
        }

        __syncthreads();

        // read in shared memory into registers and perform computation
        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx)
        {
            // threadRow = threadIdx.x / 32
            for (uint i = 0; i < 4; ++i)
            {
                regM[i] = Xs[(bkIdx + dotIdx) * BM + threadRow * 4 + i]; // threadRow: 0-4
            }

            // threadCol = threadIdx.x % 32
            for (uint i = 0; i < 4; ++i)
            {
                regN[i] = Ws[dotIdx * BN + threadCol * 4 + i]; // threadCol: 0-32
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

        Wd += BK * N;
    }
    //}

    // now write output to global memory
    for (int resM = 0; resM < 4; ++resM)
    {
        const int i = resM * 4;

        float4 tmp_node1;

        tmp_node1.x = threadResults[i + 0];
        tmp_node1.y = threadResults[i + 1];
        tmp_node1.z = threadResults[i + 2];
        tmp_node1.w = threadResults[i + 3];

        reinterpret_cast<float4 *>(&OUT[node * M * N + cCol * BN + (threadRow * 4 + resM) * N +
                                        threadCol * 4])[0] = tmp_node1;

        // OUT[(threadRow * 4 + resM) * N + (threadCol * 4) + resN] = threadResults[resM * 4 + resN];
    }
}

torch::Tensor do_matmul(torch::Tensor X, torch::Tensor W)
{

    const int NNODES = X.size(0);
    const int M = X.size(2);
    const int N = W.size(1);
    const int K = W.size(0);

    torch::Tensor output = torch::empty({NNODES, M, N},
                                        torch::TensorOptions()
                                            .dtype(X.dtype())
                                            .device(X.device()));

    dim3 gridDim, blockDim;

    blockDim.x = MATMUL_NUM_THREADS;

    gridDim.x = NNODES;
    gridDim.y = find_integer_divisor(N, 128);

    // template <const int BM, const int BN, const int BK, const int BK_X, const int TM, const int TN>

    matmul_kernel<16, 128, 16, 128, 4, 4><<<gridDim, blockDim>>>(X.data_ptr<float>(), W.data_ptr<float>(), output.data_ptr<float>(), NNODES, M, N, K);

    // cudaDeviceSynchronize();

    return output;
}

template <const int BM, const int BN, const int BK, const int BK_X, const int TM, const int TN>
__global__ void __launch_bounds__(MATMUL_NUM_THREADS) matmul_test_kernel(float *__restrict__ X, float *__restrict__ W, float *__restrict__ OUT, const int NNODES, const int M, const int N, const int K)
{

    const uint node = blockIdx.x;
    const uint cCol = blockIdx.y;

    __shared__ float Xs[BK * 128]; // 16 * 64 BK, BM
    __shared__ float Ws[BK * 128]; // 16 * 64, BK, BN

    // X += node * M * K; // move pointer to start of X
    W += cCol * BN;
    // OUT += node * M * N + cCol * BN; // move pointer to start of OUT

    float threadResults[4 * 4] = {0.0};
    float regM[4] = {0.0};
    float regN[4] = {0.0};

    const int threadCol = threadIdx.x % 32; // [0-32]
    const int threadRow = threadIdx.x / 32; //  128 / 32 = [0-4]

    const uint innerRowA = threadIdx.x / 4; // 32
    const uint innerColA = threadIdx.x % 4; // 4
    const uint rowStrideA = 32;

    const uint innerRowB = threadIdx.x / 32; // [0-4]
    const uint innerColB = threadIdx.x % 32; // [0-32]
    const uint rowStrideB = 4;

    // L=0: 0, 16, 32, 48

    // gid       : 0 1 2 3   4  5  6  7   8  9  10 11
    // gid % 3   : 0 1 2 0   1  2  0  1   2  0  1  2
    // gid / 3   : 0 0 0 1   1  1  2  2   2  3  3  3
    // l=1       : 1 2 3 17, 18 19 33 34, 35 49 50 51

    // gid = blockIdx.x * 16 + offset + threadIdx.y (offset is iter * blockDim.y)
    // l_start + (id /3 * 16) + id % 3 = l_start + (gid / nl * 16) + gid % nl

    //(id+1) / 3 * 16
    // blockIdx.x = 0, L=1, 1, 2, 3, 17

    // gridDim.x = nnodes/16
    // threadIdx.y: 0-4
    // l_start = 0
    // l_end = 1
    // 16 %

    // X[1000, 16, 128]
    // need to load[1000, l_start:l_end, 128]
    // want to load 16 elements into this block, have 4 warps in this dimension, lets say
    // e.g l_start =4, l_end=9
    // [4-9,128], [20-25, 128], [36,41, 128], [42, 128] -> [16,128]
    // next block loads [43-47, 128]
    //  warpId
    // read in all of X
    for (uint lm = 0; lm < 4; lm++)
    {
        for (uint offset = 0; offset + rowStrideA <= K; offset += rowStrideA)
        {
            Xs[(innerRowA + offset) * BM + lm * 4 + innerColA] = X[node * BM * K + (innerRowA + offset) * BM + lm * 4 + innerColA];
        }
    }

    //__syncthreads();

    // read in X once and re-use, since its small

    // for (int ll = 0; ll < 4; ll++)
    //{
    float *Wd = W;

    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) // 0, 16, 32
    {

        /* if (K % 128 == 0) {
            //read in next chunk of X
        } */

        for (uint k = 0; k < 4; k++)
        {
            for (uint offset = 0; offset + rowStrideB <= BK; offset += rowStrideB)
            {
                Ws[(innerRowB + offset) * BN + k * 32 + innerColB] = Wd[(innerRowB + offset) * N + k * 32 + innerColB];
            }
        }

        __syncthreads();

        // read in shared memory into registers and perform computation
        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx)
        {
            // threadRow = threadIdx.x / 32
            for (uint i = 0; i < 4; ++i)
            {
                regM[i] = Xs[(bkIdx + dotIdx) * BM + threadRow * 4 + i]; // threadRow: 0-4
            }

            // threadCol = threadIdx.x % 32
            for (uint i = 0; i < 4; ++i)
            {
                regN[i] = Ws[dotIdx * BN + threadCol * 4 + i]; // threadCol: 0-32
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

        Wd += BK * N;
    }
    //}

    // now write output to global memory
    for (int resM = 0; resM < 4; ++resM)
    {
        const int i = resM * 4;

        float4 tmp_node1;

        tmp_node1.x = threadResults[i + 0];
        tmp_node1.y = threadResults[i + 1];
        tmp_node1.z = threadResults[i + 2];
        tmp_node1.w = threadResults[i + 3];

        reinterpret_cast<float4 *>(&OUT[node * M * N + cCol * BN + (threadRow * 4 + resM) * N +
                                        threadCol * 4])[0] = tmp_node1;

        // OUT[(threadRow * 4 + resM) * N + (threadCol * 4) + resN] = threadResults[resM * 4 + resN];
    }
}

torch::Tensor matmul_test(torch::Tensor X, torch::Tensor W)
{

    const int NNODES = X.size(0);
    const int M = X.size(2);
    const int N = W.size(1);
    const int K = W.size(0);

    torch::Tensor output = torch::empty({NNODES, M, N},
                                        torch::TensorOptions()
                                            .dtype(X.dtype())
                                            .device(X.device()));

    dim3 gridDim, blockDim;

    blockDim.x = MATMUL_NUM_THREADS;

    gridDim.x = NNODES;
    gridDim.y = find_integer_divisor(N, 128);

    matmul_test_kernel<16, 128, 16, 128, 4, 4><<<gridDim, blockDim>>>(X.data_ptr<float>(), W.data_ptr<float>(), output.data_ptr<float>(), NNODES, M, N, K);

    // cudaDeviceSynchronize();

    return output;
}

template <const int BK, const int L>
__global__ void linear_wmma_kernel(float *__restrict__ X, float *__restrict__ W, float *__restrict__ OUT, const int NNODES, const int M, const int N, const int K)
{

    const uint cCol = blockIdx.y;

    extern __shared__ char buffer[];

    void *sptr = buffer;
    size_t space = 0;


    float *Xs = shared_array<float>(K_BATCH * (K_BATCH + 1), sptr, &space);
    float *buffer_out = shared_array<float>(M * blockDim.y * WMMA_N, sptr, &space);

    // X += node * M * K; // move pointer to start of X

    // OUT += node * M * N + cCol * BN; // move pointer to start of OUT

    const float path_weight = 1.0f / sqrt((float)K);

    const uint lstart = L * L;
    const uint nl = 2 * L + 1;

    W += L * K * N + cCol * blockDim.y * WMMA_N;

    const uint threadCol = threadIdx.x; // [0-32]
    const uint threadRow = threadIdx.y; //  128 / 32 = [0-4]
    const uint gtid = threadRow * blockDim.x + threadCol;
    const uint gid = blockIdx.x * 16 + threadRow;
    const uint rowStrideB = blockDim.y;

    if (gid >= NNODES * nl)
    {
        return;
    }

    for (uint offset = 0; offset < 16 * N; offset += blockDim.x * blockDim.y)
    {
        if (offset + gtid < 16 * N)
            buffer_out[offset + gtid] = 0.0;
    }

    // wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> a_frag, delta_a_frag;
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::col_major> a_frag, delta_a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> b_frag, delta_b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> ab_frag;

    wmma::fill_fragment(ab_frag, 0.0f);

    const uint aRow = 0;
    const uint bCol = (blockIdx.y * blockDim.y + threadIdx.y) * WMMA_N;

    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) // 0, 16, 32
    {
        if (bkIdx % 32 == 0)
        {
            // load the next MxK (16x32) elements of X into shared memory, transposed -> KxM (32x16)
            for (uint offset = 0; offset < 16; offset += rowStrideB)
            {
                if (offset + threadRow < 16)
                {
                    uint lidx = lstart + ((gid + offset) / nl) * 16 + (gid + offset) % nl; // which lm index we need to pick along X to load into shared memory

                    // Xs[(offset + threadRow) * 32 + threadCol] = X[lidx * K + bkIdx + threadCol];
                    Xs[threadCol * 33 + offset + threadRow] = X[lidx * K + bkIdx + threadCol];
                }
            }
        }

        __syncthreads();

        // wmma::load_matrix_sync(a_frag, Xs + bkIdx % 32, 32);
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

        W += BK * N; // move the pointer to W along by BKxN
    }

    wmma::store_matrix_sync(buffer_out + threadIdx.y * WMMA_N + aRow * N, ab_frag, N, wmma::mem_row_major);

    __syncthreads();

    // now write output to global memory
    for (uint n_block = 0; n_block < (blockDim.y * WMMA_N) / blockDim.x; ++n_block)
    {
        for (uint offset = 0; offset < 16; offset += rowStrideB)
        {

            if (offset + threadRow < 16 && cCol * (blockDim.y * WMMA_N) + n_block * 32 + threadCol < N)
            {
                uint lidx = lstart + (((gid + offset) / nl) * 16) + (gid + offset) % nl; // which lm index we need to pick along OUT to write

                // Xs[threadCol * 33 + offset * 4 + threadRow]
                // OUT[lidx * N + offset * 32 + threadCol] = buffer_out[];
                OUT[lidx * N + cCol * (blockDim.y * WMMA_N) + n_block * 32 + threadCol] = path_weight * buffer_out[threadRow * N + n_block * 32 + threadCol];
            }
        }
    }
}

torch::Tensor linear_wmma_test(torch::Tensor X, torch::Tensor W)
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

    cudaStream_t stream[4];

    dim3 griddims[4];

    size_t shared_size = 0;
    void *sptr = nullptr;

    shared_array<float>(K_BATCH * (K_BATCH + 1), sptr, &shared_size);
    shared_array<float>(M * blockDim.y * WMMA_N, sptr, &shared_size);

    for (int l = 0; l < 4; l++)
    {
        cudaStreamCreate(&stream[l]);

        griddims[l].x = find_integer_divisor(NNODES * (2 * l + 1), 16);
        griddims[l].y = find_integer_divisor(N, blockDim.y * WMMA_N);
    }
    linear_wmma_kernel<8, 0><<<griddims[0], blockDim, shared_size, stream[0]>>>(X.data_ptr<float>(), W.data_ptr<float>(), output.data_ptr<float>(), NNODES, M, N, K);
    linear_wmma_kernel<8, 1><<<griddims[1], blockDim, shared_size, stream[1]>>>(X.data_ptr<float>(), W.data_ptr<float>(), output.data_ptr<float>(), NNODES, M, N, K);
    linear_wmma_kernel<8, 2><<<griddims[2], blockDim, shared_size, stream[2]>>>(X.data_ptr<float>(), W.data_ptr<float>(), output.data_ptr<float>(), NNODES, M, N, K);
    linear_wmma_kernel<8, 3><<<griddims[3], blockDim, shared_size, stream[3]>>>(X.data_ptr<float>(), W.data_ptr<float>(), output.data_ptr<float>(), NNODES, M, N, K);

    for (int l = 0; l < 4; l++)
    {
        cudaStreamDestroy(stream[l]);
    }

    return output;
}

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

        torch::Tensor result = linear_wmma_test(X, W);

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

        torch::Tensor dX = linear_wmma_test(grad_outputs[0], W_T);

        torch::Tensor undef;

        return {dX, undef, undef};
    }
};

torch::Tensor linear(
    torch::Tensor X,
    torch::Tensor W,
    torch::Tensor W_T)
{
    return LinearAutograd::apply(X, W, W_T);
}

torch::Tensor linear_test(torch::Tensor X, torch::Tensor W)
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

    blockDim.x = MATMUL_NUM_THREADS;

    cudaStream_t stream[4];

    dim3 griddims[4];
    for (int l = 0; l < 4; l++)
    {
        cudaStreamCreate(&stream[l]);

        griddims[l].x = find_integer_divisor(NNODES * (2 * l + 1), 16);
        griddims[l].y = find_integer_divisor(N, 128);
    }
    linear_kernel<16, 0><<<griddims[0], blockDim, 0, stream[0]>>>(X.data_ptr<float>(), W.data_ptr<float>(), output.data_ptr<float>(), NNODES, M, N, K);
    linear_kernel<16, 1><<<griddims[1], blockDim, 0, stream[1]>>>(X.data_ptr<float>(), W.data_ptr<float>(), output.data_ptr<float>(), NNODES, M, N, K);
    linear_kernel<16, 2><<<griddims[2], blockDim, 0, stream[2]>>>(X.data_ptr<float>(), W.data_ptr<float>(), output.data_ptr<float>(), NNODES, M, N, K);
    linear_kernel<16, 3><<<griddims[3], blockDim, 0, stream[3]>>>(X.data_ptr<float>(), W.data_ptr<float>(), output.data_ptr<float>(), NNODES, M, N, K);

    for (int l = 0; l < 4; l++)
    {
        cudaStreamDestroy(stream[l]);
    }

    return output;
}

TORCH_LIBRARY(matmul, m)
{
    m.def("do_matmul", &do_matmul);
    m.def("matmul_test", &matmul_test);
    m.def("linear_test", &linear_test);
    m.def("linear_wmma_test", &linear_wmma_test);
    m.def("linear", &linear);
}
