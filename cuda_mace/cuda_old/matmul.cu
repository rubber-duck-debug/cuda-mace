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

TORCH_LIBRARY(matmul, m)
{
    m.def("do_matmul", &do_matmul);
    m.def("matmul_test", &matmul_test);

}
