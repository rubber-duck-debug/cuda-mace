#include <mma.h>
#include <cuda.h>
#include <torch/script.h>

#include <stdio.h>
#include <stdlib.h>
#include <cuda/barrier>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace nvcuda;
using namespace std;

#define WARP_SIZE 32

// MMA matrix tile dimensions.
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 8

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

__global__ void matmul_wmma_kernel(const float *__restrict__ X, const float *__restrict__ W, float *OUT, const int NNODES, const int M_TOTAL, const int N_TOTAL, const int K_TOTAL)
{

   extern __shared__ char buffer[];

   void *sptr = buffer;
   size_t space = 0;

   float *buffer_X = shared_array<float>(K_TOTAL * M_TOTAL, sptr, &space);

   const float *X_i = X + blockIdx.x * M_TOTAL * K_TOTAL;
   float *OUT_i = OUT + blockIdx.x * M_TOTAL * N_TOTAL;

   for (int i = 0; i < M_TOTAL / blockDim.y; i++)
   {
      for (int j = 0; j < K_TOTAL / blockDim.x; j++)
      {
         // reinterpret_cast<float4 *>(buffer_X)[(j * blockDim.x + threadIdx.x) * M_TOTAL + (i * blockDim.y + threadIdx.y)] = reinterpret_cast<float4 *>(X + blockIdx.x * M_TOTAL * K_TOTAL)[(i * blockDim.y + threadIdx.y) * K_TOTAL + (j * blockDim.x + threadIdx.x)];

         buffer_X[(j * blockDim.x + threadIdx.x) * M_TOTAL + (i * blockDim.y + threadIdx.y)] = X_i[(i * blockDim.y + threadIdx.y) * K_TOTAL + (j * blockDim.x + threadIdx.x)];
      }
   }

   __syncthreads();

   // bar.arrive_and_wait();

   // wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> a_frag;
   wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::col_major> a_frag;
   wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> b_frag;
   wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> ab_frag;

   wmma::fill_fragment(ab_frag, 0.0f);

   int a_row = 0;
   int b_col = (blockIdx.z * blockDim.y + threadIdx.y) * WMMA_N;

   for (int k = 0; k < K_TOTAL; k += WMMA_K)
   {
      // W 128, 128, load in [16, 128] slices here
      wmma::load_matrix_sync(a_frag, buffer_X + k * M_TOTAL + a_row, M_TOTAL);
      wmma::load_matrix_sync(b_frag, W + b_col + k * N_TOTAL, N_TOTAL);

      for (int i = 0; i < a_frag.num_elements; i++)
      {
         a_frag.x[i] = wmma::__float_to_tf32(a_frag.x[i]);
      }

      for (int i = 0; i < b_frag.num_elements; i++)
      {
         b_frag.x[i] = wmma::__float_to_tf32(b_frag.x[i]);
      }

      // Perform the matrix multiplication
      wmma::mma_sync(ab_frag, a_frag, b_frag, ab_frag);
   }

   // if (a_row < M_TOTAL && b_col < N_TOTAL)
   //{
   wmma::store_matrix_sync(OUT_i + b_col + a_row * N_TOTAL, ab_frag, N_TOTAL, wmma::mem_row_major);
   //}
}

torch::Tensor matmul_wmma(torch::Tensor X, torch::Tensor W, bool print_debug = false)
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

   torch::Tensor output = torch::zeros({NNODES, M, N},
                                       torch::TensorOptions()
                                           .dtype(X.dtype())
                                           .device(X.device()));

   dim3 gridDim, blockDim;
   blockDim.x = WARP_SIZE;
   blockDim.y = 8;

   // gridDim.x = (M_TOTAL + (M * blockDim.x / WARP_SIZE - 1)) / (M * blockDim.x / WARP_SIZE);
   // gridDim.y = (N_TOTAL + N * blockDim.y - 1) / (N * blockDim.y);

   gridDim.x = NNODES;
   gridDim.y = 1;
   gridDim.z = (N + WMMA_N * blockDim.y - 1) / (WMMA_N * blockDim.y);

   if (print_debug)
   {
      std::cout << gridDim.x << " " << gridDim.y << " " << gridDim.z << std::endl;
      std::cout << blockDim.x << " " << blockDim.y << std::endl;
   }

   size_t shared_size = 0;
   void *sptr = nullptr;

   shared_array<float>(M * K, sptr, &shared_size); // X

   matmul_wmma_kernel<<<gridDim, blockDim, shared_size>>>(X.data_ptr<float>(), W.data_ptr<float>(), output.data_ptr<float>(),
                                                          NNODES, M, N, K);

   cudaDeviceSynchronize();

   return output;
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

   const float *X_i = X + blockIdx.x * M_TOTAL * K_TOTAL;
   float *OUT_i = OUT + blockIdx.x * M_TOTAL * N_TOTAL;

   for (int i = 0; i < M_TOTAL / blockDim.y; i++)
   {
      for (int j = 0; j < K_TOTAL / blockDim.x; j++)
      {
         // reinterpret_cast<float4 *>(buffer_X)[(j * blockDim.x + threadIdx.x) * M_TOTAL + (i * blockDim.y + threadIdx.y)] = reinterpret_cast<float4 *>(X + blockIdx.x * M_TOTAL * K_TOTAL)[(i * blockDim.y + threadIdx.y) * K_TOTAL + (j * blockDim.x + threadIdx.x)];

         buffer_X[(j * blockDim.x + threadIdx.x) * M_TOTAL + (i * blockDim.y + threadIdx.y)] = X_i[(i * blockDim.y + threadIdx.y) * K_TOTAL + (j * blockDim.x + threadIdx.x)];
      }
   }

   __syncthreads();

   // bar.arrive_and_wait();

   // wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> a_frag;
   wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::col_major> a_frag;
   wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> b_frag;
   wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> ab_frag;

   int a_row = 0;
   int b_col = (blockIdx.z * blockDim.y + threadIdx.y) * WMMA_N;

   for (int instruction = 0; instruction < ninstructions; instruction++)
   {

      int lstart = l_start[instruction];
      int lend = l_end[instruction];
      float pathw = path_weights[instruction];

      wmma::fill_fragment(ab_frag, 0.0f);

      for (int k = 0; k < K_TOTAL; k += WMMA_K)
      {
         wmma::load_matrix_sync(a_frag, buffer_X + k * M_TOTAL + a_row, M_TOTAL);
         wmma::load_matrix_sync(b_frag, W + (instruction * K_TOTAL * N_TOTAL) + b_col + k * N_TOTAL, N_TOTAL);

         for (int i = 0; i < a_frag.num_elements; i++)
         {
            a_frag.x[i] = wmma::__float_to_tf32(a_frag.x[i]);
         }

         for (int i = 0; i < b_frag.num_elements; i++)
         {
            b_frag.x[i] = wmma::__float_to_tf32(b_frag.x[i]);
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
            OUT_i[lm * N_TOTAL + channel] = buffer_tmp_output[lm * N_TOTAL + channel];
         }
      }
   }
}

torch::Tensor linear_wmma(
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

   torch::Tensor output = torch::zeros({NNODES, M, N},
                                       torch::TensorOptions()
                                           .dtype(X.dtype())
                                           .device(X.device()));

   dim3 gridDim, blockDim;
   blockDim.x = WARP_SIZE;
   blockDim.y = 8;

   gridDim.x = NNODES;
   gridDim.y = 1;
   gridDim.z = (N + WMMA_N * blockDim.y - 1) / (WMMA_N * blockDim.y);

   if (print_debug)
   {
      std::cout << gridDim.x << " " << gridDim.y << " " << gridDim.z << std::endl;
      std::cout << blockDim.x << " " << blockDim.y << std::endl;
   }

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

TORCH_LIBRARY(linear_wmma, m)
{
   m.def("matmul", &matmul_wmma);
   m.def("linear_wmma", &linear_wmma);
}
