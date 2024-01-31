#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

#include <cuda.h>
#include <cstdint>

using namespace std;

__host__ __device__ int32_t find_integer_divisor(int32_t x, int32_t bdim);

template <class T>
__host__ __device__ T *shared_array(std::size_t n_elements, void *&ptr, std::size_t *space) noexcept;

template <class T>
__host__ __device__ T *align_array(std::size_t n_elements, void *&ptr, const std::size_t alignment, std::size_t *space) noexcept;

#endif // CUDA_UTILS_CUH