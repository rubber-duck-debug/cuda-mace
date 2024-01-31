#include <cuda.h>
#include <cstdint>
#include "cuda_utils.cuh"

using namespace std;

__host__ __device__ int32_t find_integer_divisor(int32_t x, int32_t bdim)
{
    return (x + bdim - 1) / bdim;
}

template <class T>
__host__ __device__ T *shared_array(std::size_t n_elements, void *&ptr, std::size_t *space) noexcept
{
    const std::uintptr_t inptr = reinterpret_cast<uintptr_t>(ptr);
    const std::uintptr_t end = inptr + n_elements * sizeof(T);
    if (space)
        *space += static_cast<std::size_t>(end - inptr);
    ptr = reinterpret_cast<void *>(end);
    return reinterpret_cast<T *>(inptr);
}

//forward declare multiple types...
template float * shared_array<float>(std::size_t n_elements, void *&ptr, std::size_t *space) noexcept;
template double * shared_array<double>(std::size_t n_elements, void *&ptr, std::size_t *space) noexcept;
template int * shared_array<int>(std::size_t n_elements, void *&ptr, std::size_t *space) noexcept;
template short * shared_array<short>(std::size_t n_elements, void *&ptr, std::size_t *space) noexcept;

template <class T>
__host__ __device__ T *align_array(std::size_t n_elements, void *&ptr, const std::size_t alignment, std::size_t *space) noexcept
{
    const std::uintptr_t intptr = reinterpret_cast<uintptr_t>(ptr);
    const std::uintptr_t aligned = (intptr + alignment - 1) & -alignment;
    const std::uintptr_t end = aligned + n_elements * sizeof(T);
    if (space)
        *space += static_cast<std::size_t>(end - intptr);
    ptr = reinterpret_cast<void *>(end);
    return reinterpret_cast<T *>(aligned);
}