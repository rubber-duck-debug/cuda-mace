#include <torch/script.h>
#include <iostream>
#include <cuda/barrier>

__constant__ int const_mus[353];
__constant__ float const_cg_coeffs[353];

using namespace std;
using namespace torch::indexing;
using namespace torch::autograd;

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

#define checkCudaErrors(call)                                       \
    do                                                              \
    {                                                               \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess)                                     \
        {                                                           \
            printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err));                        \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

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
__global__ __launch_bounds__(256) void forward_kernel(
    const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> X, // [nedges, (L1 + 1) ** 2, nchannels]
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> Y, // [nedges, (L2 + 1)**2]
    const torch::PackedTensorAccessor64<int32_t, 1, torch::RestrictPtrTraits> mus,
    const torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> cg_coeffs,
    const torch::PackedTensorAccessor64<int32_t, 1, torch::RestrictPtrTraits> sender_list,      // the list of edges we need to sum in the range specified by first_occurences
    const torch::PackedTensorAccessor64<int32_t, 1, torch::RestrictPtrTraits> receiver_list,    // which index we need to sum a particular edge into -> monotonically increasing.
    const torch::PackedTensorAccessor64<int32_t, 1, torch::RestrictPtrTraits> first_occurences, // the indexes in reciever_list which deliniate the set of edges per node.
    const int64_t l1,
    const int64_t l2,
    const int64_t l3,
    torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> output)
{

    const uint threadCol = threadIdx.x % 16;
    const uint threadRow = threadIdx.x / 16;
    const uint nthready = blockDim.x / 16;

    const uint N = X.size(2);
    const int nl3 = (l3 + 1) * (l3 + 1);

    extern __shared__ char buffer[];

    void *sptr = buffer;
    size_t space = 0;

    scalar_t *buffer_out = shared_array<scalar_t>(nthready * nl3 * N, sptr, &space);
    scalar_t *buffer_edge_y = shared_array<scalar_t>(nthready * 16, sptr, &space);
    scalar_t *buffer_edge_x = shared_array<scalar_t>(nthready * 16 * 16, sptr, &space);

    int *buffer_mus = shared_array<int>(mus.size(0), sptr, &space);
    scalar_t *buffer_cg_coeffs = shared_array<scalar_t>(mus.size(0), sptr, &space);
    // scalar_t regX[16] = {0.0};
    //  scalar_t regY[16] = {0.0};
    // scalar_t regOut[16] = {0.0};

    const uint edge_start = first_occurences[blockIdx.x];
    const uint edge_end = (blockIdx.x == first_occurences.size(0) - 1) ? receiver_list.size(0) : first_occurences[blockIdx.x + 1];
    const uint node_index = receiver_list[edge_start];

    for (int tid = threadIdx.x; tid < nthready * (l3 + 1) * (l3 + 1) * N; tid += blockDim.x)
    {
        buffer_out[tid] = 0.0;
    }

    for (int tid = threadIdx.x; tid < mus.size(0); tid += blockDim.x)
    {
        buffer_mus[tid] = mus[tid];
        buffer_cg_coeffs[tid] = cg_coeffs[tid];
    }

    __syncthreads();

    // check if this node has neighbours
    if (edge_end - edge_start == 0)
    {
        return;
    }

    int niter = find_integer_divisor(edge_end - edge_start, nthready);

    for (uint ni = 0; ni < niter; ni++)
    {
        uint edge = edge_start + ni * nthready + threadRow;

        // load shared mem

        if (edge < edge_end)
        {
            // uint sender_id = sender_list[edge];

            for (int m = 0; m < 16; m++)
            {
                // regX[m] = X[edge][m][threadCol];
                buffer_edge_x[threadRow * 16 * 16 + m * 16 + threadCol] = X[edge][m][threadCol];
            }

            buffer_edge_y[threadRow * 16 + threadCol] = Y[edge][threadCol];
        }

        __syncthreads();

        if (edge < edge_end)
        {
            // for (int m = 0; m < 16; m++)
            //{
            //     regX[m] = buffer_edge_x[threadRow * 16 * 16 + m * 16 + threadCol];
            //    regY[m] = buffer_edge_y[threadRow * 16 + m];
            //}

            for (int ins = 0; ins < 353; ins++)
            {
                int mus = buffer_mus[ins];

                int mu1 = (mus >> 8) & 0xFF;
                int mu2 = (mus >> 16) & 0xFF;
                int mu3 = (mus >> 24) & 0xFF;

                scalar_t cg_coeff = buffer_cg_coeffs[ins];

                scalar_t x = buffer_edge_x[threadRow * 16 * 16 + mu1 * 16 + threadCol];
                scalar_t y = buffer_edge_y[threadRow * 16 + mu2];

                //regOut[mu3] += x * y * cg_coeff;
                buffer_out[threadRow * nl3 * X.size(2) + mu3 * X.size(2) + threadCol] += x * y * cg_coeff;
            }
        }
    }

    //for (int m = 0; m < 16; m++)
    //{
    //    buffer_out[threadRow * nl3 * X.size(2) + m * X.size(2) + threadCol] = regOut[m];
    //} 

    __syncthreads();

    for (int m = threadRow; m < 16; m += nthready)
    {
        scalar_t tmp = 0.0;

        for (int i = 0; i < nthready; i++)
        {
            tmp += buffer_out[i * nl3 * 16 + m * 16 + threadCol];
        }

        output[node_index][m][threadCol] = tmp;
    }
}

__global__ void print_const_kernel()
{
    for (int i = 0; i < 353; i++)
    {
        int mus = const_mus[i];
        int mu1 = (mus >> 8) & 0xFF;
        int mu2 = (mus >> 16) & 0xFF;
        int mu3 = (mus >> 24) & 0xFF;

        printf("thread: %d const mem mu1: %d mu2: %d mu3: %d cg_coeffs: %f\n", threadIdx.x, mu1, mu2, mu3, const_cg_coeffs[i]);
    }
}

void copyToConstantMemory(torch::Tensor mus, torch::Tensor cg_coeffs, bool debug)
{
    checkCudaErrors(cudaMemcpyToSymbol(const_mus, mus.data_ptr<int>(), 353 * sizeof(int), 0, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyToSymbol(const_cg_coeffs, cg_coeffs.data_ptr<float>(), 353 * sizeof(float), 0, cudaMemcpyHostToDevice));

    if (debug)
    {
        print_const_kernel<<<1, 1>>>();

        cudaDeviceSynchronize();
    }
}

torch::Tensor forward_gpu(
    torch::Tensor X,
    torch::Tensor Y,
    torch::Tensor mus,
    torch::Tensor cg_coeffs,
    torch::Tensor sender_list,
    torch::Tensor receiver_list,
    torch::Tensor first_occurences,
    const int64_t nnodes,
    const int64_t l1,
    const int64_t l2,
    const int64_t l3)
{
    uint nedges = Y.size(0);
    int nspherical_harm = Y.size(1);
    int nfeatures = X.size(2);

    TORCH_CHECK(nfeatures % 16 == 0, "feature dimension must be a multiple of 32");
    TORCH_CHECK(nspherical_harm == 16, "number of edge spherical harmonics must be 16");
    TORCH_CHECK(nfeatures <= 64, "feature dimension cannot be greater than 128");

    torch::Tensor output = torch::empty({nnodes, (l3 + 1) * (l3 + 1), nfeatures},
                                        torch::TensorOptions()
                                            .dtype(X.dtype())
                                            .device(X.device()));

    dim3 gridDim(nnodes, 1);

    dim3 blockDim(256, 1, 1);

    AT_DISPATCH_FLOATING_TYPES(
        X.type(), "forward_gpu", ([&]
                                  {
        size_t space = 0;
        void* sptr = nullptr;

        const uint nthready = blockDim.x / 16;

    
        shared_array<scalar_t>(nthready * (l3 + 1) * (l3 + 1) * nfeatures, sptr, &space);
        shared_array<scalar_t>(nthready * 16, sptr, &space);
        shared_array<scalar_t>(nthready * 16 * 16, sptr, &space);
        shared_array<scalar_t>(mus.size(0), sptr, &space);
        shared_array<scalar_t>(cg_coeffs.size(0), sptr, &space);

        forward_kernel<scalar_t><<<gridDim, blockDim, space>>>(
            X.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
            Y.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
            mus.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>(),
            cg_coeffs.packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>(),
            sender_list.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>(),
            receiver_list.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>(),
            first_occurences.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>(),
            l1,
            l2,
            l3,
            output.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>()
            ); }));

    cudaDeviceSynchronize();

    return output;
}

#define NEIGHBOUR_NEDGES_PER_BLOCK 512

/*
This function takes a sorted input sender_list, which maps each edge to a node by index, and finds the positions of first occurences

This is required by the CUDA code so we can send all calculations per-node to a single block.

the function loads NEIGHBOUR_NEDGES_PER_BLOCK + 1 elements into shared memory, and then loops through the buffer twice. Once for even boundaries, once for odd boundaries.
*/

__global__ void calculate_first_occurences_kernel(const torch::PackedTensorAccessor64<int32_t, 1, torch::RestrictPtrTraits> receiver_list,
                                                  const int32_t *__restrict__ sort_idx,
                                                  bool use_sort,
                                                  torch::PackedTensorAccessor64<int32_t, 1, torch::RestrictPtrTraits> first_occurences)
{
    extern __shared__ char buffer[];
    size_t offset = 0;
    int32_t *smem = reinterpret_cast<int32_t *>(buffer + offset);

    int32_t block_start = blockIdx.x * NEIGHBOUR_NEDGES_PER_BLOCK;

    int32_t nedges = receiver_list.size(0);

    // load all elements of senderlist needed by block into shared memory
    for (int32_t i = threadIdx.x; i < NEIGHBOUR_NEDGES_PER_BLOCK + 1; i += blockDim.x)
    {
        int32_t idx = block_start + i;

        if (idx < nedges)
        {
            if (use_sort)
            {
                smem[i] = receiver_list[sort_idx[idx]];
            }
            else
            {
                smem[i] = receiver_list[idx];
            }
        }
    }

    __syncthreads();

    // deal with even boundaries
    for (int32_t i = 2 * threadIdx.x; i < NEIGHBOUR_NEDGES_PER_BLOCK; i += 2 * blockDim.x)
    {
        int32_t idx = block_start + i;

        if (idx + 1 < nedges)
        {
            int32_t loc1 = smem[i];
            int32_t loc2 = smem[i + 1];

            if (loc1 != loc2)
            {
                first_occurences[loc2] = idx + 1;
            }
        }
    }

    // deal with odd boundaries
    for (int32_t i = 2 * threadIdx.x + 1; i < NEIGHBOUR_NEDGES_PER_BLOCK + 1; i += 2 * blockDim.x)
    {
        int32_t idx = block_start + i;

        if (idx + 1 < nedges)
        {
            int32_t loc1 = smem[i];
            int32_t loc2 = smem[i + 1];

            if (loc1 != loc2)
            {
                first_occurences[loc2] = idx + 1;
            }
        }
    }

    // deal with 0th element specifically, so we dont need to use torch::zeros
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        first_occurences[0] = 0;
    }
}

torch::Tensor calculate_first_occurences_gpu(torch::Tensor receiver_list, int64_t natoms, int64_t nthreadx, torch::Tensor sort_indices)
{
    torch::Tensor first_occurences = torch::empty(natoms,
                                                  torch::TensorOptions()
                                                      .dtype(receiver_list.dtype())
                                                      .device(receiver_list.device()));

    int32_t nbx = find_integer_divisor(receiver_list.size(0), NEIGHBOUR_NEDGES_PER_BLOCK);

    dim3 block_dim(nbx);

    dim3 grid_dim(nthreadx, 1, 1);

    size_t total_buff_size = 0;

    total_buff_size += (NEIGHBOUR_NEDGES_PER_BLOCK + 1) * sizeof(int32_t);

    calculate_first_occurences_kernel<<<block_dim, grid_dim, total_buff_size>>>(
        receiver_list.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>(),
        sort_indices.data_ptr<int32_t>(),
        sort_indices.defined() && sort_indices.numel() != 0,
        first_occurences.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>());

    cudaDeviceSynchronize();

    return first_occurences;
}

class EquivariantMessagePassingTPAutograd : public Function<EquivariantMessagePassingTPAutograd>
{
public:
    static torch::Tensor forward(
        AutogradContext *ctx,
        torch::Tensor X,
        torch::Tensor Y,
        torch::Tensor mus,
        torch::Tensor cg_coeffs,
        torch::Tensor sender_list,
        torch::Tensor receiver_list,
        torch::Tensor first_occurences,
        const int64_t nnodes,
        const int64_t l1,
        const int64_t l2,
        const int64_t l3)
    {

        return forward_gpu(X, Y, mus, cg_coeffs, sender_list, receiver_list, first_occurences, nnodes, l1, l2, l3);
    }

    static variable_list backward(AutogradContext *ctx, variable_list grad_outputs)
    {

        torch::Tensor undef;

        return {undef, undef, undef, undef, undef, undef, undef, undef, undef};
    }
};

torch::Tensor equivariant_message_passing_tensor_product(
    torch::Tensor X,
    torch::Tensor Y,
    torch::Tensor mus,
    torch::Tensor cg_coeffs,
    torch::Tensor sender_list,
    torch::Tensor receiver_list,
    torch::Tensor first_occurences,
    const int64_t nnodes,
    const int64_t l1,
    const int64_t l2,
    const int64_t l3)
{
    return EquivariantMessagePassingTPAutograd::apply(X, Y, mus, cg_coeffs, sender_list, receiver_list, first_occurences, nnodes, l1, l2, l3);
}

TORCH_LIBRARY(equivariant_tp, m)
{
    m.def("forward", &equivariant_message_passing_tensor_product);
    m.def("calculate_first_occurences", &calculate_first_occurences_gpu);
    m.def("copyToConstantMemory", &copyToConstantMemory);
}
