#include <torch/script.h>
#include "mace_ops/cuda/include/embedding_tools.h"
#include "mace_ops/cuda/include/cuda_utils.h"

#define FULL_MASK 0xffffffff

__global__ void parse_elemental_embedding_kernel(const int *__restrict__ elemental_embedding, const int nelements, const int nnodes, int *__restrict__ output, int *__restrict__ output_n_elements)
{
    extern __shared__ char buffer[];

    void *sptr = buffer;
    size_t space = 0;

    // int32_t *buffer_embedding = shared_array<int32_t>(blockDim.y * blockDim.x, sptr, &space);
    int32_t *buffer_indices = shared_array<int32_t>(blockDim.y * blockDim.x, sptr, &space);

    for (int element_id = threadIdx.y; element_id < nelements; element_id += blockDim.y)
    {
        int global_element_id = 0;
        for (int offset = 0; offset < nnodes; offset += blockDim.x)
        {
            int condition = elemental_embedding[(threadIdx.x + offset) * nelements + element_id] == 1;

            int warpBallot = __ballot_sync(0xFFFFFFFF, condition);
            int nBitsSet = __popc(warpBallot);                            // how many threads match the condition
            int position = __popc(warpBallot & ((1 << threadIdx.x) - 1)); // position of each thread in warpBallot, ordered from 0

            // Update the buffer array with the ordered list of lane IDs
            if (condition)
            {
                buffer_indices[threadIdx.y * blockDim.x + position] = offset + threadIdx.x; // should be conflict free, although bank mapping isn't ordered
            }

            __syncwarp();
            // writeout to global memory using all threads
            if (threadIdx.x < nBitsSet)
            {
                output[element_id * nnodes + global_element_id + threadIdx.x] = buffer_indices[threadIdx.y * blockDim.x + threadIdx.x];
            }

            global_element_id += nBitsSet;
        }

        if (threadIdx.x == 0)
            output_n_elements[element_id] = global_element_id;
    }
}

std::vector<torch::Tensor> parse_elemental_embedding_v1(torch::Tensor elemental_embedding)
{

    const uint nnodes = elemental_embedding.size(0);
    const uint nelements = elemental_embedding.size(1);

    torch::Tensor output = torch::empty({nelements, nnodes},
                                        torch::TensorOptions()
                                            .dtype(elemental_embedding.dtype())
                                            .device(elemental_embedding.device()));

    torch::Tensor output_nelements = torch::empty({nelements},
                                                  torch::TensorOptions()
                                                      .dtype(elemental_embedding.dtype())
                                                      .device(elemental_embedding.device()));

    dim3 blockDim;

    blockDim.y = max((int64_t)16, elemental_embedding.size(-1));
    blockDim.x = 32;

    size_t shared_size = 0;
    void *sptr = nullptr;

    // shared_array<int32_t>(blockDim.y * blockDim.x, sptr, &shared_size);
    shared_array<int32_t>(blockDim.y * blockDim.x, sptr, &shared_size);

    dim3 griddim(1);

    parse_elemental_embedding_kernel<<<griddim, blockDim, shared_size>>>(
        elemental_embedding.data_ptr<int32_t>(),
        elemental_embedding.size(-1),
        elemental_embedding.size(0),
        output.data_ptr<int32_t>(),
        output_nelements.data_ptr<int32_t>());

    cudaDeviceSynchronize();

    return {output, output_nelements};
}

#define NODES_PER_BLOCK 32

__global__ void parse_elemental_embedding_kernel_v2_1(const int *__restrict__ elemental_embedding, const int nnodes, const int nelements, int *__restrict__ output, int *__restrict__ output_n_elements_per_block)
{
    extern __shared__ char buffer[];

    void *sptr = buffer;
    size_t space = 0;

    // int32_t *buffer_embedding = shared_array<int32_t>(blockDim.y * blockDim.x, sptr, &space);
    int32_t *buffer_indices = shared_array<int32_t>(blockDim.y * NODES_PER_BLOCK, sptr, &space);

    int node_start = blockIdx.x * NODES_PER_BLOCK;

    for (int element_id = threadIdx.y; element_id < nelements; element_id += blockDim.y)
    {
        int global_element_id = 0;

        for (int offset = 0; offset < NODES_PER_BLOCK; offset += blockDim.x)
        {
            int node = node_start + offset + threadIdx.x;

            int condition = node < nnodes ? elemental_embedding[node * nelements + element_id] == 1 : false;

            int warpBallot = __ballot_sync(FULL_MASK, condition);
            int nBitsSet = __popc(warpBallot);                            // how many threads match the condition
            int position = __popc(warpBallot & ((1 << threadIdx.x) - 1)); // position of each thread in warpBallot, ordered from 0

            // Update the buffer array with the ordered list of lane IDs
            if (condition)
            {
                buffer_indices[threadIdx.y * blockDim.x + global_element_id + position] = node; // should be conflict free, although bank mapping isn't ordered
            }

            __syncwarp();
            // writeout per-block per-element sorted node lists to global memory using all threads
            if (threadIdx.x < nBitsSet && nBitsSet > 0)
            {
                output[blockIdx.x * nelements * NODES_PER_BLOCK + element_id * NODES_PER_BLOCK + global_element_id + threadIdx.x] = buffer_indices[threadIdx.y * blockDim.x + threadIdx.x];
            }

            global_element_id += nBitsSet;
        }

        if (threadIdx.x == 0)
            output_n_elements_per_block[blockIdx.x * nelements + element_id] = global_element_id; // write-out partial nelements_per_block
    }
}

__global__ void parse_elemental_embedding_kernel_v2_2(
    const int *__restrict__ per_block_element_node_ids,
    const int *__restrict__ per_block_n_nodes_per_element,
    const int nnodes,
    const int nelements,
    int *__restrict__ output)
{

    extern __shared__ char buffer[];

    void *sptr = buffer;
    size_t space = 0;

    int *buffer_block_n_nodes_per_element = shared_array<int>(gridDim.x * nelements, sptr, &space);

    // input: nblocks, nelements, ids
    // input: nblocks, nelements, elements_per_block

    // output: nelements, ids

    // read in per_block_n_nodes_per_element into shared memory first...
    for (int tid = threadIdx.y * blockDim.x + threadIdx.x; tid < gridDim.x * nelements; tid += blockDim.x * blockDim.y)
        buffer_block_n_nodes_per_element[tid] = per_block_n_nodes_per_element[tid];

    __syncthreads();

    for (int element_id = threadIdx.y; element_id < nelements; element_id += blockDim.y)
    {
        int out_start = 0;
        // now find starting position
        for (int i = 0; i < blockIdx.x; i++)
        {
            out_start += buffer_block_n_nodes_per_element[i * nelements + element_id];
        }
        
        int n_nodes_local = per_block_n_nodes_per_element[blockIdx.x * nelements + element_id];

        for (int offset = 0; offset < n_nodes_local; offset += blockDim.x)
        {
            if (offset + threadIdx.x < n_nodes_local)
            {
                output[element_id * nnodes + out_start + offset + threadIdx.x] = per_block_element_node_ids[blockIdx.x * nelements * NODES_PER_BLOCK + element_id * NODES_PER_BLOCK + offset + threadIdx.x];
            }
        }
    }
}

std::vector<torch::Tensor> parse_elemental_embedding_v2(torch::Tensor elemental_embedding)
{

    const uint nnodes = elemental_embedding.size(0);
    const uint nelements = elemental_embedding.size(1);

    const uint nblocks = find_integer_divisor(nnodes, NODES_PER_BLOCK);
    torch::Tensor output_tmp = torch::empty({nblocks, nelements, NODES_PER_BLOCK},
                                            torch::TensorOptions()
                                                .dtype(elemental_embedding.dtype())
                                                .device(elemental_embedding.device()));

    torch::Tensor output = torch::empty({nelements, nnodes},
                                        torch::TensorOptions()
                                            .dtype(elemental_embedding.dtype())
                                            .device(elemental_embedding.device()));

    torch::Tensor output_nelements = torch::empty({nblocks, nelements},
                                                  torch::TensorOptions()
                                                      .dtype(elemental_embedding.dtype())
                                                      .device(elemental_embedding.device()));

    dim3 blockDim;

    blockDim.y = max((int64_t)16, elemental_embedding.size(-1));
    blockDim.x = 32;

    size_t shared_size = 0;
    void *sptr = nullptr;

    // shared_array<int32_t>(blockDim.y * blockDim.x, sptr, &shared_size);
    shared_array<int32_t>(blockDim.y * NODES_PER_BLOCK, sptr, &shared_size);

    dim3 griddim(nblocks);

    parse_elemental_embedding_kernel_v2_1<<<griddim, blockDim, shared_size>>>(
        elemental_embedding.data_ptr<int32_t>(),
        elemental_embedding.size(0),
        elemental_embedding.size(-1),
        output_tmp.data_ptr<int32_t>(),
        output_nelements.data_ptr<int32_t>());

    shared_size = 0;
    sptr = nullptr;

    // shared_array<int32_t>(blockDim.y * blockDim.x, sptr, &shared_size);
    shared_array<int32_t>(griddim.x * nelements, sptr, &shared_size);

    parse_elemental_embedding_kernel_v2_2<<<griddim, blockDim, shared_size>>>(
        output_tmp.data_ptr<int32_t>(),
        output_nelements.data_ptr<int32_t>(),
        elemental_embedding.size(0),
        elemental_embedding.size(-1),
        output.data_ptr<int32_t>());

    cudaDeviceSynchronize();

    return {output_tmp, output, output_nelements.sum(0)};
}

TORCH_LIBRARY(embedding_tools, m)
{
    m.def("parse_elemental_embedding_v1", &parse_elemental_embedding_v1);
    m.def("parse_elemental_embedding_v2", &parse_elemental_embedding_v2);
}
