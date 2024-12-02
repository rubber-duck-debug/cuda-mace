#include "linear_wrapper.hpp"
#include "cuda_cache.hpp"
#include "cuda_utils.hpp"
#include <iostream>

using namespace c10;
using namespace std;
using namespace torch::autograd;
using namespace torch::indexing;

#define FULL_MASK 0xffffffff
#define WARP_SIZE 32
#define NWARPS_PER_BLOCK 4

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 8

#define M_BATCH 16
#define N_BATCH 32
#define K_BATCH 32

torch::Tensor jit_linear(torch::Tensor X, torch::Tensor W) {
  static const char* CUDA_CODE =
#include "generated/wrapped_linear_impl.cu"
        ;

    int NNODES = X.size(0);
    int M = X.size(1);
    int N = W.size(2);
    int K = W.size(1);

    torch::Tensor output =
    torch::empty({NNODES, M, N}, torch::TensorOptions().dtype(X.dtype()).device(X.device()));

    auto kernel_name = [&]() -> std::string {
        if (N >= 128) return getKernelName<std::integral_constant<int, 8>>("linear_kernel_ptr");
        if (N == 96)  return getKernelName<std::integral_constant<int, 6>>("linear_kernel_ptr");
        if (N == 64)  return getKernelName<std::integral_constant<int, 4>>("linear_kernel_ptr");
        if (N == 32)  return getKernelName<std::integral_constant<int, 2>>("linear_kernel_ptr");
        return ""; // Default case
    }();

    auto& kernel_factory = KernelFactory::instance();

    CachedKernel* kernel = kernel_factory.create(
        kernel_name, std::string(CUDA_CODE), "wrapped_linear_impl.cu", {}
    );
    
    dim3 bdim(WARP_SIZE, N/WMMA_N);

    vector<CUstream> streams;

    unsigned int space = 0;
    void *sptr;

    shared_array<float>(K_BATCH * (K_BATCH + 1), sptr, &space);
    shared_array<float>(M * bdim.y * WMMA_N, sptr, &space);


      for (int l = 0; l < 4; l++) {

        dim3 gdim;
        gdim.x = find_integer_divisor(NNODES * (2 * l + 1), 16);
        gdim.y = find_integer_divisor(N, bdim.y * WMMA_N);

        float * _X = X.data_ptr<float>();
        float * _W = W.data_ptr<float>();
        int thisl = l;
        float * _output = output.data_ptr<float>();

        CUstream stream;
        CUDADRIVER_SAFE_CALL(CUDA_DRIVER_INSTANCE.cuStreamCreate(&stream, CU_STREAM_DEFAULT));
        streams.push_back(stream);

        std::vector<void*> args = {
        &_X,
        &_W,
        &_output,
        &NNODES,
        &M, 
        &N, 
        &K, 
        &thisl
        };

        kernel->launch(gdim, bdim, space, stream, args);
  }

  for (int l = 0; l < streams.size(); l++) {
    CUDADRIVER_SAFE_CALL(CUDA_DRIVER_INSTANCE.cuStreamDestroy(streams[l]));
  }

  return output;
}

torch::Tensor jit_elemental_linear(torch::Tensor X, torch::Tensor W,
                                    torch::Tensor elemental_embedding) {

  static const char* CUDA_CODE =
#include "generated/wrapped_linear_impl.cu"
        ;

    int NNODES = X.size(0);
    int M = X.size(1);
    int N = W.size(-1);
    int K = W.size(2);

    torch::Tensor output =
    torch::empty({NNODES, M, N}, torch::TensorOptions().dtype(X.dtype()).device(X.device()));

    auto kernel_name = [&]() -> std::string {
        if (N >= 128) return getKernelName<std::integral_constant<int, 8>>("elemental_linear_kernel_ptr");
        if (N == 96)  return getKernelName<std::integral_constant<int, 6>>("elemental_linear_kernel_ptr");
        if (N == 64)  return getKernelName<std::integral_constant<int, 4>>("elemental_linear_kernel_ptr");
        if (N == 32)  return getKernelName<std::integral_constant<int, 2>>("elemental_linear_kernel_ptr");
        return ""; // Default case
    }();

    auto& kernel_factory = KernelFactory::instance();

    CachedKernel* kernel = kernel_factory.create(
        kernel_name, std::string(CUDA_CODE), "wrapped_linear_impl.cu", {}
    );
    
    dim3 bdim(WARP_SIZE, N/WMMA_N);

    vector<CUstream> streams;

    unsigned int space = 0;
    void *sptr;

    shared_array<float>(K_BATCH * (K_BATCH + 1), sptr, &space);
    shared_array<float>(M * bdim.y * WMMA_N, sptr, &space);


    for (int element_id = 0; element_id < elemental_embedding.size(1);
       element_id++) {
    torch::Tensor elemental_embedding_ =
        elemental_embedding.index({Ellipsis, element_id});
    torch::Tensor node_idx = torch::where(elemental_embedding_ == 1)[0];

    int _nselected = node_idx.size(0);

    if (_nselected > 0) {

      for (int l = 0; l < 4; l++) {

        dim3 gdim;
        gdim.x = find_integer_divisor(_nselected * (2 * l + 1), 16);
        gdim.y = find_integer_divisor(N, bdim.y * WMMA_N);

        float * _X = X.data_ptr<float>();
        float * _W = W.data_ptr<float>();
        long * _node_idx = node_idx.data_ptr<long>();
        int _nelements = elemental_embedding.size(-1);
        int _element_id = element_id;
        int thisl = l;
        float * _output = output.data_ptr<float>();

        CUstream stream;
        CUDADRIVER_SAFE_CALL(CUDA_DRIVER_INSTANCE.cuStreamCreate(&stream, CU_STREAM_DEFAULT));
        streams.push_back(stream);

        std::vector<void*> args = {
        &_X,
        &_W,
        &_node_idx,
        &_nselected,
        &_element_id, 
        &_nelements,
        &_output,
        &NNODES,
        &M, 
        &N, 
        &K, 
        &thisl
        };

        kernel->launch(gdim, bdim, space, stream, args);

      }
    }
  }

  for (int l = 0; l < streams.size(); l++) {
    CUDADRIVER_SAFE_CALL(CUDA_DRIVER_INSTANCE.cuStreamDestroy(streams[l]));
  }

  return output;
}