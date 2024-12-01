cmake_minimum_required(VERSION 3.10)
project(cuda_mace LANGUAGES CXX)

include(${CMAKE_SOURCE_DIR}/cmake/MakeIncludeable.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/PrependHeadersToSource.cmake)

set(LIB_INSTALL_DIR "lib" CACHE PATH "Path relative to CMAKE_INSTALL_PREFIX where to install libraries")
set(BIN_INSTALL_DIR "bin" CACHE PATH "Path relative to CMAKE_INSTALL_PREFIX where to install DLL/binaries")
set(INCLUDE_INSTALL_DIR "include" CACHE PATH "Path relative to CMAKE_INSTALL_PREFIX where to install headers")

find_package(Python COMPONENTS Interpreter REQUIRED)

execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import torch.utils; print(torch.utils.cmake_prefix_path)"
    RESULT_VARIABLE TORCH_CMAKE_PATH_RESULT
    OUTPUT_VARIABLE TORCH_CMAKE_PATH_OUTPUT
    ERROR_VARIABLE TORCH_CMAKE_PATH_ERROR
)

if (NOT ${TORCH_CMAKE_PATH_RESULT} EQUAL 0)
message(FATAL_ERROR "failed to find your pytorch installation, error: ${TORCH_CMAKE_PATH_ERROR}\n")
endif()

string(STRIP ${TORCH_CMAKE_PATH_OUTPUT} TORCH_CMAKE_PATH_OUTPUT)
set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};${TORCH_CMAKE_PATH_OUTPUT}")

find_package(Torch 1.13 REQUIRED)

add_library(cuda_mace SHARED
    "cuda/include/cuda_utils.hpp"

    "jit_wrappers/include/cuda_cache.hpp"
    "jit_wrappers/include/dynamic_cuda.hpp"
    
    "jit_wrappers/include/cubic_spline_wrapper.hpp"
    "jit_wrappers/include/invariant_message_passing_wrapper.hpp"
    "jit_wrappers/include/spherical_harmonics_wrapper.hpp"
    "jit_wrappers/include/linear_wrapper.hpp"
    "jit_wrappers/include/symmetric_contraction_wrapper.hpp"

    "jit_wrappers/src/cubic_spline_wrapper.cpp"
    "jit_wrappers/src/invariant_message_passing_wrapper.cpp"
    "jit_wrappers/src/spherical_harmonics_wrapper.cpp"
    "jit_wrappers/src/linear_wrapper.cpp"
    "jit_wrappers/src/symmetric_contraction_wrapper.cpp"

    "src/linear.cpp"
    "src/invariant_message_passing.cpp"
    "src/symmetric_contraction.cpp"
    "src/cubic_spline.cpp"
    "src/spherical_harmonics.cpp"
)

set(CUDA_MACE_SHARED_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/cuda/include/cuda_utils.hpp")

set(cuda_sources 
    "cubic_spline_impl.cu" 
    "invariant_message_passing_impl.cu" 
    "spherical_harmonics_impl.cu" 
    "linear_impl.cu"
    "symmetric_contraction_impl.cu"
)

set(cuda_headers 
    "cubic_spline_impl.cuh" 
    "invariant_message_passing_impl.cuh" 
    "spherical_harmonics_impl.cuh" 
    "linear_impl.cuh"
    "symmetric_contraction_impl.cuh"
)

list(LENGTH cuda_sources len_list)
math(EXPR len "${len_list} -1")

# Enumerate through both lists
foreach(i RANGE 0 ${len})
    list(GET cuda_sources ${i} file)
    list(GET cuda_headers ${i} header)

    set (tmp_src file)
    set(tmp_header "${CMAKE_CURRENT_SOURCE_DIR}/cuda/include/${header}")

    list (APPEND tmp_header ${CUDA_MACE_SHARED_HEADERS})

    prepend_headers_to_source(
        "${CMAKE_CURRENT_SOURCE_DIR}/cuda/src/${file}"
        "${CMAKE_CURRENT_BINARY_DIR}/generated/tmp.cu" 
        "${tmp_header}"
        )

    make_includeable(
        "${CMAKE_CURRENT_BINARY_DIR}/generated/tmp.cu"
        "${CMAKE_CURRENT_BINARY_DIR}/generated/wrapped_${file}"
        )
    
    message(STATUS "Processing file: ${CMAKE_CURRENT_SOURCE_DIR}/cuda/src/${file} with headers: ${tmp_header}")
    # Add your logic here
endforeach()

target_include_directories(cuda_mace PRIVATE
    ${CMAKE_SOURCE_DIR}/cuda/include
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/jit_wrappers/include
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

target_link_libraries(cuda_mace PRIVATE "${TORCH_LIBRARIES}" ${CMAKE_DL_LIBS})

target_compile_features(cuda_mace PRIVATE cxx_std_17)

#set_target_properties(cuda_mace PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
#set_target_properties(cuda_mace PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
#set_target_properties(cuda_mace PROPERTIES CUDA_ARCHITECTURES "80;86;89;90")

install(TARGETS cuda_mace
    LIBRARY DESTINATION "lib"
)