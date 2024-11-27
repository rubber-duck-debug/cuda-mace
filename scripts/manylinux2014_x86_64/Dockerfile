# Use the official manylinux_2_28_x86_64 base image
FROM quay.io/pypa/manylinux_2_28_x86_64

# Set environment variable for CUDA version
ARG CUDA_VER="12-4"

# Install system dependencies and CUDA
RUN yum install -y yum-utils gcc gcc-c++ make zlib-devel bzip2-devel libffi-devel \
    && yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo \
    && yum install -y \
    cuda-compiler-${CUDA_VER} \
    cuda-libraries-${CUDA_VER} \
    cuda-libraries-devel-${CUDA_VER} \
    && yum clean all \
    && rm -rf /var/cache/yum/* \
    && echo "/usr/local/cuda/lib64" >> /etc/ld.so.conf.d/999_nvidia_cuda.conf

RUN /opt/python/cp311-cp311/bin/python3.11 -m ensurepip --upgrade

# Set environment variables for CUDA
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
ENV CUDA_HOME=/usr/local/cuda
ENV CUDA_ROOT=/usr/local/cuda
ENV CUDA_PATH=/usr/local/cuda
ENV CUDADIR=/usr/local/cuda

RUN dnf install git-all

# Verify the CUDA installation
RUN echo "CUDA_HOME: ${CUDA_HOME}" && \
    nvcc --version

RUN mkdir envs

RUN /opt/python/cp311-cp311/bin/pip install torch==2.4.0+cu124 --extra-index-url https://download.pytorch.org/whl/cu124
RUN /opt/python/cp311-cp311/bin/pip install cmake

# Clean up to reduce image size
RUN yum clean all && rm -rf /var/cache/yum/*

RUN mkdir /workspace

# Default command (bash)
CMD ["/bin/bash"]
