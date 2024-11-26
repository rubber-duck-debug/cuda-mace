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

# Install Python 3.11 (if available from the repository)
RUN yum install -y python3.11 python3.11-devel

RUN ln -s /usr/bin/python3 /usr/bin/python

# Install pip for Python 3.11
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py \
    && python get-pip.py \
    && rm get-pip.py

RUN pip install virtualenv

# Set environment variables for CUDA
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
ENV CUDA_HOME=/usr/local/cuda
ENV CUDA_ROOT=/usr/local/cuda
ENV CUDA_PATH=/usr/local/cuda
ENV CUDADIR=/usr/local/cuda

# Verify the CUDA installation
RUN echo "CUDA_HOME: ${CUDA_HOME}" && \
    nvcc --version

# Clean up to reduce image size
RUN yum clean all && rm -rf /var/cache/yum/*

# Default command (bash)
CMD ["/bin/bash"]
