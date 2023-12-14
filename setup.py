from distutils.command.install_lib import install_lib as _install_lib
from setuptools import setup, find_packages
from torch.utils import cpp_extension
from torch.utils.cpp_extension import CppExtension, CUDAExtension, BuildExtension
from torch.utils.cpp_extension import CUDA_HOME
import os
import re
import torch

ext_modules = []

__author__ = "Nicholas J. Browning"
__credits__ = "Nicholas J. Browning (2023), https://github.com/nickjbrowning"
__license__ = "MIT"
__version__ = "0.1"
__maintainer__ = "Nicholas J. Browning"
__email__ = "nickjbrowning@gmail.com"
__status__ = "Alpha"
__description__ = "GPU-Accelerated Sparse Symmetric Contractions and Tensor Products"
__url__ = "TODO"

host_flags = []  # ['-O3']
debug_flags = []  # ['-G', '-lineinfo']
nvcc_flags = []  # ['-O3', '--use_fast_math']


def readme():
    with open('README.md') as f:
        return f.read()


def requirements():
    with open('requirements.txt') as f:
        return [line.rstrip() for line in f]


def batch_rename(src, dst, src_dir_fd=None, dst_dir_fd=None):
    '''Same as os.rename, but returns the renaming result.'''
    os.rename(src, dst,
              src_dir_fd=src_dir_fd,
              dst_dir_fd=dst_dir_fd)
    return dst


class _CommandInstallCythonized(_install_lib):
    def __init__(self, *args, **kwargs):
        _install_lib.__init__(self, *args, **kwargs)

    def install(self):
        # let the distutils' install_lib do the hard work
        outfiles = _install_lib.install(self)
        # batch rename the outfiles:
        # for each file, match string between
        # second last and last dot and trim it
        matcher = re.compile('\.([^.]+)\.so$')
        return [batch_rename(file, re.sub(matcher, '.so', file))
                for file in outfiles]


if torch.cuda.is_available() and CUDA_HOME is not None:

    equivariant_message_passing = CUDAExtension(
        '.cuda.equivariant_message_passing', [
            'mace_ops/cuda/equivariant_message_passing.cu'
        ],
        extra_compile_args={'cxx': host_flags,
                            'nvcc': nvcc_flags})

    
    invariant_message_passing = CUDAExtension(
        '.cuda.invariant_message_passing', [
            'mace_ops/cuda/invariant_message_passing.cu'
        ],
        extra_compile_args={'cxx': host_flags,
                            'nvcc': nvcc_flags})
    
    invariant_message_passing_old = CUDAExtension(
        '.cuda.invariant_message_passing_old', [
            'mace_ops/cuda/invariant_message_passing_old.cu'
        ],
        extra_compile_args={'cxx': host_flags,
                            'nvcc': nvcc_flags})

    symmetric_contraction = CUDAExtension(
        '.cuda.symmetric_contraction', [
            'mace_ops/cuda/symmetric_contraction_kernels.cu'
        ],
        extra_compile_args={'cxx': host_flags,
                            'nvcc': nvcc_flags})


    linear_wmma = CUDAExtension(
        '.cuda.linear_wmma', [
            'mace_ops/cuda/linear_wmma.cu'
        ],
        extra_compile_args={'cxx': host_flags,
                            'nvcc': nvcc_flags})
    
    
    print (cpp_extension.include_paths()+ ['/mace_ops/cuda/include'])
    embedding_tools = CUDAExtension(
        '.cuda.embedding_tools', [
            'mace_ops/cuda/embedding_tools.cu',
        ],
        extra_compile_args={'cxx': host_flags,
                            'nvcc': nvcc_flags})


    ext_modules.append(invariant_message_passing_old)
    ext_modules.append(invariant_message_passing)
    ext_modules.append(equivariant_message_passing)
    ext_modules.append(symmetric_contraction)
    #ext_modules.append(linear)
    ext_modules.append(linear_wmma)
    ext_modules.append(embedding_tools)

else:
    print("ERROR: cuda not available, or CUDA_HOME not set.")
    exit()

setup(
    name='mace ops',
    packages=['mace_ops.cuda', 'mace_ops.ops'],
    version=__version__,
    author=__author__,
    author_email=__email__,
    platforms='Any',
    description=__description__,
    long_description='',
    keywords=['Machine Learning'],
    classifiers=[],
    url=__url__,
    install_requires=requirements(),

    ext_package='mace_ops',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension,
              'install_lib': _CommandInstallCythonized
              })
