from setuptools import setup, find_packages
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

host_flags = ['-O3']
#device_flags = ['-G', '-lineinfo']
nvcc_flags = ['-O3', '--use_fast_math']

def readme():
    with open('README.md') as f:
        return f.read()

def requirements():
    with open('requirements.txt') as f:
        return [line.rstrip() for line in f]


from distutils.command.install_lib import install_lib as _install_lib

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
    
    tensor_contraction = CUDAExtension(
        '.cuda.tensor_product', [
            'mace_ops/cuda/tensor_product_kernels.cu'
        ],
         extra_compile_args={'cxx': host_flags,
                            'nvcc': nvcc_flags})

    symmetric_contraction = CUDAExtension(
        '.cuda.symmetric_contraction', [
            'mace_ops/cuda/symmetric_contraction_kernels.cu'
        ],
         extra_compile_args={'cxx': host_flags,
                            'nvcc': nvcc_flags})
    
    ext_modules.append(tensor_contraction)
    ext_modules.append(symmetric_contraction)
    
else:
    print("ERROR: cuda not available, or CUDA_HOME not set.")
    exit()
    
setup(
    name='mace ops',
    packages=['mace_ops.cuda'],
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
