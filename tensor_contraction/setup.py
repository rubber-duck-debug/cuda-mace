import torch.cuda
from setuptools import setup, find_packages
from torch.utils.cpp_extension import CppExtension, CUDAExtension, BuildExtension
from torch.utils.cpp_extension import CUDA_HOME

ext_modules = []

__author__ = "Nicholas J. Browning"
__credits__ = "Nicholas J. Browning (2021), https:://TODO"
__license__ = "MIT"
__version__ = "0.1"
__maintainer__ = "Nicholas J. Browning"
__email__ = "nickjbrowning@gmail.com"
__status__ = "Alpha"
__description__ = "GPU-Accelerated Sparse Tensor Contraction"
__url__ = "TODO"

# optimisation_level_host = '-g'
# optimisation_level_device = '-G'

optimisation_level_host = '-O2'
optimisation_level_device = '-O2'


def readme():
    with open('README.md') as f:
        return f.read()


def requirements():
    with open('requirements.txt') as f:
        return [line.rstrip() for line in f]


if torch.cuda.is_available() and CUDA_HOME is not None:
    
    tensor_contraction = CUDAExtension(
        '.cuda.tensor_contraction', [
            'cuda/tensor_contraction.cpp',
            'cuda/tensor_contraction_kernels.cu'
        ],
         extra_compile_args={'cxx': [optimisation_level_host],
                            'nvcc': [optimisation_level_device]})

    symmetric_contraction = CUDAExtension(
        '.cuda.symmetric_contraction', [
            'cuda/symmetric_contraction_kernels.cu'
        ],
         extra_compile_args={'cxx': [optimisation_level_host],
                            'nvcc': [optimisation_level_device]})
    
    ext_modules.append(tensor_contraction)
    #ext_modules.append(symmetric_contraction)
    
else:
    print("ERROR: cuda not available, or CUDA_HOME not set.")
    exit()
    
setup(
    name='tensor_contraction',
    packages=['tests',
              'cuda'],
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
    
    ext_package='tensor_contraction',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension})
