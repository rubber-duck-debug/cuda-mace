
# CUDA-MACE

CUDA-MACE is an optimized implementation of the MACE (Multi-Atomic Cluster Expansion) model that leverages custom CUDA code to significantly improve performance.

## Features

- **GPU-Accelerated Performance**: Custom CUDA kernels optimized for modern NVIDIA GPUs (Ampere architecture and later)
- **Model Transplantation**: Convert existing MACE models to optimized CUDA implementations
- **Memory Efficiency**: Optimized memory usage patterns for large-scale molecular simulations
- **High Precision**: Maintains numerical accuracy typically required for force evaluations

## System Requirements

### Hardware Requirements
- **GPU**: NVIDIA Ampere GPU or later (Compute Capability >= 8.0)
  - Recommended: RTX 30 series, RTX 40 series, A100, H100, or equivalent

### Software Prerequisites
- **CUDA Toolkit**: Version 11.0 or later
  - Download from [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads)
  - Ensure `CUDA_HOME` environment variable is set (typically `/usr/local/cuda`)
- **PyTorch**: Compatible version with CUDA support
- **Python**: 3.10 or later

### Dependencies
- MACE 0.3.14
- NumPy
- SciPy
- ASE (Atomic Simulation Environment)

## Installation

### Quick Install
```bash
git clone https://github.com/rubber-duck-debug/cuda-mace.git
cd cuda-mace
pip install . --no-build-isolation
```

### Development Install
```bash
git clone https://github.com/rubber-duck-debug/cuda-mace.git
cd cuda-mace
pip install -e . --no-build-isolation
```
## Quick Start

### Model Optimization

Transform your existing MACE models:

```python
import torch
from cuda_mace.models import OptimizedInvariantMACE
from copy import deepcopy

# Load your trained MACE model
original_model = torch.load("your_model.pt")

# Create optimized CUDA version
optimized_model = OptimizedInvariantMACE(deepcopy(original_model.double()))
```

### Command-Line Model Surgery

Use the provided tool to convert models:

```bash
python tools/model_surgery.py --model path/to/your/model.pt --output optimized_model.pt [--accuracy] [--benchmark]
```

This will create an optimized model at `optimized_model.pt` ready for use.

## Current Limitations

- **Invariant Models Only**: Equivariant models are not currently supported
- **CUDA Compatibility**: Requires NVIDIA GPUs with Compute Capability >= 8.0

## Citation

If you use CUDA-MACE in your research, please cite:

```bibtex
@software{cuda_mace,
  title={CUDA-MACE: High-Performance GPU-Accelerated MACE Implementation},
  url={https://github.com/rubber-duck-debug/cuda_mace},
  year={2024}
}

@article{doi:10.1021/jacs.4c07099,
author = {Kovács, Dávid P{\'e}ter and Moore, J. Harry and Browning, Nicholas J. and Batatia, Ilyes and Horton, Joshua T. and Pu, Yixuan and Kapil, Venkat and Witt, William C. and Magdău, Ioan-Bogdan and Cole, Daniel J. and Csányi, Gábor},
title = {MACE-OFF: Short-Range Transferable Machine Learning Force Fields for Organic Molecules},
journal = {Journal of the American Chemical Society},
volume = {147},
number = {21},
pages = {17598-17611},
year = {2025},
doi = {10.1021/jacs.4c07099},
note ={PMID: 40387214},
URL = {https://doi.org/10.1021/jacs.4c07099},
eprint = {https://doi.org/10.1021/jacs.4c07099}
}

```

## License

This project is licensed under an Academic Software License. See [LICENSE](LICENSE) for full terms and conditions.
