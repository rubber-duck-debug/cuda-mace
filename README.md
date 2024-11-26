
# Installation

```bash
git clone https://github.com/nickjbrowning/cuda_mace.git
cd cuda_mace
pip install . --no-build-isolation
```

# Requirements

Nvidia Ampere GPU or later (Compute Capability >= 8)

Pytorch

`high_perf` branch of MACE: https://github.com/ACEsuit/mace/tree/high_perf

CUDA SDK (https://developer.nvidia.com/cuda-downloads) installed, with `CUDA_HOME` set (typically to /usr/local/cuda).

# Model Transplants

## Invariant Models

```python
from cuda_mace.models import OptimizedInvariantMACE

model = OptimizedInvariantMACE(torch.load("model.pt").double())
```

A more detailed example can be found in `cuda_mace/examples/model_surgery.py`:

```python
python tools/model_surgery.py --model examples/SPICE_sm_inv_neut_E0_swa.model
```

The above surgery code will save the optimized model by default to: `./optimized_model.model`

## Equivariant Models

Not currently implemented

# License

This codebase is provided under an Academic Software License. See LICENSE for further details.
