
# Installation

```bash
git clone https://github.com/nickjbrowning/cuda_mace.git
cd cuda_mace
pip install . --no-build-isolation
```

# Requirements
Nvidia Ampere GPU or later (Compute Capability >= 8)

`high_perf` branch of MACE: https://github.com/ACEsuit/mace/tree/high_perf

# Model Transplants

## Invariant Models

```python
from cuda_mace.models import OptimizedInvariantMACE

model = OptimizedInvariantMACE(torch.load("model.pt").double())
```

To add nvtx ranges for profiling with nsight-sys, you can pass `profile=True` in the constructor of `OptimizedInvariantMACE`. Doing so will add ranges for key operations and provide a clearer view of computational cost on the nsight timeline.

A more detailed example can be found in `cuda_mace/examples/model_surgery.py`:

```python
python examples/model_surgery.py --model examples/SPICE_sm_inv_neut_E0_swa.model
```

The above surgery code will save the optimized model by default to: `./optimized_model.model`

## Equivariant Models

Not currently implemented

# License

This codebase is provided under an Academic Software License. See LICENSE for further details.
