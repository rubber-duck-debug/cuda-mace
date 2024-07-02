
# Installation

```bash
git clone https://github.com/nickjbrowning/cuda_mace.git
cd cuda_mace
pip install . --no-build-isolation
```

# Requirements
Nvidia Ampere GPU or later (Compute Capability >= 8)

# Model Transplants

## Invariant Models

### ScaleShift
```python
from cuda_mace.models import OptimizedScaleShiftInvariantMACE

model = OptimizedScaleShiftInvariantMACE(torch.load("model.pt").double())
```

A more detailed example can be found in `cuda_mace/examples/model_surgery.py`:

```python
python examples/model_surgery.py --model examples/SPICE_sm_inv_neut_E0_swa.model
```

## Equivariant Models

Not currently implemented