
## Model Transplants

# Invariant Models
```python
from cuda_mace.models import OptimizedScaleShiftInvariantMACE

model = OptimizedScaleShiftInvariantMACE(torch.load("model.pt").double())
```

A more detailed example can be found in `cuda_mace/examples/model_surgery`:

```
python examples/model_surgery.py --model examples/SPICE_sm_inv_neut_E0_swa.model
```

# Equivariant Models

Not currently implemented