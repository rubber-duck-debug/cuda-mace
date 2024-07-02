
## USE
```python
from cuda_mace.models import OptimizedScaleShiftInvariantMACE

model = OptimizedScaleShiftInvariantMACE(torch.load("model.pt").double())

```