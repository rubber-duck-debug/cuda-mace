import torch
from torch.profiler import profile, record_function, ProfilerActivity

X = torch.rand(1000, 16, 128, dtype=torch.float32).cuda()
W = torch.rand(128, 128, dtype=torch.float32).cuda()

torch.matmul(X, W)


torch.matmul(X, W)


with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
    with record_function("model_inference"):
        out = torch.matmul(X, W)


print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))