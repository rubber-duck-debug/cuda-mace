from typing import Dict

import torch
from tensor_contraction.cuda import  tensor_contraction

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

torch.matmul(torch.rand(1024, 1024, dtype=torch.float, device='cuda'), torch.rand(1024, 1024, dtype=torch.float, device='cuda'))

A = torch.rand(48, 48, dtype=torch.float, device='cuda')
B = torch.rand(48, 48, dtype=torch.float, device='cuda')

start.record()
C = torch.matmul(A, B)
end.record()
torch.cuda.synchronize()
print("torch matmul", start.elapsed_time(end), "ms")

print (C)

start.record()
c_cuda_1 = tensor_contraction.get_multiwarp_matmul(A, B, 1)
end.record()
torch.cuda.synchronize()
print("cuda matmul 1 warp", start.elapsed_time(end), "ms")

print (c_cuda_1)

start.record()
c_cuda_3 = tensor_contraction.get_multiwarp_matmul(A, B, 6)
end.record()
torch.cuda.synchronize()
print("cuda matmul 3 warps", start.elapsed_time(end), "ms")

print (c_cuda_3)

