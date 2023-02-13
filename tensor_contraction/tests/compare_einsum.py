from typing import Dict

import torch
from tensor_contraction.cuda import  tensor_contraction

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

repeats = 50

torch.matmul(torch.rand(1024, 1024, dtype=torch.float, device='cuda'), torch.rand(1024, 1024, dtype=torch.float, device='cuda'))

A = torch.rand(16, 16, 48, 16, dtype=torch.float, device='cuda')
B = torch.rand(16, 96, dtype=torch.float, device='cuda')

start.record()
for i in range(repeats):
    out = tensor_contraction.get_u4w_matmul_tc16x16_f32(A, B)
end.record()
torch.cuda.synchronize()
print("u4w tensorcore time %.4f" % (start.elapsed_time(end) / repeats), "ms")

start.record()
for i in range(repeats):
    out = torch.einsum('...mk,kn->...mn', A, B)
end.record()
torch.cuda.synchronize()
print("einsum time %.4f" % (start.elapsed_time(end) / repeats), "ms")
