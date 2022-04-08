from typing import Dict

import torch
from tensor_contraction.cuda import  tensor_contraction

A = torch.rand(48, 48, dtype=torch.float, device='cuda')
B = torch.rand(48, 48, dtype=torch.float, device='cuda')

C = torch.matmul(A, B)

c_cuda = tensor_contraction.get_multiwarp_matmul(A, B)

print (C)
print (c_cuda)

