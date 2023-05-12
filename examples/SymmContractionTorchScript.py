import numpy as np
from time import time
import torch
import logging
import traceback
from mace_ops.cuda import SymmetricContraction
from torch.profiler import profile, record_function, ProfilerActivity

nchannels = 128

X = np.fromfile('symm_contraction_data/X.npy').reshape(21, 128, 16)[:, :nchannels, :]

Y = np.fromfile('symm_contraction_data/Y.npy').reshape(21, 3)

U3 = np.fromfile('symm_contraction_data/U_3.npy').reshape(16,16,16,23)
U2 = np.fromfile('symm_contraction_data/U_2.npy').reshape(16,16, 4)
U1 = np.fromfile('symm_contraction_data/U_1.npy').reshape(16, 1)

U3 = torch.from_numpy(U3).float().cuda()
U2 = torch.from_numpy(U2).float().cuda()
U1 = torch.from_numpy(U1).float().cuda()

W3 = np.fromfile('symm_contraction_data/W_3.npy').reshape(3,23,128)[:, :, :nchannels]

W2 = np.fromfile('symm_contraction_data/W_2.npy').reshape(3,4, 128)[:, :, :nchannels]

W1 = np.fromfile('symm_contraction_data/W_1.npy').reshape(3,1, 128)[:, :, :nchannels]


W3 = torch.from_numpy(W3).float().cuda()

W2 = torch.from_numpy(W2).float().cuda()

W1 = torch.from_numpy(W1).float().cuda()
        
correlation = 3

U_tensors = {"3": U3, "2":  U2, "1": U1}
W_tensors = {"3": W3, "2": W2, "1": W1}

nrepeats = int((5000.0) / 21.0)

X = torch.from_numpy(X).float().cuda().repeat(nrepeats, 1, 1)
Y = torch.from_numpy(Y).float().cuda().repeat(nrepeats, 1)

X_torch = X.transpose(-1, -2).contiguous()
atom_types = np.array([1,1,1,1,1,1,1,2,2,2,1,1,2,0,0,0,0,0,0,0,0])
atom_types_torch = torch.from_numpy(atom_types).repeat(nrepeats).int().cuda()

symm_contract = SymmetricContraction(U_tensors, W_tensors, device='cuda', dtype=torch.float32)

script = torch.jit.script(symm_contract)



start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)


X_torch.requires_grad = False

start.record()

for i in range (1000):
    out = script(X_torch, atom_types_torch)

end.record()

torch.cuda.synchronize()

elapsed = start.elapsed_time(end)

print ("FWD_ONLY:", elapsed)


X_torch.requires_grad = True

start.record()

for i in range (1000):

    out = script(X_torch, atom_types_torch)

    os = out.sum()

    os.backward()

end.record()

torch.cuda.synchronize()

elapsed = start.elapsed_time(end)

print ("FWD + BWD:", elapsed)

print (out.shape)
print (out)

from torch.profiler import schedule

my_schedule = schedule(
    skip_first=10,
    wait=5,
    warmup=1,
    active=3,
    repeat=2)


X_torch.requires_grad=False

def trace_handler(p):
    output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
    print(output)
    p.export_chrome_trace("/tmp/trace_" + str(p.step_num) + ".json")

with profile(
    activities=[ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(
        wait=1,
        warmup=1,
        active=1),
    on_trace_ready=trace_handler
) as p:
    for idx in range(3):
        out = script(X_torch, atom_types_torch)
        p.step()

    print(p.key_averages().table(sort_by="cuda_time_total", row_limit=10))

X_torch.requires_grad=True

def trace_handler(p):
    output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
    print(output)
    p.export_chrome_trace("/tmp/trace_" + str(p.step_num) + ".json")

with profile(
    activities=[ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(
        wait=1,
        warmup=1,
        active=1),
    on_trace_ready=trace_handler
) as p:
    for idx in range(3):
        out = script(X_torch, atom_types_torch)

        os = out.sum()

        os.backward()
        p.step()

    print(p.key_averages().table(sort_by="cuda_time_total", row_limit=10))


