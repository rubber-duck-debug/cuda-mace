
from time import time
import torch
from e3nn import o3

from mace_ops.ops.symmetric_contraction import SymmetricContraction as CUDAContraction_
from mace.modules.symmetric_contraction import SymmetricContraction

nchannels = 96
max_ell = 3
correlation = 3
natoms = 5000
dtype = torch.float32
torch.set_default_dtype(dtype)

hidden_irreps=o3.Irreps(str(nchannels) + "x0e")

sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
num_features = hidden_irreps.count(o3.Irrep(0, 1))
interaction_irreps = (sh_irreps * num_features).sort()[0].simplify()

#need to pull the weights from the existing MACE SymmetricContraction module.
symm_contract = SymmetricContraction(interaction_irreps, hidden_irreps, correlation, num_elements=3).to("cuda")

for param in symm_contract.parameters():
    param.requires_grad = False

X = torch.rand((natoms, (max_ell+1) ** 2, nchannels), device='cuda', dtype=dtype, requires_grad=True)
Y = torch.rand((natoms, 3), device='cuda', dtype=dtype)
atom_types = Y.argmax(dim=-1).int()

coupling_irreps = o3.Irreps([irrep.ir for irrep in interaction_irreps])

#we include all contractions in the weights dictionary. The CUDA code will internally format these into the sparsity storage.
all_weights = {}
for i in range(len(symm_contract.contractions)):
    all_weights[str(i)] = {}
    all_weights[str(i)][3] =  symm_contract.contractions[i].weights_max.detach().clone().type(dtype)
    all_weights[str(i)][2] =  symm_contract.contractions[i].weights[0].detach().clone().type(dtype)
    all_weights[str(i)][1] =  symm_contract.contractions[i].weights[1].detach().clone().type(dtype)


cuda_contraction = CUDAContraction_(coupling_irreps, hidden_irreps, all_weights, nthreadX = 32, nthreadY = 4, nthreadZ = 1, dtype=dtype)

torch.cuda.synchronize()
torch.matmul(torch.randn(1024, 1024, device='cuda'),torch.randn(1024, 1024, device='cuda'))


ntrials = 1000

torch.cuda.cudart().cudaProfilerStart()
torch.cuda.synchronize()
start = time()
for i in range (ntrials):
    out_cuda = cuda_contraction.forward(X, atom_types)
    os = out_cuda.sum()
    os.backward()
torch.cuda.synchronize()
end = time()
print (end - start)
torch.cuda.cudart().cudaProfilerStop()

model = torch.jit.script(cuda_contraction)
print (model)
model.save('model.pt')
model = torch.jit.load('model.pt')
