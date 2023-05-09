from e3nn import o3
import torch
from mace.tools import cg

def test_U_matrix():
    irreps_in = o3.Irreps("1x0e + 1x1o + 1x2e + 1x3o")
    irreps_out = o3.Irreps("1x1o")
    irreps_out = o3.Irreps("1x0e")
    u_matrix = cg.U_matrix_real(
        irreps_in=irreps_in, irreps_out=irreps_out, correlation=3
    )[-1]
    return u_matrix

u_matrix = test_U_matrix()

print (torch.count_nonzero(u_matrix), u_matrix.numel())

for i in range (u_matrix.shape[0]):
    for j in range (u_matrix.shape[1]):
        for k in range (u_matrix.shape[2]):
            ldx, mdx = torch.where(u_matrix[i,j,k] !=0.0)

            #print (ldx, mdx)