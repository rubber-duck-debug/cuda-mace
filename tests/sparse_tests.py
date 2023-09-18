import torch
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))

U  = torch.load(dir_path + '/sparsified_U_L=2.pt')


for irrep in range(U.shape[0]):
    count_irrep_1 = 0
    count_irrep_2 = 0
    count_irrep_3 = 0

    # i, j, nk
    # i, j, k_indices as int32
    # i, j, l_indices as int32

    total_sum = 0

    for i in range(U.shape[1]):

        for j in range (U.shape[2]):
                
                kdx1, ldx1  = torch.where(U[irrep][i, j] != 0.0)
                kdx2, ldx2  = torch.where(U[irrep][j, i, :, :] != 0.0)
                kdx3, ldx3  = torch.where(U[irrep][j, :, i, :] != 0.0)
                
                if (len(ldx1) !=0):
                    count_irrep_1 +=1

                if (len(ldx2) !=0):
                    count_irrep_2 +=1

                if (len(ldx3) !=0):
                    count_irrep_3 +=1

                if len(ldx1) > 0 or len(ldx2) > 0 or len(ldx3) > 0:
                    #print (irrep, i, j, "(", len(ldx1), len(ldx2), len(ldx3), ")")

                    #print (kdx1, kdx2, kdx3)
                    #print (ldx1, ldx2, ldx3)

                    total_sum += 1

                    

    print (count_irrep_1,count_irrep_2, count_irrep_3)
    print (total_sum)
#torch.Size([5, 16, 16, 16, 11])
print (U.shape)