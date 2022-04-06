import torch


def sparsifyU(U_tensor, num_irreps):
    sparseU_indices = torch.zeros(num_irreps, num_irreps, num_irreps, 16, device='cuda', dtype=torch.int).fill_(-1)
    sparseU_nvals = torch.zeros(num_irreps, num_irreps, num_irreps, device='cuda', dtype=torch.int)
    sparseU_vals = torch.zeros(num_irreps, num_irreps, num_irreps, 16, device='cuda')
    
    sparseUw_indices = torch.zeros(num_irreps, num_irreps, 6, device='cuda', dtype=torch.int).fill_(-1)
    sparseUw_nvals = torch.zeros(num_irreps, num_irreps, device='cuda', dtype=torch.int)
    
    sparseU2w_indices = torch.zeros(num_irreps, num_irreps, device='cuda', dtype=torch.int).fill_(-1)
    sparseU2w_nvals = torch.zeros(num_irreps, device='cuda', dtype=torch.int)
    
    for i in range(num_irreps):
        
        idx2 = []
        
        for j in range(num_irreps):
            count = 0
            idxs = []
            for k in range(num_irreps):
                (indexes,) = torch.where(U_tensor[3][i, j, k,:] != 0)
    
                if (indexes.shape[0] > 0):
                    # print (i, j, k, indexes)
                    sparseU_indices[i, j, k,:indexes.shape[0]] = indexes
                    sparseU_nvals[i, j, k] = indexes.shape[0]
                    sparseU_vals[i, j, k,:indexes.shape[0]] = U_tensor[3][i, j, k, indexes]
                    count += 1
                    idxs.append(k)
            
            sparseUw_indices[i, j,:count] = torch.tensor([idxs]).int().cuda()
            sparseUw_nvals[i, j] = count
            
            if (count > 0): 
                idx2.append(j)
        
        idx2 = torch.tensor(idx2).int().cuda()
        
        if (idx2.shape[0] > 0):
            sparseU2w_indices[i, 0:idx2.shape[0]] = idx2
            sparseU2w_nvals[i] = idx2.shape[0]
            
    return {'contraction_3_U': (sparseU_indices, sparseU_nvals, sparseU_vals),
            'contraction_3_Uw': (sparseUw_indices, sparseUw_nvals),
            'contraction_2': (sparseU2w_indices, sparseU2w_nvals)}
