'''
Created on 18 Mar 2022

@author: Nicholas J. Browning
'''

import numpy as np

nfeatures = 80
nirreps = 48
natoms = 1

#                                 b         c         i  
node_feautures = np.random.rand(natoms, nfeatures, nirreps)
#                     .         .        i          c
Uw = np.random.rand(48, 48, nirreps, nfeatures)

#                   b         c 
test2 = np.zeros((natoms, nfeatures, nirreps, nirreps))
test1 = np.einsum('...ic, bci-> bc...',
                               Uw,
                               node_feautures)

test3 = np.zeros((natoms, nfeatures, nirreps, nirreps))

for b in range(natoms):
    for c in range(nfeatures):
        for i in range(nirreps):
                v = node_feautures[b, c , i] * Uw[:,:, i, c]
            
                test2[b, c] += v
         
for b in range(natoms):
    for c in range(nfeatures):
        # [1, 1, nirreps] x nirreps, nirreps, nirreps
        v = (node_feautures[b, c ,:] * Uw[:,:,:, c])

        test3[b, c] += v.sum(-1)

print (np.linalg.norm(test3 - test1))
print (np.linalg.norm(test2 - test1))
