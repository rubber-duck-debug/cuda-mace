import numpy as np


L = 0
lstart = L * L
NNODES=1000
nl = 2 * L + 1
NL_TOT = NNODES * nl
blocks = int(np.ceil(NL_TOT / 16))

print (blocks)

nthreadY = 4
nthreadX = 32

for i in range (blocks):
    print (f"block: {i}")
    
    for offset in range (4):
        for j in range (nthreadY):
            gid = i * 16 + offset * 4 + j
            lidx = lstart + (int((gid / nl)) * 16) + gid % nl
            print (f"offset: {offset} threadRow: {j}: {offset*4 + j}, gid%nl: {gid %nl} gid: {gid} lidx: {lidx} node: {int(lidx/16)}")
        
        