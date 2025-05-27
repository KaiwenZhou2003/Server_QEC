
import numpy as np
import os
from qldpcdecoder.decoupling.blockize import solve_flexible_block_transform

np.random.seed(1234561)


pathdir = "results/"+"BB72right"+"/"
if not os.path.exists(pathdir):
    os.makedirs(pathdir)
# css_code,_,_ = gen_HP_ring_code(7,7)
codehx = np.loadtxt('test/circuit_level_decouple.txt',dtype=int)
print(codehx.shape)
col_part = [36]*3
row_part = [12]*3
T,C,THC = solve_flexible_block_transform(codehx, row_part, col_part,equal_blocks=False)
if T is None:
    print("block transform failed")
else:
    print("block transform succeeded")
    np.save(pathdir+"A_T.npy",T)
    np.save(pathdir+"A_C.npy",C)
    np.save(pathdir+"A_THC.npy",THC)
    np.save(pathdir+"A_row_part.npy",row_part)
    np.save(pathdir+"A_col_part.npy",col_part)
    with open(pathdir+"A_THC.txt","w") as f:
        for row in THC:
            f.write(" ".join(map(str,row))+"\n")
    
        