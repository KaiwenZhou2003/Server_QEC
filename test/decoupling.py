import qldpcdecoder
from qldpcdecoder.codes import gen_BB_code, gen_HP_ring_code
from qldpcdecoder.bpdecoders import BPOSD_decoder, BP_decoder
from qldpcdecoder.simulation.independentsim import measure_noise_simulation,independentnoise_simulation
from qldpcdecoder.simulation.bbcodesim import circuit_level_simulation
from qldpcdecoder.gauss_decoder import guass_decoder
from qldpcdecoder.decoupleddecoder import ReShapeBBDecoder
from functools import reduce
import numpy as np
from rich.pretty import pprint
import os
from qldpcdecoder.decoupling.blockize import solve_flexible_block_transform

np.random.seed(1234561)

css_code = gen_BB_code(756)
pathdir = "results/"+css_code.name+"/"
if not os.path.exists(pathdir):
    os.makedirs(pathdir)
# css_code,_,_ = gen_HP_ring_code(7,7)
codehx = css_code.hx
lm = codehx.shape[0]
Apart = codehx[:,:lm]
Bpart = codehx[:,lm:]
with open(pathdir+"A_part.txt","w") as f:
    for row in Apart:
        f.write(" ".join(map(str,row))+"\n")
with open(pathdir+"B_part.txt","w") as f:
    for row in Bpart:
        f.write(" ".join(map(str,row))+"\n")

row_part = [lm//3]*3
col_part = [lm//3]*3
T,C,THC = solve_flexible_block_transform(Apart, row_part, col_part,equal_blocks=True)
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
    
        

T,C,THC = solve_flexible_block_transform(Bpart, row_part, col_part,equal_blocks=True)
if T is None:
    print("block transform failed")
else:
    np.save(pathdir+"B_T.npy",T)
    np.save(pathdir+"B_C.npy",C)
    np.save(pathdir+"B_THC.npy",THC)
    np.save(pathdir+"B_row_part.npy",row_part)
    np.save(pathdir+"B_col_part.npy",col_part)
    with open(pathdir+"B_THC.txt","w") as f:
        for row in THC:
            f.write(" ".join(map(str,row))+"\n")


