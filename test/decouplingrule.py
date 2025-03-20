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

def decouping(N):
    assert N in [72,144,288,756]
    css_code = gen_BB_code(N)
    pathdir = "results/"+css_code.name+"/"
    if not os.path.exists(pathdir):
        os.makedirs(pathdir)
    # css_code,_,_ = gen_HP_ring_code(7,7)
    codehx = css_code.hx
    lm = codehx.shape[0]
    A = codehx[:,:lm]
    B = codehx[:,lm:]

    row_part = [lm//3]*3
    col_part = [lm//3]*3
    with open(pathdir+"A_part.txt","w") as f:
        for row in A:
            f.write(" ".join(map(str,row))+"\n")
    with open(pathdir+"B_part.txt","w") as f:
        for row in B:
            f.write(" ".join(map(str,row))+"\n")
    A_small = np.zeros((lm//3,lm//3),dtype=int)

    T = np.zeros((lm,lm),dtype=int)
    C = np.zeros((lm,lm),dtype=int)
    if lm == 36:
        unit = 6
    else:
        unit = lm//12

    for i in range(0,lm//3):
        for j in range(0,lm//3):
            A_small[i][j] += A[3*unit*(i//unit)+(i%unit)][3*unit*(j//unit)+(j%unit)]
    for i in range(lm):
        new_row = (i%(3*unit))//unit*(lm//3) + i//(3*unit)*unit + (i%unit)
        T[new_row][i] = 1
    for j in range(lm):
        new_col = (j%(3*unit))//unit*(lm//3) + j//(3*unit)*unit + (j%unit)
        C[j][new_col] = 1
    THC = T@A@C
    assert np.all(np.kron(np.identity(3),A_small)==THC)
    with open(pathdir+"A_small.txt","w") as f:
        for row in A_small:
            f.write(" ".join(map(str,row))+"\n")

    np.save(pathdir+"A_T.npy",T)
    np.save(pathdir+"A_C.npy",C)
    np.save(pathdir+"A_THC.npy",THC)
    np.save(pathdir+"A_row_part.npy",row_part)
    np.save(pathdir+"A_col_part.npy",col_part)
    with open(pathdir+"A_THC.txt","w") as f:
        for row in THC:
            f.write(" ".join(map(str,row))+"\n")
        
            
    l,m = B.shape
    B_small = np.zeros((l//3,m//3),dtype=int)
    for i in range(0,l,3):
        for j in range(0,m,3):
            B_small[i//3][j//3] = B[i][j] 
    assert np.all(np.kron(B_small,np.identity(3))==B)
    with open(pathdir+"B_small.txt", "w") as file:
        for row in B_small:
            file.write(" ".join(map(str, row)) + "\n")
    T = np.zeros((l,l),dtype=int)
    C = np.zeros((m,m),dtype=int)
    for i in range(l):
        new_row = (i%3)*l//3 + i//3
        T[new_row][i] = 1
    for j in range(m):
        new_col = (j%3)*m//3 + j//3
        C[j][new_col] = 1
    THC = T@B@C
    assert np.all(np.kron(np.identity(3),B_small)==THC)
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


    print("block transform succeeded")
    
if __name__ == "__main__":
    for N in [72,144,288]:
        decouping(N)