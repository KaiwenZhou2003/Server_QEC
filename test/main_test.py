from qldpcdecoder.codes import gen_BB_code, gen_HP_ring_code
from bposd.hgp import hgp
from qldpcdecoder.bpdecoders import BPOSD_decoder, BP_decoder
from qldpcdecoder.simulation.independentsim import independentnoise_simulation
from qldpcdecoder.gauss_decoder import guass_decoder
from qldpcdecoder.decoupleddecoder import ReShapeBBDecoder
from functools import reduce
import numpy as np
p = 0.01
css_code = gen_BB_code(108)
A = reduce(lambda x, y: x + y, css_code.A_list).toarray()
with open("results/A.txt", "w") as file:
    for row in A:
        file.write(" ".join(map(str, row)) + "\n")
l,m = A.shape

B = reduce(lambda x, y: x + y, css_code.B_list).toarray()
with open("results/B.txt", "w") as file:
    for row in B:
        file.write(" ".join(map(str, row)) + "\n")
l,m = B.shape
B_small = np.zeros((l//3,m//3),dtype=int)
for i in range(0,l,3):
    for j in range(0,m,3):
        B_small[i//3][j//3] = B[i][j] 
assert np.all(np.kron(B_small,np.identity(3))==B)
with open("results/B_small.txt", "w") as file:
    for row in B_small:
        file.write(" ".join(map(str, row)) + "\n")
exit()
bposddecoder = BPOSD_decoder(css_code.hx,p)

bpdecoder = BP_decoder(css_code.hx,p)
gaussdecoder = guass_decoder(css_code.hx,p)

reshapeddecoder = ReShapeBBDecoder(css_code.hx,p,css_code.lz,A,B)

independentnoise_simulation(5000,css_code.hx,css_code.lz,p,[bposddecoder,bpdecoder,gaussdecoder,reshapeddecoder])

