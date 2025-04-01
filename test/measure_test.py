import qldpcdecoder
from qldpcdecoder.codes import gen_BB_code, gen_HP_ring_code,create_generalized_bicycle_codes,rep_code,ring_code,hypergraph_product,hamming_code,create_circulant_matrix
from qldpcdecoder.bpdecoders import BPOSD_decoder, BP_decoder
from qldpcdecoder.simulation.independentsim import measure_noise_simulation,independentnoise_simulation,measure_noise_simulation_parallel
from qldpcdecoder.simulation.bbcodesim import circuit_level_simulation
from qldpcdecoder.gauss_decoder import guass_decoder
from qldpcdecoder.decoupleddecoder import DecoupledDecoder
from functools import reduce
import numpy as np
from rich.pretty import pprint
np.random.seed(12561)
p = 0.005
# code1 = create_circulant_matrix(63,[0,3,34,41,57])
code1 = create_circulant_matrix(24,[0,2,8,15])
code2 = create_circulant_matrix(24,[0,2,8,15])
# code2 = create_circulant_matrix(63,[0,3,34,41,57])
css_code = hypergraph_product(code1, code2)
print(css_code.name)
print(css_code.N)
print(css_code.K)
print(css_code.D)
exit()
# css_code = gen_HP_ring_code(7,7)
# A = reduce(lambda x, y: x + y, css_code.A_list).toarray()
# with open("results/A.txt", "w") as file:
#     for row in A:
#         file.write(" ".join(map(str, row)) + "\n")
# l,m = A.shape

# B = reduce(lambda x, y: x + y, css_code.B_list).toarray()
# with open("results/B.txt", "w") as file:
#     for row in B:
#         file.write(" ".join(map(str, row)) + "\n")
# l,m = B.shape
# B_small = np.zeros((l//3,m//3),dtype=int)
# for i in range(0,l,3):
#     for j in range(0,m,3):
#         B_small[i//3][j//3] = B[i][j] 
# assert np.all(np.kron(B_small,np.identity(3))==B)
# with open("results/B_small.txt", "w") as file:
#     for row in B_small:
#         file.write(" ".join(map(str, row)) + "\n")
bposddecoder = BPOSD_decoder()

bpdecoder = BP_decoder()
gaussdecoder = guass_decoder()
reshapeddecoder = DecoupledDecoder(css_code,p)
# reshapedgaussdecoder = DecoupledDecoder(css_code,p,decoders_mode='greedy')
with open("results/results_independent_noise_sim.txt", "w") as file:
    file.write(measure_noise_simulation_parallel(css_code,p,[bposddecoder],num_trials=3000,num_repeat=5))

