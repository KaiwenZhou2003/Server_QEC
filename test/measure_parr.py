import qldpcdecoder
from qldpcdecoder.codes import gen_BB_code, gen_HP_ring_code,create_generalized_bicycle_codes,create_QC_GHP_codes,ring_code,hypergraph_product,hamming_code,create_circulant_matrix
from qldpcdecoder.bpdecoders import BPOSD_decoder, BP_decoder
from qldpcdecoder.simulation.independentsim import measure_noise_simulation,independentnoise_simulation,measure_noise_simulation_parallel
from qldpcdecoder.simulation.bbcodesim import circuit_level_simulation
from qldpcdecoder.sparsegauss_decoder import guass_decoder
from qldpcdecoder.decoupleddecoder import DecoupledDecoder
from functools import reduce
import numpy as np
from rich.pretty import pprint
np.random.seed(12561)
p = 0.005
############################# GP codes #####################################
## [[72,8,4]]
# code1 = create_circulant_matrix(6,[1,2,3])
# code2 = create_circulant_matrix(6,[1,2,3])
## [[162,8,6]]
# code1 = create_circulant_matrix(9,[1,2,3])
# code2 = create_circulant_matrix(9,[1,2,3])
## [[288,12,6]]
# code1 = create_circulant_matrix(12,[0,1,2,3])
# code2 = create_circulant_matrix(12,[0,3,2,7])
## [[784,8,12]]
# code1 = create_circulant_matrix(28,[0,26,9,20])
# code2 = create_circulant_matrix(14,[0,6,7,8])
# [[744,20,6]]
# code1 = create_circulant_matrix(31,[0,2,5])
# code2 = create_circulant_matrix(12,[0,3,2,7])
## [[1488,30,7]]
# code1 = create_circulant_matrix(31,[0,2,5])
# code2 = create_circulant_matrix(24,[0,2,8,15])
# css_code = hypergraph_product(code1, code2)
#################################### GHP codes ########################################
## [[882,48,8]]
# A = 28*create_circulant_matrix(7,[0]) +1* create_circulant_matrix(7,[4])+ 19* create_circulant_matrix(7,[3]) + 28* create_circulant_matrix(7,[2])+1*create_circulant_matrix(7,[1])
# A -= np.ones((7,7),dtype=int)
# print(A)
# css_code = create_QC_GHP_codes(63,A,[0,1,6])
## [[882,24,6]]
# A_lists = [(27,0),(54,1),(0,2)]
# A = sum([(num+1)*create_circulant_matrix(7,[idx]) for num,idx in A_lists]) - np.ones((7,7),dtype=int)
# css_code = create_QC_GHP_codes(63,A,[0,1,6])
css_code = gen_HP_ring_code(9,9)   # comment: 
print(css_code.name)
print(css_code.N)
print(css_code.K)
print(css_code.D)
bposddecoder = BPOSD_decoder()

bpdecoder = BP_decoder()
gaussdecoder = guass_decoder()
# reshapeddecoder = DecoupledDecoder(css_code,p)
reshapedgaussdecoder = DecoupledDecoder(css_code,p)
pprint(measure_noise_simulation_parallel(css_code,p,[gaussdecoder,reshapedgaussdecoder],num_trials=3000,num_repeat=5)[0])

