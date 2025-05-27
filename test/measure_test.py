import qldpcdecoder
from qldpcdecoder.codes import gen_BB_code, gen_HP_ring_code,create_generalized_bicycle_codes,create_QC_GHP_codes,ring_code,hypergraph_product,hamming_code,create_circulant_matrix
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
############################## BB codes           ###########################################
## [[90]]
# css_code = gen_BB_code(72)
# import matplotlib.pyplot as plt
# plt.imshow(bb_code.hx,cmap='gist_yarg')
# plt.savefig(f"results/bb_code_hx_{bb_code.name}.png")
# exit()
############################# GP codes #####################################
## [[72,8,4]]
# code1 = create_circulant_matrix(6,[1,2,3])
# code2 = create_circulant_matrix(6,[1,2,3])
# [[162,8,6]]
# code1 = create_circulant_matrix(9,[1,2,3])
# code2 = create_circulant_matrix(9,[1,2,3])
# # [[288,12,6]]
# code1 = create_circulant_matrix(12,[0,1,2,3])
# code2 = create_circulant_matrix(12,[0,3,2,7])
# [[744,20,6]]
# code1 = create_circulant_matrix(31,[0,2,5])
# code2 = create_circulant_matrix(12,[0,3,2,7])
# css_code = hypergraph_product(code1, code2)
## [[882,48,8]]
# A = 28*create_circulant_matrix(7,[0]) +1* create_circulant_matrix(7,[4])+ 19* create_circulant_matrix(7,[3]) + 28* create_circulant_matrix(7,[2])+1*create_circulant_matrix(7,[1])
# A -= np.ones((7,7),dtype=int)
# css_code = create_QC_GHP_codes(63,A,[0,1,6])
code1 = create_circulant_matrix(31,[0,2,5])
code2 = create_circulant_matrix(24,[0,2,8,15])
css_code = hypergraph_product(code1, code2)
np.save(f'results/matrices/HP_code_hx_{css_code.name}.npy',css_code.hx)
print(css_code.name)
print(css_code.N)
print(css_code.K)
print(css_code.D)
import matplotlib.pyplot as plt
plt.imshow(css_code.hx,cmap='gist_yarg')
plt.savefig(f"results/HP_code_hx_{css_code.name}.png")
exit()
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
A = 28*create_circulant_matrix(7,[0]) +1* create_circulant_matrix(7,[4])+ 19* create_circulant_matrix(7,[3]) + 28* create_circulant_matrix(7,[2])+1*create_circulant_matrix(7,[1])
A -= np.ones((7,7),dtype=int)
css_code = create_QC_GHP_codes(63,A,[0,1,6])

print(css_code.hx.sum(axis=0)[:441])
## [[882,24,6]]
# A_lists = [(27,0),(54,1),(0,2)]
# A = sum([(num+1)*create_circulant_matrix(7,[idx]) for num,idx in A_lists]) - np.ones((7,7),dtype=int)
# css_code = create_QC_GHP_codes(63,A,[0,1,6])
## [[1270,28,6]]
# A = np.array([[0,-1,51,52,-1],
#      [-1,0,-1,111,20],
#      [0,-1,98,-1,122],
#      [0,80,-1,119,-1],
#      [-1,0,5,-1,106]
#      ])
# css_code = create_QC_GHP_codes(127,A,[0,1,7])

print(css_code.name)
print(css_code.N)
print(css_code.K)
print(css_code.D)
exit()
bposddecoder = BPOSD_decoder()

bpdecoder = BP_decoder()
gaussdecoder = guass_decoder(mode = 'greedy')
# reshapeddecoder = DecoupledDecoder(css_code,p)
reshapedgaussdecoder = DecoupledDecoder(css_code,p,decoders_mode='greedy')
pprint(measure_noise_simulation_parallel(css_code,p,[bposddecoder],num_trials=3000,num_repeat=5))

