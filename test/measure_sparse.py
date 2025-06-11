import qldpcdecoder
from qldpcdecoder.codes import gen_BB_code, gen_HP_ring_code,create_circulant_matrix,hypergraph_product,create_QC_GHP_codes
from qldpcdecoder.bpdecoders import BPOSD_decoder, BP_decoder
from qldpcdecoder.simulation.independentsim import measure_noise_simulation,measure_noise_simulation_by_trial_data,measure_noise_simulation_parallel
from qldpcdecoder.simulation.bbcodesim import circuit_level_simulation
from qldpcdecoder.sparsegauss_decoder import guass_decoder
from qldpcdecoder.sparsedecoupleddecoder import DecoupledDecoder
from functools import reduce
import numpy as np
from rich.pretty import pprint
np.random.seed(12561)

# code = gen_HP_ring_code(7,7)
## [[72,8,4]]
# code1 = create_circulant_matrix(6,[1,2,3])
# code2 = create_circulant_matrix(6,[1,2,3])
# [[162,8,6]]
# code1 = create_circulant_matrix(9,[1,2,3])
# code2 = create_circulant_matrix(9,[1,2,3])
## [[288,12,6]]
# code1 = create_circulant_matrix(12,[0,1,2,3])
# code2 = create_circulant_matrix(12,[0,3,2,7])
# [[744,20,6]]
# code1 = create_circulant_matrix(31,[0,2,5])
# code2 = create_circulant_matrix(12,[0,3,2,7])
# css_code = hypergraph_product(code1, code2)
## [[1488,30,7]]
# code1 = create_circulant_matrix(31,[0,2,5])
# code2 = create_circulant_matrix(24,[0,2,8,15])

# css_code = hypergraph_product(code1, code2)
css_code = gen_HP_ring_code(13,13)
## [[882,48,8]]
# A = 28*create_circulant_matrix(7,[0]) +1* create_circulant_matrix(7,[4])+ 19* create_circulant_matrix(7,[3]) + 28* create_circulant_matrix(7,[2])+1*create_circulant_matrix(7,[1])
# A -= np.ones((7,7),dtype=int)
# css_code = create_QC_GHP_codes(63,A,[0,1,6])
## [[1270,28,6]]
# A = np.array([[0,-1,51,52,-1],
#      [-1,0,-1,111,20],
#      [0,-1,98,-1,122],
#      [0,80,-1,119,-1],
#      [-1,0,5,-1,106]
#      ])
# css_code = create_QC_GHP_codes(127,A,[0,1,7])

# css_code = gen_HP_ring_code(7,7)
bposddecoder = BPOSD_decoder()

bpdecoder = BP_decoder()
gaussdecoder = guass_decoder(mode='greedy')
numrepeat = 5
# greedydecoder = guass_decoder(mode='greedy')
with open(f"results/rebuttal/{css_code.name}.log", "w") as f:
    f.write(f"code={css_code.name}\n")
    for p ,num_trials in [(5e-4,100000),(6e-4,100000),(7e-4,80000),(8e-4,80000),(9e-4,50000),(1e-3,50000),(2e-3,40000),(3e-3,30000),(4e-3,20000),(5e-3,10000)]:
        # reshapeddecoder = DecoupledDecoder(css_code,p)
        reshapedgaussdecoder = DecoupledDecoder(css_code,p,decoders_mode='greedy')
        f.write(f"p={p}\n")
        gauss_res, trail_data = measure_noise_simulation_parallel(css_code,p,[gaussdecoder,reshapedgaussdecoder],num_trials=num_trials,num_repeat=numrepeat)
        for key,value in gauss_res[0].items():
            f.write(f"logical_error_rate_{key}: {value}\n")
        for key,value in gauss_res[1].items():
            f.write(f"logical_error_rate_per_round_{key}: {value}\n")
        f.flush()