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
############################## BB codes           ###########################################
## [[90]]
css_code = gen_BB_code(288)

## [[72,8,4]]
# code1 = create_circulant_matrix(6,[1,2,3])
# code2 = create_circulant_matrix(6,[1,2,3])
## [[162,8,6]]
# code1 = create_circulant_matrix(9,[1,2,3])
# code2 = create_circulant_matrix(9,[1,2,3])
# [[288,12,6]]
# code1 = create_circulant_matrix(12,[0,1,2,3])
# code2 = create_circulant_matrix(12,[0,3,2,7])
# css_code = hypergraph_product(code1, code2)
# [[744,20,6]]
# code1 = create_circulant_matrix(31,[0,2,5])
# code2 = create_circulant_matrix(12,[0,3,2,7])
## [[1488,30,7]]
# code1 = create_circulant_matrix(31,[0,2,5])
# code2 = create_circulant_matrix(24,[0,2,8,15])
# css_code = hypergraph_product(code1, code2)
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
# css_code = gen_BB_code(784)
# bposddecoder = BPOSD_decoder()

# bpdecoder = BP_decoder()
import os
os.makedirs("results/order_sim/",exist_ok=True)
# greedydecoder = guass_decoder(mode='greedy')
with open(f"results/order_sim/{css_code.name}.log", "w") as f:
    f.write(f"code={css_code.name}\n")
    for p ,num_trials in [(5e-4,10000),(7e-4,8000),(9e-4,5000),(1e-3,5000),(3e-3,3000),(5e-3,1000)]:
        gaussdecoders = [guass_decoder(order=1,mode='greedy'),guass_decoder(order=2,mode='greedy'),guass_decoder(order=3,mode='greedy'),guass_decoder(order=4,mode='greedy'),guass_decoder(order=5,mode='greedy'),guass_decoder(order=6,mode='greedy'),guass_decoder(order=7,mode='greedy')]
        f.write(f"p={p}\n")
        gauss_res, trail_data = measure_noise_simulation_parallel(css_code,p,gaussdecoders,num_trials=num_trials,num_repeat=5)
        for key,value in gauss_res[0].items():
            f.write(f"logical_error_rate_{key}: {value}\n")
        for key,value in gauss_res[1].items():
            f.write(f"logical_error_rate_per_round_{key}: {value}\n")
        f.flush()

