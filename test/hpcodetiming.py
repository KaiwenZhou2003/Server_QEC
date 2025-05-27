import qldpcdecoder
from qldpcdecoder.codes import gen_BB_code, gen_HP_ring_code,create_circulant_matrix,hypergraph_product,create_QC_GHP_codes,ring_code
from qldpcdecoder.bpdecoders import BPOSD_decoder, BP_decoder
from qldpcdecoder.simulation.independentsim import measure_noise_simulation,measure_noise_simulation_by_trial_data,measure_noise_simulation_parallel
from qldpcdecoder.simulation.bbcodesim import circuit_level_simulation
from qldpcdecoder.sparsegauss_decoder import guass_decoder
from qldpcdecoder.sparsedecoupleddecoder import DecoupledDecoder
from functools import reduce
import numpy as np
from rich.pretty import pprint
import ray
# np.random.seed(12561)
np.random.seed(1251)
def gen_benchmark_codes(N):
    if N == 162:
        code1 = ring_code(9)
        code2 = ring_code(12)
        css_code = hypergraph_product(code1, code2)
    elif N == 338:
        code1 = ring_code(13)
        code2 = ring_code(13)
        css_code = hypergraph_product(code1, code2)
    elif N == 288:
        code1 = create_circulant_matrix(12,[0,1,2,3])
        code2 = create_circulant_matrix(12,[0,1,2,3])
        css_code = hypergraph_product(code1, code2)
    elif N == 744:
        code1 = create_circulant_matrix(31,[0,2,5])
        code2 = create_circulant_matrix(12,[0,3,2,7])
        css_code = hypergraph_product(code1, code2)
    elif N == 1488:
        code1 = create_circulant_matrix(31,[0,2,5])
        code2 = create_circulant_matrix(24,[0,2,8,15])
        css_code = hypergraph_product(code1, code2)
    elif N == 882:
        A = 28*create_circulant_matrix(7,[0]) +1* create_circulant_matrix(7,[4])+ 19* create_circulant_matrix(7,[3]) + 28* create_circulant_matrix(7,[2])+1*create_circulant_matrix(7,[1])
        A -= np.ones((7,7),dtype=int)
        css_code = create_QC_GHP_codes(63,A,[0,1,6])
    return css_code
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


@ray.remote
def test_remote(N):
    css_code = gen_benchmark_codes(N)
    bposddecoder = BPOSD_decoder()

    bpdecoder = BP_decoder()
    numrepeat = 5

    # greedydecoder = guass_decoder(mode='greedy')
    with open(f"results/time_{css_code.name}.log", "w") as f:
        p = 5e-3
        num_trials = 10000
        _,_,bp_res = measure_noise_simulation(css_code,p,[bposddecoder,bpdecoder],num_trials=num_trials,num_repeat=numrepeat)
        for key,value in bp_res.items():
            f.write(f"{key}: {value}\n")
    bp_res['N'] = N
    return bp_res
if __name__ == "__main__":
    p_list = [5e-3]
    N_list = [288]
    results = []
    for N in N_list:
        results.append(test_remote.remote(N))
    results = ray.get(results)
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv("results/timingBBcodes.csv")

