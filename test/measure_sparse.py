import qldpcdecoder
from qldpcdecoder.codes import gen_BB_code, gen_HP_ring_code,create_circulant_matrix,hypergraph_product
from qldpcdecoder.bpdecoders import BPOSD_decoder, BP_decoder
from qldpcdecoder.simulation.independentsim import measure_noise_simulation,measure_noise_simulation_by_trial_data,measure_noise_simulation_parallel
from qldpcdecoder.simulation.bbcodesim import circuit_level_simulation
from qldpcdecoder.sparsegauss_decoder import guass_decoder
from qldpcdecoder.sparsedecoupleddecoder import DecoupledDecoder
from functools import reduce
import numpy as np
from rich.pretty import pprint
np.random.seed(12561)

code1 = create_circulant_matrix(24,[0,2,8,15])
code2 = create_circulant_matrix(24,[0,2,8,15])
css_code = hypergraph_product(code1, code2)
name = "HP_1152_18_8"
# css_code = gen_HP_ring_code(7,7)
# css_code = gen_BB_code(784)
bposddecoder = BPOSD_decoder()

bpdecoder = BP_decoder()
gaussdecoder = guass_decoder()
# greedydecoder = guass_decoder(mode='greedy')
with open(f"{name}.log", "w") as f:
    f.write(f"code={css_code.name}\n")
    for p ,num_trials in [(5e-4,100000),(6e-4,100000),(7e-4,80000),(8e-4,80000),(9e-4,50000),(1e-3,50000),(2e-3,40000),(3e-3,30000),(4e-3,20000),(5e-3,10000)]:
        reshapeddecoder = DecoupledDecoder(css_code,p)
        # reshapedgaussdecoder = DecoupledDecoder(css_code,p,decoders_mode='greedy')
        f.write(f"p={p}\n")
        gauss_res, trail_data = measure_noise_simulation_parallel(css_code,p,[reshapeddecoder,gaussdecoder],num_trials=num_trials,num_repeat=5)
        for key,value in gauss_res[0].items():
            f.write(f"logical_error_rate_{key}: {value}\n")
        for key,value in gauss_res[1].items():
            f.write(f"logical_error_rate_per_round_{key}: {value}\n")
        f.flush()
        bp_res = measure_noise_simulation_by_trial_data(css_code,p,[bposddecoder,bpdecoder],num_trials=num_trials,num_repeat=5,trial_data=trail_data)
        for key,value in bp_res[0].items():
            f.write(f"logical_error_rate_{key}: {value}\n")
        for key,value in bp_res[1].items():
            f.write(f"logical_error_rate_per_round_{key}: {value}\n")
        f.write('\n')
        f.flush()

