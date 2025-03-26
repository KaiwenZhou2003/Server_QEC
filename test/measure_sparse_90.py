import qldpcdecoder
from qldpcdecoder.codes import gen_BB_code, gen_HP_ring_code
from qldpcdecoder.bpdecoders import BPOSD_decoder, BP_decoder
from qldpcdecoder.simulation.independentsim import measure_noise_simulation,independentnoise_simulation
from qldpcdecoder.simulation.bbcodesim import circuit_level_simulation
from qldpcdecoder.sparsegauss_decoder import guass_decoder
from qldpcdecoder.sparsedecoupleddecoder import DecoupledDecoder
from functools import reduce
import numpy as np
from rich.pretty import pprint
np.random.seed(12561)


p_list = [0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001, 0.002, 0.003, 0.004, 0.005]
# p_list = [0.001]
data_qubits_num = 90
css_code = gen_BB_code(data_qubits_num)
# css_code = gen_HP_ring_code(7,7)


for p in p_list:
  
    bposddecoder = BPOSD_decoder()
    bpdecoder = BP_decoder()
    gaussdecoder = guass_decoder() # [H, I] no decouple, hybrid
    # reshapeddecoder = DecoupledDecoder(css_code,p) # [H, I] decouple, hybrid

    if p in [0.0005, 0.0006, 0.0007, 0.0008, 0.0009]:
        ler, ler_per_round = measure_noise_simulation(css_code,p,[bposddecoder,bpdecoder,gaussdecoder,],num_trials=100000,num_repeat=10)
    if p in [0.001, 0.002, 0.003, 0.004, 0.005]:
        ler, ler_per_round = measure_noise_simulation(css_code,p,[bposddecoder,bpdecoder,gaussdecoder,],num_trials=10000,num_repeat=10)

    filename = f"LER/BBcode_{data_qubits_num}/dataq_{data_qubits_num}_p{p}.txt"

    with open(filename, "w") as f:
        f.write('ler ' + str(ler) + '\n')
        f.write('ler_per_round ' + str(ler_per_round))
