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
p = 0.005
css_code = gen_BB_code(144)
# css_code = gen_HP_ring_code(7,7)
bposddecoder = BPOSD_decoder()

bpdecoder = BP_decoder()
gaussdecoder = guass_decoder()
reshapeddecoder = DecoupledDecoder(css_code,p)
reshapedgaussdecoder = DecoupledDecoder(css_code,p,decoders_mode='greedy')
pprint(measure_noise_simulation(css_code,p,[reshapeddecoder,reshapedgaussdecoder,gaussdecoder,bposddecoder,bpdecoder],num_trials=1000,num_repeat=5))

