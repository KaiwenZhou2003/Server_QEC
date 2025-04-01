import qldpcdecoder
from qldpcdecoder.codes import gen_BB_code, gen_HP_ring_code
from qldpcdecoder.bpdecoders import BPOSD_decoder, BP_decoder
from qldpcdecoder.simulation.independentsim import independentnoise_simulation
from qldpcdecoder.simulation.bbcodesim import circuit_level_simulation
from qldpcdecoder.sparsegauss_decoder import guass_decoder
from qldpcdecoder.sparsedecoupleddecoder import DecoupledDecoder
from functools import reduce
import numpy as np
from rich.pretty import pprint
import ray
p = 0.001
css_code = gen_BB_code(72)
bposddecoder = BPOSD_decoder()

bpdecoder = BP_decoder()
gaussdecoder = guass_decoder()

# reshapeddecoder = ReShapeBBDecoder(,css_code.lz,A,B)
pprint(circuit_level_simulation(css_code,p,[gaussdecoder,bposddecoder,bpdecoder],num_trials=1000,num_repeat=6,method=0,plot=True,W=1))

