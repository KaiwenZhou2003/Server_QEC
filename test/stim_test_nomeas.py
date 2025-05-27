import qldpcdecoder
from qldpcdecoder.codes import gen_BB_code, gen_HP_ring_code
from qldpcdecoder.bpdecoders import BPOSD_decoder, BP_decoder,BPOSD0_decoder
from qldpcdecoder.simulation.independentsim import independentnoise_simulation
from qldpcdecoder.simulation.bbcodesim import circuit_level_simulation
from qldpcdecoder.sparsegauss_decoder import guass_decoder
from qldpcdecoder.sparsedecoupleddecoder import DecoupledDecoder
from functools import reduce
import numpy as np
from rich.pretty import pprint
import ray

import os
pathdir = 'results/nomeas1roundBpMore/'
if not os.path.exists(pathdir):
    os.makedirs(pathdir)
@ray.remote
def test_remote(p,N):
    p = p
    css_code = gen_BB_code(N)
    bposddecoder = BPOSD_decoder()
    bposd0decoder = BPOSD0_decoder()
    bpdecoder = BP_decoder()
    gaussdecoder = guass_decoder()

    # reshapeddecoder = ReShapeBBDecoder(,css_code.lz,A,B)
    _,res = circuit_level_simulation(css_code,p,[bposddecoder,bposd0decoder,bpdecoder],num_trials=int(1/p)*500,num_repeat=2,method=0,W=1)
    with open(f"{pathdir}results_p{p}_N{N}_D{int(css_code.D)}.txt", "w") as file:
        for name,err in res.items():
            file.write(name + " " + str(err) + "\n")
    return res

if __name__ == "__main__":
    ray.init()
    p_list = reversed([5e-4,6e-4,7e-4,8e-4,9e-4,1e-3,2e-3,3e-3,4e-3,5e-3])
    N_list = [288,784]
    results = []
    for p in p_list:
        for N in N_list:
            results.append(test_remote.remote(p,N))
    results = ray.get(results)
    for res in results:
        pprint(res)

