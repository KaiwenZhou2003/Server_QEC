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


@ray.remote
def test_remote(p,N):
    p = p
    css_code = gen_BB_code(N)
    bposddecoder = BPOSD_decoder()

    bpdecoder = BP_decoder()

    # reshapeddecoder = ReShapeBBDecoder(,css_code.lz,A,B)
    _,_,res = circuit_level_simulation(css_code,p,[bposddecoder,bpdecoder],num_trials=int(1/p)*5,num_repeat=5,method=0,W=1)
    with open("results/results_p"+str(p)+"_N"+str(N)+".txt", "w") as file:
        for name,err in res.items():
            file.write(name + " " + str(err) + "\n")
    res['N'] = N
    return res

if __name__ == "__main__":
    ray.init()
    p_list = [5e-3]
    N_list = [72,90,108,144,288,784]
    results = []
    for p in p_list:
        for N in N_list:
            results.append(test_remote.remote(p,N))
    results = ray.get(results)
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv("results/timingBBcodes.csv")

