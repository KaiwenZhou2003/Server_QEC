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
import os
pathdir = "results/circuit_level_results/"
os.makedirs(pathdir,exist_ok=True)
@ray.remote
def test_remote(p,N,K,D):
    p = p
    css_code = gen_BB_code(N)
    bposddecoder = BPOSD_decoder()
    bpdecoder = BP_decoder()
    _,res,_ = circuit_level_simulation(css_code,p,[bposddecoder,bpdecoder],num_trials=int(1/p)*500,num_repeat=int(D),method=0,W=1)
    with open(pathdir+"p"+str(p)+f"_{res['code']}.txt", "w") as file:
        for name,err in res.items():
            file.write(name + " " + str(err) + "\n")
    res['code'] = f"[[{N},{K},{D}]]"
    res['p'] = p
    return res

if __name__ == "__main__":
    ray.init()
    p_list = [5e-4,6e-4,7e-4,8e-4,9e-4,1e-3,2e-3,3e-3,4e-3,5e-3]
    N_list = [[72,12,6],[90,8,10],[108,8,10],[144,12,10],[288,12,18],[784,24,24]]
    results = []
    for p in p_list:
        for N,K,D in N_list:
            results.append(test_remote.remote(p,N,K,D))
    results = ray.get(results)
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(pathdir+"results.csv")

