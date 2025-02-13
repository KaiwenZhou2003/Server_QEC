from ldpc.codes import rep_code
from bposd.hgp import hgp
import time

import numpy as np
from bposd import bposd_decoder
from mqt.qecc import *  # UFDecoder

h = rep_code(3)
surface_code = hgp(h1=h, h2=h, compute_distance=True)
# surface_code.test()

print(f"surface_code.hz = {surface_code.hz}")
print(f"surface_code.hx = {surface_code.hx}")
print(f"surface_code lz = {surface_code.lz}")
print(f"surface_code lx = {surface_code.lx}")

p = 0.05  # 错误率

# BP+OSD
bposd_decoder = bposd_decoder(
    surface_code.hz,
    error_rate=p,
    channel_probs=[None],
    max_iter=surface_code.N,
    bp_method="ms",  # minimum sum
    ms_scaling_factor=0,
    osd_method="osd_0",
    osd_order=7,
)

# UFDecoder
code = Code(surface_code.hx, surface_code.hz)
uf_decoder = UFHeuristic()
uf_decoder.set_code(code)
