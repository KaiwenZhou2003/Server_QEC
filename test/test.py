from our_bp_decoder import bp_decoder

from ldpc.codes import rep_code, ring_code
from bposd.hgp import hgp

p = 0.05
h = ring_code(3)
surface_code = hgp(h1=h, h2=h, compute_distance=True)

our_bp_decoder = bp_decoder(
    surface_code.hz,
    error_rate=p,
    channel_probs=[None],
    max_iter=surface_code.N,
    bp_method="ms",  # minimum sum
    ms_scaling_factor=0,
)
