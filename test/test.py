from our_bp_decoder import bp_decoder
from ldpc import bposd_decoder
from ldpc.codes import rep_code, ring_code
from bposd.hgp import hgp

import numpy as np

p = 0.05
h = ring_code(3)
surface_code = hgp(h1=h, h2=h, compute_distance=True)

bposd_decoder = bposd_decoder(
    surface_code.hz,
    error_rate=p,
    channel_probs=[None],
    max_iter=surface_code.N,
    bp_method="ms",
    ms_scaling_factor=0,
    osd_method="osd_0",
    osd_order=7,
)

our_bp_decoder = bp_decoder(
    surface_code.hz,
    error_rate=p,
    channel_probs=[None],
    max_iter=surface_code.N,
    bp_method=3,  # minimum sum
    ms_scaling_factor=0,
)

num_trials = 10

num_bposd_success = 0
num_our_success = 0

for _ in range(num_trials):

    # Generate random error
    error = np.zeros(surface_code.N).astype(int)
    for q in range(surface_code.N):
        if np.random.rand() < p:
            error[q] = 1

    # Obtain syndrome
    syndrome = surface_code.hz @ error % 2

    """Decode"""
    # 1. BP+OSD
    bposd_decoder.decode(syndrome)
    bposd_result = bposd_decoder.osdw_decoding
    bposd_residual_error = (bposd_result + error) % 2
    bposd_flag = (surface_code.lz @ bposd_residual_error % 2).any()
    if bposd_flag == 0:
        num_bposd_success += 1

    # 2. Our BP decoder
    our_bp_decoder.decode(syndrome)
    our_result = our_bp_decoder.bp_decoding
    our_residual_error = (our_result + error) % 2
    our_flag = (surface_code.lz @ our_residual_error % 2).any()
    if our_flag == 0:
        num_our_success += 1

bposd_success_rate = num_bposd_success / num_trials
our_success_rate = num_our_success / num_trials
print(f"\nTotal trials: {num_trials}")
print(f"BP Success rate: {bposd_success_rate * 100:.2f}%")
print(f"Our Success rate: {our_success_rate * 100:.2f}%")
