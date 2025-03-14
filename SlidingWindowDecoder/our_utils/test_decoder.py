"""
测量不同decoder的解码准确度
"""

import numpy as np
import ray


# @ray.remote
def test_decoder(num_trials, surface_code, p, ourdecoder):
    from ldpc import bposd_decoder, bp_decoder

    # 校验矩阵H的行和列数
    m = surface_code.hz.shape[0]
    n = surface_code.hz.shape[1]

    print(f"m = {m}, n = {n}, surface.N = {surface_code.N}")

    # BP+OSD
    bposddecoder = bposd_decoder(
        surface_code.hz,
        error_rate=p,
        channel_probs=[None],
        max_iter=surface_code.N,
        bp_method="ms",
        ms_scaling_factor=0,
        osd_method="osd_0",
        osd_order=7,
    )

    # BP
    bpdecoder = bp_decoder(
        surface_code.hz,
        error_rate=p,
        channel_probs=[None],
        max_iter=surface_code.N,
        bp_method="ms",  # minimum sum
        ms_scaling_factor=0,
    )

    # # UFDecoder
    # code = Code(surface_code.hx, surface_code.hz)
    # uf_decoder = UFHeuristic()
    # uf_decoder.set_code(code)

    bposd_num_success = 0
    bp_num_success = 0
    uf_num_success = 0
    our_num_success = 0

    for i in range(num_trials):

        # generate error
        error = np.zeros(surface_code.N).astype(int)
        for q in range(surface_code.N):
            if np.random.rand() < p:
                error[q] = 1

        syndrome = surface_code.hz @ error % 2

        """Decode"""
        # 0. BP
        bpdecoder.decode(syndrome)

        # 1. BP+OSD
        bposddecoder.decode(syndrome)
        # bposd_result =  bposddecoder.osdw_decoding

        bposd_residual_error = (bposddecoder.osdw_decoding + error) % 2
        bposdflag = (surface_code.lz @ bposd_residual_error % 2).any()
        if bposdflag == 0:
            bposd_num_success += 1

        bp_residual_error = (bpdecoder.bp_decoding + error) % 2
        bpflag = (surface_code.lz @ bp_residual_error % 2).any()
        if bpflag == 0:
            bp_num_success += 1

        # 2. UFDecoder
        # uf_decoder.decode(syndrome)
        # uf_result = np.array(uf_decoder.result.estimate).astype(int)
        # uf_residual_error = (uf_result + error) % 2
        # ufflag = (surface_code.lz @ uf_residual_error % 2).any()
        # if ufflag == 0:
        #     uf_num_success += 1

        # 3. Our Decoder
        our_predicates, g = ourdecoder.decode(syndrome)
        # 如果我们的greedy算出来的HW大于某个阈值，并且此时HW比BP的还要大，那就取BP的结果

        # print(f"our_predicates len = {len(our_predicates)}")
        if np.sum(our_predicates) > 2 and np.sum(our_predicates) > np.sum(
            bpdecoder.bp_decoding
        ):
            our_predicates = bpdecoder.bp_decoding
        our_residual_error = (our_predicates + error) % 2
        # assert not ((surface_code.lz @ our_predicates)%2).all(), (surface_code.lz @our_predicates)
        flag = (surface_code.lz @ our_residual_error % 2).any()
        if flag == 0:
            our_num_success += 1
            if bposdflag == 1:
                # print(
                #     f"BP+OSD fail, we success: our HW = {np.sum(our_predicates)}, bposd HW = {np.sum(bposddecoder.osdw_decoding)}"
                # )
                pass
        # 解码失败
        else:
            our_g_len = len(g)
            real_g = error[len(error) - our_g_len :]
            print(f"our g = {np.nonzero(g)[0]}, real g = {np.nonzero(real_g)[0]}")
            print(
                f"our e = {np.nonzero(our_predicates)[0]}, real e = {np.nonzero(error)[0]}"
            )
            print("\n")

            if bposdflag == 0:
                # print(
                #     f"BP+OSD success, we failed: our HW = {np.sum(our_predicates)}, bposd HW = {np.sum(bposddecoder.osdw_decoding)}"
                # )
                # print(
                #     f"{our_predicates}, our HW = {np.sum(our_predicates)}\n{bposddecoder.osdw_decoding}, bposd HW = {np.sum(bposddecoder.osdw_decoding)}"
                # )
                # print("\n")
                pass
            # print(our_predicates,error)

    bposd_error_rate = 1 - bposd_num_success / num_trials
    bp_error_rate = 1 - bp_num_success / num_trials
    # uf_error_rate = 1- uf_num_success / num_trials
    our_error_rate = 1 - our_num_success / num_trials
    print(f"\nTotal trials: {num_trials}")

    """Error rate"""
    # print(f"BP error rate: {bp_error_rate * 100:.2f}%")
    # print(f"BP+OSD error rate: {bposd_error_rate * 100:.2f}%")
    # # print(f"UF Success rate: {uf_error_rate * 100:.2f}%")
    # print(f"Our error rate: {our_error_rate * 100:.2f}%")

    """Error number"""
    print(f"BP error number: {num_trials - bp_num_success}")
    print(f"BP+OSD error number: {num_trials - bposd_num_success}")
    # print(f"UF Success rate: {uf_error_rate * 100:.2f}%")
    print(f"Our error number: {num_trials - our_num_success}")
