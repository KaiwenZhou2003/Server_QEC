import numpy as np
from mqt.qecc import *  # UFDecoder
import math
import cvxpy as cp
from z3 import And, If, Optimize, Xor, Bool, sat
from functools import reduce

SELECT_COL = False


# 高斯消元法（mod 2）
def gauss_elimination_mod2(A):
    n = len(A)
    m = len(A[0])
    Augmented = A.copy()
    col_trans = np.arange(m)
    if SELECT_COL:
        for i in range(n):
            # 寻找主元
            if Augmented[i, i] == 0:
                # 如果主元为0，寻找下面一行有1的列交换
                # print(i,i)
                prior_jdx = 0
                min_nonzero_counts = n
                for j in range(i + 1, m):
                    if Augmented[i, j] == 1:
                        nonzero_counts = np.sum(Augmented[:, j])
                        if nonzero_counts < min_nonzero_counts:
                            prior_jdx = j
                            min_nonzero_counts = nonzero_counts
                j = prior_jdx
                col_trans[i], col_trans[j] = col_trans[j], col_trans[i]
                temp = Augmented[:, i].copy()
                Augmented[:, i] = Augmented[:, j]
                Augmented[:, j] = temp
            elif Augmented[i, i] == 1:
                # 如果主元为0，寻找下面一行有1的列交换
                # print(i,i)
                prior_jdx = i
                min_nonzero_counts = np.sum(Augmented[:, i])
                for j in range(i + 1, m):
                    if Augmented[i, j] == 1:
                        nonzero_counts = np.sum(Augmented[:, j])
                        if nonzero_counts < min_nonzero_counts:
                            prior_jdx = j
                            min_nonzero_counts = nonzero_counts
                j = prior_jdx
                if i == j:
                    continue
                col_trans[i], col_trans[j] = col_trans[j], col_trans[i]
                temp = Augmented[:, i].copy()
                Augmented[:, i] = Augmented[:, j]
                Augmented[:, j] = temp

    syndrome_transpose = np.identity(n, dtype=int)
    zero_row_counts = 0
    for i in range(n):
        # 对主元所在行进行消元
        if Augmented[i, i] == 1:
            for j in range(0, n):
                if j != i and Augmented[j, i] == 1:
                    Augmented[j] ^= Augmented[i]
                    syndrome_transpose[j] ^= syndrome_transpose[i]
        else:
            # 如果主元为0，寻找下面一行有1的列交换
            # print(i,i)
            prior_jdx = i
            min_nonzero_counts = n
            for j in range(i + 1, m):
                if Augmented[i, j] == 1:
                    nonzero_counts = np.sum(Augmented[:, j])
                    if nonzero_counts < min_nonzero_counts:
                        prior_jdx = j
                        min_nonzero_counts = nonzero_counts
            if prior_jdx == i:
                zero_row_counts += 1
                ## this i-th row is all zero, put it in the last row
                # temp = Augmented[:,i].copy()
                # Augmented[:,i]  = Augmented[:,prior_jdx]
                # Augmented[:,prior_jdx] = temp
                continue
            col_trans[i], col_trans[prior_jdx] = col_trans[prior_jdx], col_trans[i]
            temp = Augmented[:, i].copy()
            Augmented[:, i] = Augmented[:, prior_jdx]
            Augmented[:, prior_jdx] = temp

            ## 继续消元
            for j in range(0, n):
                if j != i and Augmented[j, i] == 1:
                    Augmented[j] ^= Augmented[i]
                    syndrome_transpose[j] ^= syndrome_transpose[i]
    Augmented = Augmented[: n - zero_row_counts, :]
    print("find zero rows", zero_row_counts)
    # syndrome_transpose = syndrome_transpose[:n-zero_row_counts,:]
    return Augmented, col_trans, syndrome_transpose


def calculate_tran_syndrome(syndrome, syndrome_transpose):
    return syndrome_transpose @ syndrome % 2


def calculate_original_error(our_result, col_trans):
    trans_results = np.zeros_like(our_result, dtype=int)
    col_trans = col_trans.tolist()
    for i in np.arange(len(col_trans)):
        trans_results[i] = our_result[col_trans.index(i)]
    return trans_results


################################################################################################################

from ldpc import bposd_decoder, bp_decoder
from our_bp_decoder import bp_decoder as our_bp_decoder


class min_sum_decoder:
    def __init__(self, hz, p):
        self.hz = hz
        self.p = p

    def count_conflicts(self, syndrome, cur_guess):
        hzg = (self.hz @ cur_guess) % 2
        hzg = hzg.astype(int)
        return np.sum(hzg ^ syndrome)

    def greedy_decode(self, syndrome, order=6):
        """
        贪心算法，目前效果最优
        syndrom = [s', 0]
        """
        n = len(self.hz[0])  # 这里的hz是hstack[B, I]，所以这里的n实际上是n-m
        cur_guess = np.zeros(n, dtype=int)
        cur_conflicts = self.count_conflicts(syndrome, cur_guess)

        for k in range(1, order + 1):
            best_conflicts = cur_conflicts
            best_guess = cur_guess

            for i in range(n):
                if cur_guess[i] == 0:
                    try_guess = cur_guess.copy()
                    try_guess[i] = 1
                    try_conflicts = self.count_conflicts(syndrome, try_guess)
                    if try_conflicts < best_conflicts:
                        best_conflicts = try_conflicts
                        best_guess = try_guess
                # print(
                #     f"order={k},i={i}: best_conflicts={best_conflicts}, best_guess={best_guess}"
                # )

            # 如果当前order没有找到更好的解，则停止
            if best_conflicts == cur_conflicts:
                # print(f"No better solution found in order {k}, break\n")
                break
            else:
                cur_conflicts = best_conflicts
                cur_guess = best_guess

        return best_guess

    def greedy_decode_approx(self, syndrome, order=6):
        """
        近似求解的贪心算法，只求出threshold个1就退出
        syndrom = [s', 0]
        """
        n = len(self.hz[0])  # 这里的hz是hstack[B, I]，所以这里的n实际上是n-m
        cur_guess = np.zeros(n, dtype=int)
        cur_conflicts = self.count_conflicts(syndrome, cur_guess)

        threshold = 2

        for k in range(1, order + 1):
            best_conflicts = cur_conflicts
            best_guess = cur_guess

            max_conflicts = 0
            max_conficlits_idx = -1
            for i in range(n):
                if cur_guess[i] == 0:
                    try_guess = cur_guess.copy()
                    try_guess[i] = 1
                    try_conflicts = self.count_conflicts(syndrome, try_guess)
                    if try_conflicts < best_conflicts:
                        best_conflicts = try_conflicts
                        best_guess = try_guess
                        # if int(np.sum(best_guess)) >= threshold:
                        #     return best_guess
                    else:
                        if try_conflicts > max_conflicts:
                            max_conflicts = try_conflicts
                            max_conficlits_idx = i

            best_guess[max_conficlits_idx] = 2
            # 如果当前order没有找到更好的解，则停止
            if best_conflicts == cur_conflicts:
                break
            else:
                cur_conflicts = best_conflicts
                cur_guess = best_guess

        return best_guess

    # 求解我们堆叠的方程，我们自己的bp decoder
    def our_bp_decode(self, syndrome, **kwarg):
        """
        对于[B, I]g=[s', 0]，先调用greedy_decoder，找到一个近似解，然后用bp_decoder进行迭代
        """
        from ldpc import bp_decoder, bposd_decoder

        max_value = 25  # 对数似然比的大数
        max_prob_1 = 0.55  # g算出来为1的对应的概率
        max_prob_0 = 0.45  # g算出来为0的对应的概率

        g = self.greedy_decode_approx(syndrome)
        # print(f"best guess = {g}, len = {np.sum(g)}")

        channel_probs = np.zeros(len(g))
        delta = 0.0001
        for idx, hard_value in enumerate(g):
            if hard_value == 1:
                # channel_probs[idx] = 1 / (1 + np.exp(-max_value))
                channel_probs[idx] = max_prob_1
            elif hard_value == 2:
                channel_probs[idx] = max_prob_0
            else:
                channel_probs[idx] = np.random.uniform(self.p - delta, self.p + delta)
                # channel_probs[idx] = self.p

        # print(f"channel_probs = {channel_probs}")

        bp_decoder = bp_decoder(
            self.hz,
            error_rate=None,
            channel_probs=channel_probs,
            max_iter=self.hz.shape[1],
            bp_method="ms",  # minimum sum
            ms_scaling_factor=0,
        )

        bp_decoder.decode(syndrome)
        our_result = bp_decoder.bp_decoding

        # print(f"greedy g HW = {np.sum(g)}, bp g HW = {np.sum(our_result)}")
        return our_result


################################################################################################################


class guass_decoder:
    def __init__(self, code_h, error_rate, **kwargs):
        self.hz = code_h
        self.error_rate = error_rate
        pass

    def pre_decode(self):
        H_X = self.hz
        p = self.error_rate
        hz_trans, col_trans, syndrome_transpose = gauss_elimination_mod2(self.hz)
        self.hz_trans = hz_trans
        print(f"hz trans rank {len(self.hz_trans)}, original {len(self.hz)}")
        self.col_trans = col_trans
        self.syndrome_transpose = syndrome_transpose
        self.B = hz_trans[:, len(hz_trans) : len(hz_trans[0])]
        # print("density of B:",np.sum(self.B)/(len(self.B)*len(self.B[0])))
        print("row density of B:", np.sum(self.B, axis=0))
        print(f"{len(self.B)},{len(self.B[0])}")
        weights = [
            np.log((1 - p) / p) for i in range(H_X.shape[1])
        ]  # 初始每个qubit的对数似然比
        assert np.all([w > 0 for w in weights])
        Ig = np.identity(len(self.hz_trans[0]) - len(self.hz_trans))
        self.BvIg = np.vstack([self.B, Ig])
        self.ms_decoder = min_sum_decoder(self.BvIg, self.error_rate)
        # W_f = weights[: H_X.shape[0]]
        # W_g = weights[H_X.shape[0] :]

        # W_f_B = np.dot(W_f, B)  # W_f * B
        # W_g_B_W_f = W_f_B + W_g  # W_f * B + W_g
        # # print(f"W_g_B_W_f = {W_g_B_W_f}")

        # self.zero_g = np.where(
        #     W_g_B_W_f > 0,
        #     0,
        #     np.where(W_g_B_W_f < 0, 1, np.random.randint(0, 2, size=W_g_B_W_f.shape)),
        # )
        # # print(f"g = {g}")

        # self.B_g = np.dot(B, self.zero_g)  # B * g
        # print(f"B_g = {B_g}")

    def decode(self, syndrome):
        syndrome_copy = calculate_tran_syndrome(
            syndrome.copy(), self.syndrome_transpose
        )
        syndrome_copy = syndrome_copy[: len(self.hz_trans)]
        g_syn = np.hstack(
            [
                syndrome_copy,
                np.zeros(len(self.hz_trans[0]) - len(self.hz_trans), dtype=int),
            ]
        )
        g = self.ms_decoder.greedy_decode(g_syn, order=10)  # 传入g_syn = [s', 0]
        # g = self.ms_decoder.our_bp_decode(g_syn)
        f = (np.dot(self.B, g) + syndrome_copy) % 2
        our_result = np.hstack((f, g))
        assert ((self.hz_trans @ our_result) % 2 == syndrome_copy).all()
        trans_results = calculate_original_error(our_result, self.col_trans)
        # assert ((self.hz @ trans_results) % 2 == syndrome).all(), trans_results
        return trans_results


################################################################################################################


def test_decoder(num_trials, surface_code, p, ourdecoder):
    from ldpc import bposd_decoder, bp_decoder

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
    bpdecoder = bp_decoder(
        surface_code.hz,
        error_rate=p,
        channel_probs=[None],
        max_iter=surface_code.N,
        bp_method="ms",  # minimum sum
        ms_scaling_factor=0,
    )

    # UFDecoder
    code = Code(surface_code.hx, surface_code.hz)
    uf_decoder = UFHeuristic()
    uf_decoder.set_code(code)

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
        our_predicates = ourdecoder.decode(syndrome)
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
        else:
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
    # print(f"BP error rate: {bp_error_rate * 100:.2f}%")
    # print(f"BP+OSD error rate: {bposd_error_rate * 100:.2f}%")
    # # print(f"UF Success rate: {uf_error_rate * 100:.2f}%")
    # print(f"Our error rate: {our_error_rate * 100:.2f}%")
    print(f"BP error number: {num_trials - bp_num_success}")
    print(f"BP+OSD error number: {num_trials - bposd_num_success}")
    # print(f"UF Success rate: {uf_error_rate * 100:.2f}%")
    print(f"Our error number: {num_trials - our_num_success}")


################################################################################################################

if __name__ == "__main__":
    np.random.seed(0)
    from ldpc.codes import rep_code, ring_code, hamming_code
    from bposd.hgp import hgp, hgp_single
    from utils.gen_codes import (
        create_bivariate_bicycle_codes,
        hypergraph_product,
        rep_code,
        hamming_code,
    )

    """
    Bivariate Bicycle Codes
    """
    N = 108
    if N == 72:
        bb_code, A_list, B_list = create_bivariate_bicycle_codes(
            6, 6, [3], [1, 2], [1, 2], [3]
        )  # 72
    elif N == 90:
        bb_code, A_list, B_list = create_bivariate_bicycle_codes(
            15, 3, [9], [1, 2], [2, 7], [0]
        )  # 90
    elif N == 108:
        bb_code, A_list, B_list = create_bivariate_bicycle_codes(
            9, 6, [3], [1, 2], [1, 2], [3]
        )  # 108
    elif N == 144:
        bb_code, A_list, B_list = create_bivariate_bicycle_codes(
            12, 6, [3], [1, 2], [1, 2], [3]
        )  # 144
    elif N == 288:
        bb_code, A_list, B_list = create_bivariate_bicycle_codes(
            12, 12, [3], [2, 7], [1, 2], [3]
        )  # 288
    elif N == 360:
        bb_code, A_list, B_list = create_bivariate_bicycle_codes(
            30, 6, [9], [1, 2], [25, 26], [3]
        )  # 360
    elif N == 756:
        bb_code, A_list, B_list = create_bivariate_bicycle_codes(
            21, 18, [3], [10, 17], [3, 19], [5]
        )  # 756
    else:
        print("unsupported N")

    """
    Hypergraph Codes
    """
    rep_code = rep_code(3)
    hm_code = hamming_code(2)
    hg_code = hypergraph_product(rep_code, hm_code)

    # h = hamming_code(2)
    # h2 = hamming_code(4)
    # # surface_code = hgp_single(h1=h, compute_distance=True)
    # # surface_code = hgp(h1=surface_code.hz,h2 =surface_code.hz, compute_distance= True)
    # surface_code = hgp(h1=h, h2=h2, compute_distance=True)

    # print("-" * 30)
    # code.test()
    # print("-" * 30)

    p = 0.0001
    print(f"hz shape = {bb_code.hz.shape}")
    ourdecoder = guass_decoder(bb_code.hz, error_rate=p)
    ourdecoder.pre_decode()
    # print(ourdecoder.hz_trans)
    test_decoder(num_trials=100000, surface_code=bb_code, p=p, ourdecoder=ourdecoder)
