import numpy as np
from mqt.qecc import *  # UFDecoder
import math
import cvxpy as cp
from z3 import And, If, Optimize, Xor, Bool, sat
from functools import reduce

from our_utils.gauss_elimination import (
    gauss_elimination_mod2,
    calculate_tran_syndrome,
    calculate_original_error,
)


################################################################################################################

from ldpc import bposd_decoder, bp_decoder

# from our_bp_decoder import bp_decoder as our_bp_decoder


class min_sum_decoder:
    def __init__(self, hz, p):
        self.hz = hz
        self.p = p
        self.bp_decoder = bp_decoder(
            self.hz,
            error_rate=p,
            channel_probs=[None],
            max_iter=len(self.hz[0]),
            bp_method="ms",  # minimum sum
            ms_scaling_factor=0,
        )

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

    def bp_decode(self, syndrome, **kwarg):
        self.bp_decoder.decode(syndrome)
        return self.bp_decoder.bp_decoding


################################################################################################################


class gauss_decoder:
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
        # print("row density of B:", np.sum(self.B, axis=1))
        print(f"B shape = ({len(self.B)}, {len(self.B[0])})")
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
        # print(f"trans syndrome = {syndrome_copy}")
        g_syn = np.hstack(
            [
                syndrome_copy,
                np.zeros(len(self.hz_trans[0]) - len(self.hz_trans), dtype=int),
            ]
        )
        g = self.ms_decoder.greedy_decode(g_syn, order=4)  # 传入g_syn = [s', 0]
        # g = self.ms_decoder.our_bp_decode(g_syn)
        f = (np.dot(self.B, g) + syndrome_copy) % 2
        our_result = np.hstack((f, g))
        # assert ((self.hz_trans @ our_result) % 2 == syndrome_copy).all()
        trans_results = calculate_original_error(our_result, self.col_trans)
        # assert ((self.hz @ trans_results) % 2 == syndrome).all(), trans_results
        return trans_results
