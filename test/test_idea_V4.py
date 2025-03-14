"""
feature:
1. 改进了贪心，不要过度早停
2. Hybrid（贪心+BP）
"""

import numpy as np
from mqt.qecc import *  # UFDecoder
from utils.gauss_elimination import (
    gauss_elimination_mod2,
    calculate_tran_syndrome,
    calculate_original_error,
)
from utils.test_decoder import test_decoder
from ldpc import bposd_decoder, bp_decoder


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

    def greedy_decode_V1(self, syndrome, order=6):
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

            # 如果当前order没有找到更好的解，则停止
            if best_conflicts == cur_conflicts:
                # print(f"No better solution found in order {k}, break\n")
                break
            else:
                cur_conflicts = best_conflicts
                cur_guess = best_guess

        return best_guess

    def greedy_decode_V2(self, syndrome, order=6):
        """
        贪心算法，目前效果最优
        syndrom = [s', 0]
        """
        n = len(self.hz[0])  # 这里的hz是hstack[B, I]，所以这里的n实际上是n-m
        cur_guess = np.zeros(n, dtype=int)
        cur_conflicts = self.count_conflicts(syndrome, cur_guess)
        best_conflicts = cur_conflicts
        best_guess = cur_guess
        candidate_guesses = [[] for _ in range(order + 1)]
        candidate_guesses[0].append(cur_guess)
        for k in range(1, order + 1):
            for cur_guess in candidate_guesses[k - 1]:
                for i in range(n):
                    if cur_guess[i] == 0:
                        try_guess = cur_guess.copy()
                        try_guess[i] = 1
                        try_conflicts = self.count_conflicts(syndrome, try_guess)

                        if try_conflicts < best_conflicts:
                            best_conflicts = try_conflicts
                            best_guess = try_guess
                        elif try_conflicts < best_conflicts + 4:
                            candidate_guesses[k].append(try_guess)
            if len(candidate_guesses[k]) == 0:
                break

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


class guass_decoder:
    def __init__(self, code_h, error_rate, **kwargs):
        self.hz = code_h
        self.error_rate = error_rate
        pass

    # 预解码，高斯消元得到B
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
        print(f"B shape = ({len(self.B)}, {len(self.B[0])})")
        weights = [
            np.log((1 - p) / p) for i in range(H_X.shape[1])
        ]  # 初始每个qubit的对数似然比
        assert np.all([w > 0 for w in weights])
        Ig = np.identity(len(self.hz_trans[0]) - len(self.hz_trans))
        self.BvIg = np.vstack([self.B, Ig])
        self.ms_decoder = min_sum_decoder(self.BvIg, self.error_rate)

    # 正式解码
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
        g = self.ms_decoder.greedy_decode_V1(g_syn, order=3)  # 传入g_syn = [s', 0]
        # g = self.ms_decoder.our_bp_decode(g_syn)
        f = (np.dot(self.B, g) + syndrome_copy) % 2

        our_result = np.hstack((f, g))
        assert ((self.hz_trans @ our_result) % 2 == syndrome_copy).all()
        trans_results = calculate_original_error(our_result, self.col_trans)
        assert ((self.hz @ trans_results) % 2 == syndrome).all(), trans_results
        return trans_results, g


################################################################################################################

if __name__ == "__main__":
    np.random.seed(0)
    import ray
    from ldpc.codes import ring_code
    from utils.gen_codes import (
        create_bivariate_bicycle_codes,
        hypergraph_product,
        rep_code,
        hamming_code,
    )

    """
    Bivariate Bicycle Codes
    """
    N = 144
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
    code_1 = rep_code(5)
    code_2 = rep_code(5)
    hg_code = hypergraph_product(code_1, code_2)

    # h = hamming_code(2)
    # h2 = hamming_code(4)
    # # surface_code = hgp_single(h1=h, compute_distance=True)
    # # surface_code = hgp(h1=surface_code.hz,h2 =surface_code.hz, compute_distance= True)
    # surface_code = hgp(h1=h, h2=h2, compute_distance=True)

    # print("-" * 30)
    # code.test()
    # print("-" * 30)

    p = 0.001
    code = bb_code
    print(f"hz shape = {code.hz.shape}")

    ourdecoder = guass_decoder(code.hz, error_rate=p)
    ourdecoder.pre_decode()
    print(f"ourdecoder.hz_trans = {ourdecoder.hz_trans}")

    with open("./gauss_matrix_txt/hz_trans.txt", "w") as file:
        for row in ourdecoder.hz_trans:
            file.write(" ".join(map(str, row)) + "\n")

    # ray.init()

    # total_trials = 100000000  # 1e8
    # single_trials = 100000  # 每个work负责的trial
    # num_workers = int(total_trials / single_trials)

    # futures = [
    #     test_decoder.remote(single_trials, code, p, ourdecoder)
    #     for _ in range(num_workers)
    # ]
    # results = ray.get(futures)

    test_decoder(num_trials=10000, surface_code=code, p=p, ourdecoder=ourdecoder)
