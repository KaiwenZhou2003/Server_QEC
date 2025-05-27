import numpy as np
from scipy.sparse import csr_matrix, lil_matrix, hstack, vstack, identity, csc_matrix
from .decoder import Decoder
from .timing import timing
from .bpdecoders import BP_decoder,BPOSD_decoder
from .utils import (
    gauss_elimination_mod2,
    calculate_tran_syndrome,
    calculate_original_error,
    calculate_trans_error,
    calculate_trans_prior
)


class min_sum_decoder:
    def __init__(self, hz, p):
        self.hz = csc_matrix(hz)  # 转换为稀疏矩阵
        self.n = hz.shape[1]
        self.p = p

    def update_vec(self, flip_idx, last_vec):
        new_last_vec = last_vec.copy()
        for p in self.hz[:,flip_idx].indices:
            new_last_vec[p] = not last_vec[p]
        return new_last_vec

    def our_bp_decode(self, syndrome, **kwargs):
        from ldpc import bp_decoder
        bp_decoder = bp_decoder(
            self.hz,
            channel_probs=self.p[-self.n:],
            max_iter=100,
            bp_method="ms",
            ms_scaling_factor=0,
        )
        bp_decoder.decode(syndrome)
        return bp_decoder.bp_decoding

    def greedy_decode(self, syndrome, order=6):
        n = self.n
        cur_guess = np.zeros(n, dtype=int)
        cur_conflicts = syndrome.dot(self.p)
        cur_vec = syndrome.copy()
        best_conflicts = cur_conflicts
        best_guess = cur_guess
        best_vec = cur_vec
        for _ in range(1, order + 1):
            for i in range(n):
                if cur_guess[i] == 0:
                    try_guess = cur_guess.copy()
                    try_guess[i] = 1
                    vec = self.update_vec(i, cur_vec)
                    try_conflicts = vec.dot(self.p)
                    if try_conflicts < best_conflicts:
                        best_conflicts = try_conflicts
                        best_guess = try_guess
                        best_vec = vec
            if best_conflicts == cur_conflicts:
                break
            else:
                cur_conflicts = best_conflicts
                cur_guess = best_guess
                cur_vec = best_vec
        return best_guess, best_conflicts

    def frozen_greedy_decode(self, syndrome, order=6, max_iter=10):
        cur_guess, cur_conflicts = self.greedy_decode(syndrome, order=order)
        frozen_bits = []
        best_conflicts = cur_conflicts
        best_guess = cur_guess
        for i in range(max_iter):
            if best_conflicts > 3:
                frozen_bits.extend(np.nonzero(cur_guess)[0])
                cur_guess, cur_conflicts = self.greedy_decode(syndrome, order=order, frozen_idx=frozen_bits)
                if cur_conflicts < best_conflicts:
                    best_conflicts = cur_conflicts
                    best_guess = cur_guess
            else:
                break
        return best_guess, best_conflicts

class guass_decoder(Decoder):
    def __init__(self, **kwargs):
        super().__init__("Sparse_Gauss_" + str(kwargs.get("mode", "both"))+"_"+str(kwargs.get("order", 5)))
        self.mode = kwargs.get("mode", "both")
        self.order = kwargs.get("order", 5)

    def set_h(self, code_h, prior, p, **kwargs):
        self.hz = code_h # 转换为稀疏矩阵
        self.prior = prior
        self.p = p
        self.error_rate = p
        self.pre_decode()

    def pre_decode(self):
        # print("Hz rows nonzero:", np.sum(self.hz, axis=0).reshape(-1))
        hz_trans, col_trans, syndrome_transpose = gauss_elimination_mod2(self.hz)
        # if hz_trans.shape[0] < self.hz.shape[0]:]
        self.hz_trans = hz_trans
        self.col_trans = col_trans
        self.syndrome_transpose = syndrome_transpose
        self.B = hz_trans[:, hz_trans.shape[0]:]
        if np.max(np.sum(self.B, axis=0)) > 8:
            print("B is dense, may cause low decoding performance, use bp decoder instead")
            bpdecoder = BPOSD_decoder()
            bpdecoder.set_h(self.hz,self.prior, self.p)
            self.decode = bpdecoder.decode
        else:
            print("B shape:", self.B.shape)
            # print("B cols nonzero:", np.sum(self.B, axis=0))
            self.BvIg = vstack([self.B, identity(self.B.shape[1],dtype=int)]).toarray().astype(int)
            self.ms_decoder = min_sum_decoder(self.BvIg, calculate_trans_prior(self.prior,self.col_trans))

    # @timing(decoder_info="Sparse Gauss Decoder", log_file="timing.log")
    def decode(self, syndrome):
        order = self.order
        syndrome_copy = calculate_tran_syndrome(syndrome.copy(), self.syndrome_transpose)
        syndrome_copy = syndrome_copy[:self.hz_trans.shape[0]]
        g_syn = np.hstack([syndrome_copy, np.zeros(self.BvIg.shape[0] - self.hz_trans.shape[0], dtype=int)])
        if self.mode == "bp" or self.mode == "both":
            g_bp = self.ms_decoder.our_bp_decode(g_syn)
            f_bp = (self.B.dot(g_bp) + syndrome_copy) % 2
            bp_result = np.hstack((f_bp, g_bp))
        if self.mode == "greedy" or self.mode == "both":
            g_greedy, _ = self.ms_decoder.greedy_decode(g_syn, order=order)
            f_greedy = (self.B.dot(g_greedy) + syndrome_copy) % 2
            greedy_result = np.hstack((f_greedy, g_greedy))
        if self.mode == "bp":
            our_result = bp_result
        elif self.mode == "greedy":
            our_result = greedy_result
        else:
            our_result = bp_result if bp_result.sum() <= greedy_result.sum() else greedy_result
        # assert ((self.hz_trans @ our_result) % 2 == syndrome_copy).all()
        trans_results = calculate_original_error(our_result, self.col_trans)
        # assert ((self.hz @ trans_results) % 2 == syndrome).all(), trans_results
        return trans_results