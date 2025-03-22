import numpy as np
from scipy.sparse import csr_matrix, lil_matrix, hstack, vstack, identity
from scipy.sparse.linalg import inv
from scipy.sparse import find
import os
from .decoder import Decoder
from .sparsegauss_decoder import guass_decoder
from .timing import timing
class DecoupledBBDecoder(Decoder):
    def __init__(self, code, p, **kwargs):
        super().__init__("Decoupled_BB_Decoder_" + kwargs.get("decoders_mode", "both"))
        self.code = code
        self.p = p
        self.Adecoders = []
        self.Bdecoders = []
        self.decoders_mode = kwargs.get("decoders_mode", "both")
        self.load_decoupled_matrix()

    def load_decoupled_matrix(self):
        pathdir = "results/" + self.code.name + "/"
        if not os.path.exists(pathdir):
            print("No decoupled matrix found, please run decouple matrix first")
        self.A_T = csr_matrix(np.load(pathdir + "A_T.npy"))
        self.A_C = csr_matrix(np.load(pathdir + "A_C.npy"))
        self.A_THC = csr_matrix(np.load(pathdir + "A_THC.npy"))
        self.A_row_window = np.load(pathdir + "A_row_part.npy")
        self.A_col_window = np.load(pathdir + "A_col_part.npy")
        self.A_anchors = []
        start_row, start_col = 0, 0

        for r, c in zip(self.A_row_window, self.A_col_window):
            block = self.A_THC[start_row:start_row + r, start_col:start_col + c]
            self.Adecoders.append(guass_decoder(mode=self.decoders_mode))
            h = hstack([block, identity(block.shape[0], format='csr')])
            self.Adecoders[-1].set_h(h, prior=[None], p=self.p)
            self.A_anchors.append([(start_row, start_col), (start_row + r, start_col + c)])
            start_row += r
            start_col += c

        self.B_T = csr_matrix(np.load(pathdir + "B_T.npy"))
        self.B_C = csr_matrix(np.load(pathdir + "B_C.npy"))
        self.B_THC = csr_matrix(np.load(pathdir + "B_THC.npy"))
        self.B_row_window = np.load(pathdir + "B_row_part.npy")
        self.B_col_window = np.load(pathdir + "B_col_part.npy")
        self.B_anchors = []
        start_row, start_col = 0, 0

        for r, c in zip(self.B_row_window, self.B_col_window):
            block = self.B_THC[start_row:start_row + r, start_col:start_col + c]
            self.B_anchors.append([(start_row, start_col), (start_row + r, start_col + c)])
            self.Bdecoders.append(guass_decoder(mode=self.decoders_mode))
            h = hstack([block, identity(block.shape[0], format='csr')])
            self.Bdecoders[-1].set_h(h, prior=[None], p=self.p)
            start_row += r
            start_col += c
    @timing(decoder_info="Decoupled BB Decoder", log_file="timing.log")
    def decode(self, syndrome):
        return self.adaptiveSearch(syndrome, 3)[0]

    def reshape_decode(self, l_syndrome, r_syndrome):
        corrections = []
        left_syndrome = self.A_T.dot(l_syndrome)
        for i, anchor in enumerate(self.A_anchors):
            syndrome_block = left_syndrome[anchor[0][0]:anchor[1][0]]
            error_correction = self.Adecoders[i].decode(syndrome_block)
            corrections.append(error_correction[:anchor[1][1] - anchor[0][1]])
        left_correction = np.hstack(corrections)
        left_correction = self.A_C.dot(left_correction)
        corrections = []
        right_syndrome = self.B_T.dot(r_syndrome)
        for j, anchor in enumerate(self.B_anchors):
            syndrome_block = right_syndrome[anchor[0][0]:anchor[1][0]]
            error_correction = self.Bdecoders[j].decode(syndrome_block)
            corrections.append(error_correction[:anchor[1][1] - anchor[0][1]])
        right_correction = np.hstack(corrections)
        right_correction = self.B_C.dot(right_correction)
        syn_correction = np.hstack((left_correction, right_correction))
        return syn_correction

    def decode_right(self, syndrome):
        corrections = []
        right_syndrome = self.B_T.dot(syndrome)
        for j, anchor in enumerate(self.B_anchors):
            syndrome_block = right_syndrome[anchor[0][0]:anchor[1][0]]
            error_correction = self.Bdecoders[j].decode(syndrome_block)
            corrections.append(error_correction[:anchor[1][1] - anchor[0][1]])
        right_correction = np.hstack(corrections)
        right_correction = self.B_C.dot(right_correction)
        return right_correction

    def decode_left(self, syndrome):
        corrections = []
        left_syndrome = self.A_T.dot(syndrome)
        for i, anchor in enumerate(self.A_anchors):
            syndrome_block = left_syndrome[anchor[0][0]:anchor[1][0]]
            error_correction = self.Adecoders[i].decode(syndrome_block)
            corrections.append(error_correction[:anchor[1][1] - anchor[0][1]])
        left_correction = np.hstack(corrections)
        left_correction = self.A_C.dot(left_correction)
        return left_correction

    def count_conflicts(self, syndrome, correction):
        return np.sum(syndrome ^ (self.code.hx.dot(correction) % 2)) + np.sum(correction)

    def adaptiveSearch(self, syndrome, order):
        m, n = self.code.hx.shape
        Apart = self.code.hx[:, :n // 2]
        Bpart = self.code.hx[:, n // 2:]

        # Greedy search for L
        cur_guess = np.zeros(n // 2, dtype=int)
        cur_right_syndrome = syndrome ^ (Apart.dot(cur_guess) % 2)
        cur_right_correction = self.decode_right(cur_right_syndrome)
        cur_correction = np.hstack((cur_guess, cur_right_correction))
        cur_conflicts = self.count_conflicts(syndrome, cur_correction)
        best_conflicts = cur_conflicts
        best_correction = cur_correction
        best_guess = cur_guess.copy()

        for _ in range(1, order + 1):
            for i in range(n // 2):
                if cur_guess[i] == 0:
                    try_guess = cur_guess.copy()
                    try_guess[i] = 1
                    try_right_syndrome = syndrome ^ (Apart.dot(try_guess) % 2)
                    try_right_correction = self.decode_right(try_right_syndrome)
                    try_correction = np.hstack((try_guess, try_right_correction))
                    try_conflicts = self.count_conflicts(syndrome, try_correction)
                    if try_conflicts < best_conflicts:
                        best_conflicts = try_conflicts
                        best_correction = try_correction
                        best_guess = try_guess.copy()
            if best_conflicts == cur_conflicts:
                break
            else:
                cur_conflicts = best_conflicts
                cur_guess = best_guess

        best_correction_left = best_correction
        best_conflicts_left = best_conflicts

        # Greedy search for R
        cur_guess = np.zeros(n // 2, dtype=int)
        cur_left_syndrome = syndrome ^ (Bpart.dot(cur_guess) % 2)
        cur_left_correction = self.decode_left(cur_left_syndrome)
        cur_correction = np.hstack((cur_left_correction, cur_guess))
        cur_conflicts = self.count_conflicts(syndrome, cur_correction)
        best_conflicts = cur_conflicts
        best_correction = cur_correction
        best_guess = cur_guess.copy()

        for _ in range(1, order + 1):
            for i in range(n // 2):
                if cur_guess[i] == 0:
                    try_guess = cur_guess.copy()
                    try_guess[i] = 1
                    try_left_syndrome = syndrome ^ (Bpart.dot(try_guess) % 2)
                    try_left_correction = self.decode_left(try_left_syndrome)
                    try_correction = np.hstack((try_left_correction, try_guess))
                    try_conflicts = self.count_conflicts(syndrome, try_correction)
                    if try_conflicts < best_conflicts:
                        best_conflicts = try_conflicts
                        best_correction = try_correction
                        best_guess = try_guess.copy()
            if best_conflicts == cur_conflicts:
                break
            else:
                cur_conflicts = best_conflicts
                cur_guess = best_guess

        if best_conflicts_left < best_conflicts:
            best_correction = best_correction_left
            best_conflicts = best_conflicts_left

        err_measure = (self.code.hx.dot(best_correction) % 2) ^ syndrome
        err_correction = np.hstack((best_correction, err_measure))
        return err_correction, best_conflicts


class DecoupledHGDecoder(Decoder):
    def __init__(self, code, p, **kwargs):
        super().__init__("Decouple_HG_Decoder_" + kwargs.get("decoders_mode", "both"))
        self.code = code
        self.p = p
        self.deltaA = csr_matrix(code.h1)  # 稀疏矩阵
        self.mA, self.nA = self.deltaA.shape
        self.decoders_mode = kwargs.get("decoders_mode", "both")
        self.deltaB = csr_matrix(code.h2)  # 稀疏矩阵
        self.mB, self.nB = self.deltaB.shape

        # 初始化解码器
        self.decoderA = guass_decoder(mode=self.decoders_mode)
        h_A = hstack([self.deltaA, identity(self.mA, format='csr')])
        self.decoderA.set_h(h_A, prior=[None], p=p)
        self.decoderB_T = guass_decoder(mode=self.decoders_mode)
        h_BT = hstack([self.deltaB.T[:self.mB, :], identity(self.mB, format='csr')])
        self.decoderB_T.set_h(h_BT, prior=[None], p=p)

    @timing(decoder_info="HG Decoder", log_file="timing.log")
    def decode(self, syndrome):
        return self.adaptiveSearch(syndrome, 3)[0]

    def count_conflicts(self, syndrome, correction):
        """计算修正后的错误分布中有多少个错误"""
        return np.sum(syndrome ^ (self.code.hx.dot(correction) % 2)) + np.sum(correction)

    def adaptiveSearch(self, syndrome, order):
        """自适应搜索
        对于 AL + BR = S, 先针对 L 从 0 到 order 进行贪心搜索，再针对 R 进行解码
        另一种情况，先针对 R 从 0 到 order 进行贪心搜索，再针对 L 进行解码
        两种情况分别并行计算，最后合并结果
        """
        Apart = self.code.hx[:, :self.nA * self.nB]
        Bpart = self.code.hx[:, self.nA * self.nB:]

        # Greedy search for L
        cur_guess = np.zeros(self.nA * self.nB, dtype=int)
        cur_right_syndrome = syndrome ^ (Apart.dot(cur_guess) % 2)
        cur_right_correction = self.decode_right(cur_right_syndrome)
        cur_correction = np.hstack((cur_guess, cur_right_correction))
        cur_conflicts = self.count_conflicts(syndrome, cur_correction)
        best_conflicts = cur_conflicts
        best_correction = cur_correction
        best_guess = cur_guess.copy()

        for _ in range(1, order + 1):
            for i in range(self.nA * self.nB):
                if cur_guess[i] == 0:
                    try_guess = cur_guess.copy()
                    try_guess[i] = 1
                    try_right_syndrome = syndrome ^ (Apart.dot(try_guess) % 2)
                    try_right_correction = self.decode_right(try_right_syndrome)
                    try_correction = np.hstack((try_guess, try_right_correction))
                    try_conflicts = self.count_conflicts(syndrome, try_correction)
                    if try_conflicts < best_conflicts:
                        best_conflicts = try_conflicts
                        best_correction = try_correction
                        best_guess = try_guess.copy()
            if best_conflicts == cur_conflicts:
                break
            else:
                cur_conflicts = best_conflicts
                cur_guess = best_guess

        best_correction_left = best_correction
        best_conflicts_left = best_conflicts

        # Greedy search for R
        cur_guess = np.zeros(self.mA * self.mB, dtype=int)
        cur_left_syndrome = syndrome ^ (Bpart.dot(cur_guess) % 2)
        cur_left_correction = self.decode_left(cur_left_syndrome)
        cur_correction = np.hstack((cur_left_correction, cur_guess))
        cur_conflicts = self.count_conflicts(syndrome, cur_correction)
        best_conflicts = cur_conflicts
        best_correction = cur_correction
        best_guess = cur_guess.copy()

        for _ in range(1, order + 1):
            for i in range(self.mA * self.mB):
                if cur_guess[i] == 0:
                    try_guess = cur_guess.copy()
                    try_guess[i] = 1
                    try_left_syndrome = syndrome ^ (Bpart.dot(try_guess) % 2)
                    try_left_correction = self.decode_left(try_left_syndrome)
                    try_correction = np.hstack((try_left_correction, try_guess))
                    try_conflicts = self.count_conflicts(syndrome, try_correction)
                    if try_conflicts < best_conflicts:
                        best_conflicts = try_conflicts
                        best_correction = try_correction
                        best_guess = try_guess.copy()
            if best_conflicts == cur_conflicts:
                break
            else:
                cur_conflicts = best_conflicts
                cur_guess = best_guess

        # Select the best solution
        if best_conflicts_left < best_conflicts:
            best_correction = best_correction_left
            best_conflicts = best_conflicts_left

        err_measure = (self.code.hx.dot(best_correction) % 2) ^ syndrome
        err_correction = np.hstack((best_correction, err_measure))
        return err_correction, best_conflicts

    def decode_left(self, syndrome):
        """解码左部分"""
        syndrome = syndrome.reshape(self.mA, self.nB)
        L = lil_matrix((self.nA, self.nB), dtype=int)  # 使用稀疏矩阵
        for col in range(self.nB):
            syndrome_col = syndrome[:, col]
            correction = self.decoderA.decode(syndrome_col)  # 调用经典解码器
            L[:, col] = correction[:self.nA]
        return L.toarray().flatten() 

    def decode_right(self, syndrome):
        """解码右部分"""
        syndrome = syndrome.reshape(self.mA, self.nB)
        R = lil_matrix((self.mA, self.mB), dtype=int)  # 使用稀疏矩阵
        for row in range(self.mA):
            syndrome_row = syndrome[row, :]
            correction = self.decoderB_T.decode(syndrome_row[:self.mB])  # 调用转置解码器
            R[row, :] = correction[:self.mB]
        return R.toarray().flatten() 
                
def DecoupledDecoder(code, p, **kwargs):
    """Factory function to create a decoder based on code type."""
    if code.codetype is None:
        raise ValueError("code.codetype is None")

    codetype = str(code.codetype).lower()

    if codetype == "bb":
        return DecoupledBBDecoder(code, p, **kwargs)
    elif codetype == "hg":
        return DecoupledHGDecoder(code, p, **kwargs)
    else:
        raise ValueError("Unsupported code type")