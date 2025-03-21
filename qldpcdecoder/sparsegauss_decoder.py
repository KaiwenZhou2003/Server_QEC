import numpy as np
from scipy.sparse import csr_matrix, lil_matrix, hstack, vstack, identity
from .decoder import Decoder
from .timing import timing
SELECT_COL = True

# 高斯消元法（mod 2），使用稀疏矩阵加速
def gauss_elimination_mod2(A):
    A = A.tocsc()  # 使用 CSC 格式，便于列操作
    n, m = A.shape  # 行数和列数

    # 初始化列交换记录和 syndrome_transpose
    col_trans = np.arange(m)
    syndrome_transpose = identity(n, dtype=int, format='csc')
    zero_row_counts = 0

    for i in range(min(n, m)):
        # 寻找主元
        if A[i, i] == 0:
            # 如果主元为 0，寻找下面一行有 1 的列交换
            prior_jdx = i
            min_nonzero_counts = n
            for j in range(i + 1, m):
                if A[i, j] == 1:
                    nonzero_counts = A[:, j].sum()
                    if nonzero_counts < min_nonzero_counts:
                        prior_jdx = j
                        min_nonzero_counts = nonzero_counts
            if prior_jdx == i:
                zero_row_counts += 1
                continue
            # 交换列
            col_trans[i], col_trans[prior_jdx] = col_trans[prior_jdx], col_trans[i]
            A[:, [i, prior_jdx]] = A[:, [prior_jdx, i]]

        # 对主元所在行进行消元
        if A[i, i] == 1:
            for j in range(n):
                if j != i and A[j, i] == 1:
                    A[j] = (A[j] + A[i]) % 2
                    syndrome_transpose[j] = (syndrome_transpose[j] + syndrome_transpose[i]) % 2

    # 后处理，删除全 0 行
    A = A[:n - zero_row_counts, :]
    return A, col_trans, syndrome_transpose

# 计算转换后的 syndrome
def calculate_tran_syndrome(syndrome, syndrome_transpose):
    return syndrome_transpose.dot(syndrome) % 2

# 计算原始错误
def calculate_original_error(our_result, col_trans):
    trans_results = np.zeros_like(our_result, dtype=int)
    for i in range(len(col_trans)):
        trans_results[i] = our_result[col_trans[i]]
    return trans_results

# 计算转换后的错误
def calculate_trans_error(our_result, col_trans):
    origin_results = np.zeros_like(our_result, dtype=int)
    for i in range(len(col_trans)):
        origin_results[col_trans[i]] = our_result[i]
    return origin_results

class min_sum_decoder:
    def __init__(self, hz, p):
        self.hz = hz  # 稀疏矩阵
        self.p = p

    def count_conflicts(self, syndrome,flip_idx):
        return np.sum(self.hz[:,flip_idx]^syndrome)

    def our_bp_decode(self, syndrome, **kwargs):
        from ldpc import bp_decoder
        bp_decoder = bp_decoder(
            self.hz,
            error_rate=self.p,
            channel_probs=[None],
            max_iter=100,
            bp_method="ms",
            ms_scaling_factor=0,
        )
        bp_decoder.decode(syndrome)
        return bp_decoder.bp_decoding

    def greedy_decode(self, syndrome, order=6):
        n = self.hz.shape[1]
        cur_guess = np.zeros(n, dtype=int)
        cur_conflicts = np.sum(syndrome)
        best_conflicts = cur_conflicts
        best_guess = cur_guess
        for _ in range(1, order + 1):
            best_addconflicts = 0
            for i in range(n):
                if cur_guess[i] == 0:
                    try_guess = cur_guess.copy()
                    try_guess[i] = 1
                    try_addconflicts = self.count_conflicts(syndrome, i)
                    if try_addconflicts < best_addconflicts:
                        best_addconflicts = try_addconflicts
                        best_guess = try_guess
            best_conflicts += best_addconflicts
            if best_addconflicts == 0:
                break
            else:
                cur_conflicts = best_conflicts
                cur_guess = best_guess
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
        super().__init__("Gauss_" + str(kwargs.get("mode", "both")))
        self.mode = kwargs.get("mode", "both")

    def set_h(self, code_h, prior, p, **kwargs):
        self.hz = csr_matrix(code_h)  # 转换为稀疏矩阵
        self.prior = prior
        self.p = p
        self.error_rate = 1 - self.p
        self.pre_decode()

    def pre_decode(self):
        hz_trans, col_trans, syndrome_transpose = gauss_elimination_mod2(self.hz)
        self.hz_trans = hz_trans
        self.col_trans = col_trans
        self.syndrome_transpose = syndrome_transpose
        self.B = hz_trans[:, hz_trans.shape[0]:]
        self.BvIg = vstack([self.B, identity(self.B.shape[1],dtype=int)]).toarray().astype(int)
        self.ms_decoder = min_sum_decoder(self.BvIg, self.error_rate)

    @timing(decoder_info="Gauss Decoder", log_file="timing.log")
    def decode(self, syndrome, order=3):
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
        return calculate_original_error(our_result, self.col_trans)