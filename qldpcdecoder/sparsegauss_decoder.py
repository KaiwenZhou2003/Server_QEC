import numpy as np
from scipy.sparse import csr_matrix, lil_matrix, hstack, vstack, identity, csc_matrix
from .decoder import Decoder
from .timing import timing
from .bpdecoders import BP_decoder,BPOSD_decoder
SELECT_COL = True

# 高斯消元法（mod 2），使用稀疏矩阵加速
def gauss_elimination_mod2(A):

    # 初始化列交换记录和 syndrome_transpose
    n = len(A)  # 行数
    m = len(A[0])  # 列数
   
    Augmented = A.copy()
    if n > m:
        Augmented = A[:m, :]
        n = m
    col_trans = np.arange(m)

    syndrome_transpose = np.identity(n, dtype=int)
    zero_row_counts = 0
    if SELECT_COL:
        for i in range(min(n,m)):
            # 寻找主元
            if Augmented[i, i] == 0:
                # 如果主元为0，寻找下面一行有1的列交换
                # print(i,i)
                prior_jdx = 0
                min_nonzero_counts = n
                for j in range(i+1, m):
                    if Augmented[i,j] == 1:
                        nonzero_counts = np.sum(Augmented[:,j])
                        if nonzero_counts < min_nonzero_counts:
                            prior_jdx = j
                            min_nonzero_counts = nonzero_counts
                j= prior_jdx
                col_trans[i],col_trans[j] = col_trans[j],col_trans[i]
                temp = Augmented[:,i].copy() 
                Augmented[:,i]  = Augmented[:,j]
                Augmented[:,j] = temp 
            elif Augmented[i, i] == 1:
                # 如果主元为0，寻找下面一行有1的列交换
                # print(i,i)
                prior_jdx = i
                min_nonzero_counts = np.sum(Augmented[:,i])
                for j in range(i+1, m):
                    if Augmented[i,j] == 1:
                        nonzero_counts = np.sum(Augmented[:,j])
                        if nonzero_counts < min_nonzero_counts:
                            prior_jdx = j
                            min_nonzero_counts = nonzero_counts
                j= prior_jdx
                if i == j:
                    continue
                col_trans[i],col_trans[j] = col_trans[j],col_trans[i]
                temp = Augmented[:,i].copy() 
                Augmented[:,i]  = Augmented[:,j]
                Augmented[:,j] = temp 
    
    has_all_ones = True
    for i in range(min(n,m)):
        if Augmented[i, i] == 1 and np.sum(Augmented[i]) == 1:
            continue
        else:
            has_all_ones = False
            break
    if has_all_ones:
        return Augmented, col_trans, syndrome_transpose

    for i in range(min(n,m)):
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


    """
    后处理，找到孤立的主元，把它们和前面对角线上的主元拼接到一起
    """
    start_idx = -1
    for i in range(min(n,m)):
        if Augmented[i, i] == 1:
            continue
        start_idx = i
        break

    current_idx = start_idx  # 表示当前全0行的位置
    for i in range(start_idx, min(n,m)):
        if Augmented[i, i] == 1:
            # 与上方的全0行，行交换
            temp = Augmented[current_idx, :].copy()
            Augmented[current_idx, :] = Augmented[i, :]
            Augmented[i, :] = temp

            temp2 = syndrome_transpose[i].copy()
            syndrome_transpose[i] = syndrome_transpose[current_idx]
            syndrome_transpose[current_idx] = temp2

            # 与左边的全0列，列交换
            temp = Augmented[:, current_idx].copy()
            Augmented[:, current_idx] = Augmented[:, i]
            Augmented[:, i] = temp
            col_trans[i], col_trans[current_idx] = col_trans[current_idx], col_trans[i]

            current_idx += 1

    """
    删除最后的全0行
    """
    Augmented = Augmented[: n - zero_row_counts, :]
    return Augmented, col_trans, syndrome_transpose

# 计算转换后的 syndrome
def calculate_tran_syndrome(syndrome, syndrome_transpose):
    return syndrome_transpose.dot(syndrome) % 2

# 计算原始错误
def calculate_original_error(our_result, col_trans):
    trans_results = np.zeros_like(our_result, dtype=int)
    for i in np.nonzero(our_result)[0]:
        trans_results[col_trans[i]] = 1
    # for i in range(len(col_trans)):
    #     trans_results[i] = our_result[col_trans[i]]
    return trans_results

# 计算转换后的错误
def calculate_trans_error(our_result, col_trans):
    origin_results = np.zeros_like(our_result, dtype=int)
    for i in range(len(col_trans)):
        origin_results[col_trans[i]] = our_result[i]
    return origin_results

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
            error_rate=self.p,
            channel_probs=[None],
            max_iter=100,
            bp_method="ms",
            ms_scaling_factor=0,
        )
        bp_decoder.decode(syndrome)
        return bp_decoder.bp_decoding

    def greedy_decode(self, syndrome, order=6):
        n = self.n
        cur_guess = np.zeros(n, dtype=int)
        cur_conflicts = np.sum(syndrome)
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
                    try_conflicts = np.sum(vec)
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
        super().__init__("Sparse_Gauss_" + str(kwargs.get("mode", "both")))
        self.mode = kwargs.get("mode", "both")

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
            self.ms_decoder = min_sum_decoder(self.BvIg, self.error_rate)

    @timing(decoder_info="Gauss Decoder", log_file="timing.log")
    def decode(self, syndrome, order=7):
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
        assert ((self.hz_trans @ our_result) % 2 == syndrome_copy).all()
        trans_results = calculate_original_error(our_result, self.col_trans)
        assert ((self.hz @ trans_results) % 2 == syndrome).all(), trans_results
        return trans_results