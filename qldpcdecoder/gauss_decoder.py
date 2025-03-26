import numpy as np
from .decoder import Decoder
from .bpdecoders import BP_decoder
import cvxpy as cp
from z3 import And,If,Optimize,Xor,Bool,sat
import ray
import numpy as np
SELECT_COL = True

# 高斯消元法（mod 2）
def gauss_elimination_mod2(A):

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


def calculate_tran_syndrome(syndrome, syndrome_transpose):
    return syndrome_transpose @ syndrome % 2


def calculate_original_error(our_result, col_trans):
    trans_results = np.zeros_like(our_result, dtype=int)
    col_trans = col_trans.tolist()
    for i in np.arange(len(col_trans)):
        trans_results[i] = our_result[col_trans.index(i)]
    return trans_results

def calculate_trans_error(our_result, col_trans):
    origin_results = np.zeros_like(our_result, dtype=int)
    col_trans = col_trans.tolist()
    for i in np.arange(len(col_trans)):
        origin_results[col_trans.index(i)] = our_result[i]
    return origin_results




 

class min_sum_decoder:
    def __init__(self,hz,p):
        self.hz = hz
        self.p = p
        pass
    def count_conflicts(self,syndrome,error):
        hzg = (self.hz @ error)%2
        hzg = hzg.astype(int)
        return np.sum( hzg^ syndrome)


    def sdp_relaxation_decode(self, syndrome,**kwargs):
        """
        使用半定规划松弛来求解尽可能满足最多线性方程的问题。
        A: 线性方程组的系数矩阵 (m x n)
        b: 方程组的常数项向量 (m,)
        返回: 近似解向量 x
        """
        m, n = self.hz.shape
        
        # 定义变量：x 是我们要求解的解向量（二进制）
        x = cp.Variable(n, boolean=True)  # 求解的是一个二进制向量
        
        # 定义目标函数：最小化违反的约束个数
        objective = cp.Minimize(cp.sum([cp.norm(self.hz[i, :] @ x - syndrome[i], 'fro') for i in range(m)]))
        
        # 定义约束条件：我们需要一个可行解，这里我们使用半定规划松弛的思想来转换问题
        constraints = []
        
        # 求解SDP松弛问题
        prob = cp.Problem(objective, constraints)
        
        prob.solve()
        
        return x.value




    def z3_decode(self, syndrome,**kwargs):
        """
        使用Z3求解器来求解尽可能多的线性方程 (mod 2) 满足约束的问题。
        A: 线性方程组的系数矩阵 (m x n)
        b: 方程组的常数项向量 (m,)
        返回: 最优解的二进制向量 x 和满足的约束数量
        """
        m, n = self.hz.shape
        
        # 定义布尔变量 x, 对应于解向量
        x = [Bool(f"x_{i}") for i in range(n)]
        hz = self.hz.astype(bool).tolist()
        # 创建Z3求解器
        solver = Optimize()
        syndromez3 = syndrome.astype(bool).tolist()
        total_error = 0
        for i in range(m):
            # 计算 A_i * x_i (mod 2)
            equation = [And(x[j], If(hz[i][j],True,False)) for j in range(n)]
            mod_2_result = equation[0]
            for j in range(1,n):
                mod_2_result = Xor(mod_2_result,equation[j])
            
            # 计算误差 (A_i * x_i) % 2 ⊕ b_i
            error = Xor(mod_2_result, syndromez3[i])
            
            # 将误差加入到目标函数中
            total_error += error
        
        # 添加优化目标：最小化总误差
        from time import perf_counter
        start = perf_counter()
        solver.minimize(total_error)
        
    
        if solver.check() == sat:
            duration = perf_counter()-start
            print(f'z3 solve takes {duration} s')
            model = solver.model()
            solution = [model[x_i] for x_i in x]
            # min_error = solver.objective_value()
            solution = np.array([True if x else False for x in solution]).astype(int)
            return solution
        else:
            return np.zeros(n,dtype=int)
    def greedy_decode_approx(self, syndrome, order=3):
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
    
    def our_bp_decode(self, syndrome, **kwargs):
        """
        对于[B, I]g=[s', 0]，先调用greedy_decoder，找到一个近似解，然后用bp_decoder进行迭代
        """
        from ldpc import bp_decoder, bposd_decoder

        bp_decoder = bp_decoder(
            self.hz,
            error_rate=self.p,
            channel_probs=[None],
            max_iter=100,
            bp_method="ms",  # minimum sum
            ms_scaling_factor=0,
        )

        bp_decoder.decode(syndrome)
        our_result = bp_decoder.bp_decoding

        # print(f"greedy g HW = {np.sum(g)}, bp g HW = {np.sum(our_result)}")
        return our_result



    def greedy_decode(self,syndrome,order=6,frozen_idx=[]):
        """ 贪心算法，目前效果最优
        """
        n = len(self.hz[0])
        cur_guess = np.zeros(n,dtype=int)
        cur_conflicts = self.count_conflicts(syndrome,cur_guess)
        best_conflicts = cur_conflicts
        best_guess = cur_guess
        for _ in range(1,order+1):
            for i in range(n):
                if i in frozen_idx:
                    continue
                if cur_guess[i]==0:
                    try_guess = cur_guess.copy()
                    try_guess[i] =1
                    try_conflicts = self.count_conflicts(syndrome,try_guess)
                    if try_conflicts < best_conflicts:
                        best_conflicts = try_conflicts
                        best_guess = try_guess
            if best_conflicts == cur_conflicts:
                break
            else:
                cur_conflicts = best_conflicts
                cur_guess = best_guess
        
        return best_guess,best_conflicts
    
    def frozen_greedy_decode(self,syndrome,order=6,max_iter=10):
        cur_guess,cur_conflicts = self.greedy_decode(syndrome,order=order)
        frozen_bits = []
        best_conflicts = cur_conflicts
        best_guess = cur_guess
        for i in range(max_iter):
            if best_conflicts > 3:
                frozen_bits.extend(np.nonzero(cur_guess)[0])
                print(f'Freze {frozen_bits}, conflicts {best_conflicts}')
                cur_guess,cur_conflicts = self.greedy_decode(syndrome,order=order,frozen_idx=frozen_bits)
                if cur_conflicts < best_conflicts:
                    print('try success')
                    best_conflicts = cur_conflicts
                    best_guess = cur_guess
            else:
                break
                
        return best_guess,best_conflicts
        


class guass_decoder(Decoder):
    def __init__(self, **kwargs):
        super().__init__("Gauss_"+str(kwargs.get("mode", "both")))
        self.mode = kwargs.get("mode", "both")
        pass

    def set_h(self, code_h,prior,p, **kwargs):
        self.hz = code_h
        self.prior = prior
        self.p = p
        self.error_rate = 1 - self.p
        self.pre_decode()
    
    def pre_decode(self):
        p = self.p
        hz_trans, col_trans, syndrome_transpose = gauss_elimination_mod2(self.hz)
        self.hz_trans = hz_trans
        print(f"hz trans rank {len(self.hz_trans)}, original {len(self.hz)}")
        self.col_trans = col_trans
        self.syndrome_transpose = syndrome_transpose
        self.B = hz_trans[:, len(hz_trans) : len(hz_trans[0])]
        # print("density of B:",np.sum(self.B)/(len(self.B)*len(self.B[0])))
        print("row density of B:", np.sum(self.B, axis=0))
        if np.mean(np.sum(self.B, axis=0)) > 5:
            print("B is dense, may cause low decoding performance, use bp decoder instead")
            bpdecoder = BP_decoder()
            bpdecoder.set_h(self.hz,self.prior, self.p)
            self.decode = bpdecoder.decode
        else:
            print(f"B shape = ({len(self.B)}, {len(self.B[0])})")
            weights = [
                np.log((1 - p) / p) for i in range(self.hz.shape[1])
            ]  # 初始每个qubit的对数似然比
            assert np.all([w > 0 for w in weights])
            Ig = np.identity(len(self.hz_trans[0]) - len(self.hz_trans))
            self.BvIg = np.vstack([self.B, Ig])
            self.ms_decoder = min_sum_decoder(self.BvIg, self.error_rate)

    def decode(self, syndrome,order=5):
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
        if self.mode == "bp" or self.mode == "both":
            g_bp = self.ms_decoder.our_bp_decode(g_syn)
            f_bp = (np.dot(self.B, g_bp) + syndrome_copy) % 2
            bp_result = np.hstack((f_bp, g_bp))
        if self.mode == "greedy" or self.mode == "both":
            g_greedy, _ = self.ms_decoder.greedy_decode(g_syn, order=order)
            f_greedy = (np.dot(self.B, g_greedy) + syndrome_copy) % 2
            greedy_result = np.hstack((f_greedy, g_greedy))
        if self.mode == "bp":
            our_result = bp_result
        elif self.mode == "greedy":
            our_result = greedy_result
        else:
            if bp_result.sum() <= greedy_result.sum():
                our_result = bp_result
            else:
                our_result = greedy_result
        # assert ((self.hz_trans @ our_result) % 2 == syndrome_copy).all()
        trans_results = calculate_original_error(our_result, self.col_trans)
        # assert ((self.hz @ trans_results) % 2 == syndrome).all(), trans_results
        return trans_results

