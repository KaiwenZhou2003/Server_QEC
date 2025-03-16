import numpy as np
from .decoder import Decoder
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
    def __init__(self, code_h, error_rate, **kwargs):
        super().__init__("Gauss")
        
        self.hz = code_h
        self.error_rate = error_rate
        self.pre_decode()
        pass

    def pre_decode(self):
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
            np.log((1 - p) / p) for i in range(self.hz.shape[1])
        ]  # 初始每个qubit的对数似然比
        assert np.all([w > 0 for w in weights])
        Ig = np.identity(len(self.hz_trans[0]) - len(self.hz_trans))
        self.BvIg = np.vstack([self.B, Ig])
        self.ms_decoder = min_sum_decoder(self.BvIg, self.error_rate)

    def decode(self, syndrome,order=3):
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
        g,_= self.ms_decoder.greedy_decode(g_syn, order=order)  # 传入g_syn = [s', 0]
        # g = self.ms_decoder.our_bp_decode(g_syn)
        f = (np.dot(self.B, g) + syndrome_copy) % 2
        our_result = np.hstack((f, g))
        # assert ((self.hz_trans @ our_result) % 2 == syndrome_copy).all()
        trans_results = calculate_original_error(our_result, self.col_trans)
        # assert ((self.hz @ trans_results) % 2 == syndrome).all(), trans_results
        return trans_results


################################################################################################################

@ray.remote
def one_test(surface_code,p):
    # generate error
    from ldpc import bposd_decoder,bp_decoder

    bposd_num_success = 0
    bp_num_success = 0
    our_num_success = 0
        # BP+OSD
    bposddecoder = bposd_decoder(
        surface_code.hz,
        error_rate=p,
        channel_probs=[None],
        max_iter=surface_code.N,
        bp_method="ms",
        ms_scaling_factor=0,
        osd_method="osd_e",
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
    ourdecoder = guass_decoder(surface_code.hz,error_rate=p)
    ourdecoder.pre_decode()
    N = 1000
    for i in range(N):
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
        else:
            print(np.nonzero(our_predicates)[0],np.nonzero(error)[0])
            pass
            # print(our_predicates,error)
    return bp_num_success/N,bposd_num_success/N,our_num_success/N
