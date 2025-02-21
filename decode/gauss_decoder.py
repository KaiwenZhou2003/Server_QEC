import numpy as np
from mqt.qecc import *  # UFDecoder
import math
import cvxpy as cp
from z3 import And,If,Optimize,Xor,Bool,sat
from functools import reduce
import ray
SELECT_COL = False
# 高斯消元法（mod 2）
def gauss_elimination_mod2(A):
    n = len(A)
    m = len(A[0])
    print("nozero counts:",np.sum(A,axis=0))
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
    
    syndrome_transpose= np.identity(n,dtype=int)
    zero_row_counts = 0
    for i in range(n):
        # 对主元所在行进行消元
        if Augmented[i, i] == 1:
            for j in range(0, n):
                if j!=i and Augmented[j, i] == 1:
                    Augmented[j] ^= Augmented[i]
                    syndrome_transpose[j] ^= syndrome_transpose[i]
        else:
            # 如果主元为0，寻找下面一行有1的列交换
            # print(i,i)
            prior_jdx = i
            min_nonzero_counts = n
            for j in range(i+1, m):
                if Augmented[i,j] == 1:
                    nonzero_counts = np.sum(Augmented[:,j])
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
            col_trans[i],col_trans[prior_jdx] = col_trans[prior_jdx],col_trans[i]
            temp = Augmented[:,i].copy() 
            Augmented[:,i]  = Augmented[:,prior_jdx]
            Augmented[:,prior_jdx] = temp 
            
            
            ## 继续消元
            for j in range(0, n):
                if j!=i and Augmented[j, i] == 1:
                    Augmented[j] ^= Augmented[i]
                    syndrome_transpose[j] ^= syndrome_transpose[i]
    Augmented = Augmented[:n-zero_row_counts,:]
    # print("find zero rows",zero_row_counts)
    # syndrome_transpose = syndrome_transpose[:n-zero_row_counts,:]
    return Augmented,col_trans,syndrome_transpose

def calculate_tran_syndrome(syndrome,syndrome_transpose):
    return syndrome_transpose @ syndrome %2

def calculate_original_error(our_result,col_trans):
    trans_results = np.zeros_like(our_result,dtype=int)
    col_trans = col_trans.tolist()
    for i in np.arange(len(col_trans)):
        trans_results[i] = our_result[col_trans.index(i)]
    return trans_results






class min_sum_decoder:
    def __init__(self,hz,p):
        self.hz = hz
        from ldpc import bposd_decoder,bp_decoder
        self.bp_decoder = bp_decoder(
            self.hz,
            error_rate=p,
            channel_probs=[None],
            max_iter=len(self.hz[0]),
            bp_method="ms",  # minimum sum
            ms_scaling_factor=0,
        )
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
        
        
    # # 定义约束：对于每个方程 A_i * x = b_i
    #     satisfied_constraints = []
    #     for i in range(m):
    #         # 计算 A_i * x_i
    #         equation = [self.hz[i, j] * If(x[j], 1, 0) for j in range(n)]
    #         sum_eq = Sum(equation) % 2
            
    #         # 方程约束：A_i * x = b_i (mod 2)
    #         solver.add(sum_eq == b[i])
    #         satisfied_constraints.append(sum_eq == b[i])

        # 目标是最大化满足的约束数量
        # 由于Z3求解器本身并不直接支持计数约束，我们可以通过
        # 构造一个优化问题来间接实现
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
        for _ in range(1,order+1):
            best_conflicts = cur_conflicts
            best_guess = cur_guess
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
        
    
    def bp_decode(self,syndrome,**kwarg):
        self.bp_decoder.decode(syndrome)
        return   self.bp_decoder.bp_decoding
    
    def simulated_annealing_decode(self,syndrome, initial_temp=1000, final_temp=1, alpha=0.9, max_iter=10000,**kwargs):
        """
        使用模拟退火算法找到近似解，使得尽可能多的方程得到满足。
        A: 线性方程组的系数矩阵
        b: 线性方程组的常数项
        
        这个方法效果非常差
        """
        m, n = self.hz.shape
        # 初始解：全为0的解
        current_solution = np.zeros(n, dtype=int)
        current_obj_value = self.count_conflicts(syndrome,current_solution)
        
        best_solution = current_solution.copy()
        best_obj_value = current_obj_value
            
        # 初始温度
        temperature = initial_temp

        for iteration in range(max_iter):
            # 在邻域中随机选择一个新的解
            new_solution = current_solution.copy()
            random_bit = np.random.randint(0, n-1)  # 随机选择一个变量
            new_solution[random_bit] = 1 - new_solution[random_bit]  # 翻转该变量
            
            # 计算目标函数值
            new_obj_value = self.count_conflicts(syndrome, new_solution)
            
            # 如果新解更优，则接受
            if new_obj_value < current_obj_value:
                current_solution = new_solution
                current_obj_value = new_obj_value
            else:
                # 否则，以一定概率接受新解
                probability = math.exp((current_obj_value - new_obj_value) / temperature)
                if np.random.random() < probability:
                    current_solution = new_solution
                    current_obj_value = new_obj_value
            
            # 更新最佳解
            if current_obj_value < best_obj_value:
                best_solution = current_solution.copy()
                best_obj_value = current_obj_value
            
            # 降低温度
            temperature *= alpha
            
            # 如果温度足够低，停止
            if temperature < final_temp:
                break
        
        return best_solution


class guass_decoder:
    def __init__(self,code_h,error_rate,**kwargs):
        self.hz = code_h
        self.error_rate = error_rate
        pass
    
    def pre_decode(self):
        H_X = self.hz
        p = self.error_rate
        hz_trans, col_trans, syndrome_transpose = gauss_elimination_mod2(self.hz)
        self.hz_trans = hz_trans
        # print(f"hz trans rank {len(self.hz_trans)}, original {len(self.hz)}")
        self.col_trans = col_trans
        self.syndrome_transpose = syndrome_transpose
        self.B = hz_trans[:,len(hz_trans):len(hz_trans[0])]
        # print("density of B:",np.sum(self.B)/(len(self.B)*len(self.B[0])))
        # print("row density of B:",np.sum(self.B,axis=0))
        # print(f"{len(self.B)},{len(self.B[0])}")
        weights = [
            np.log((1 - p) / p) for i in range(H_X.shape[1])
        ]  # 初始每个qubit的对数似然比
        assert np.all([w > 0 for w in weights])
        Ig = np.identity(len(self.hz_trans[0])-len(self.hz_trans))
        self.BvIg = np.vstack([self.B,Ig])
        self.ms_decoder = min_sum_decoder(self.BvIg,self.error_rate)
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
        
    
    def decode(self,syndrome):
        syndrome_copy = calculate_tran_syndrome(syndrome.copy(),self.syndrome_transpose)
        syndrome_copy = syndrome_copy[:len(self.hz_trans)]
        g_syn = np.hstack([syndrome_copy,np.zeros(len(self.hz_trans[0])-len(self.hz_trans),dtype=int)])
        g,_ = self.ms_decoder.frozen_greedy_decode(g_syn,order=10)
        f = (np.dot(self.B, g) + syndrome_copy)%2
        our_result = np.hstack((f, g))
        assert ((self.hz_trans @ our_result)%2 == syndrome_copy).all()
        trans_results = calculate_original_error(our_result,self.col_trans)
        assert ((self.hz @ trans_results)%2 == syndrome).all(), (trans_results)
        return trans_results

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

def test_decoder(num_trials,surface_code,p):



    # UFDecoder
    code = Code(surface_code.hx, surface_code.hz)
    # uf_decoder = UFHeuristic()
    # uf_decoder.set_code(code)

    results = ray.get([one_test.remote(surface_code,p) for i in range(int(num_trials/1000))])
    res = np.sum(results,axis=0)/(int(num_trials/1000))
    print(res)

        
        
        
            

    bposd_error_rate = 1- res[1]
    bp_error_rate = 1- res[0]
    # uf_error_rate = 1- uf_num_success / num_trials
    our_error_rate = 1- res[2]
    print(f"Logical error rate: {1/num_trials:.9f}")
    print(f"BP error rate: {bp_error_rate :.9f}")
    print(f"BP+OSD error rate: {bposd_error_rate :.9f}")
    # print(f"UF Success rate: {uf_error_rate * 100:.2f}%")
    print(f"Our error rate: {our_error_rate :.9f}")

if __name__ == "__main__":
    np.random.seed(0)
    from ldpc.codes import rep_code,ring_code,hamming_code
    from bposd.hgp import hgp,hgp_single
    h = ring_code(5)
    h2 = rep_code(7)
    h3 = hgp_single(h1=h,compute_distance=True)
    surface_code = hgp(h1=h2, h2=h3.hz, compute_distance=True)
    print(surface_code.hz)
    print(surface_code.lz)
    # print(surface_code.hz @ surface_code.lx.T)
    print("-"*30)
    
    # print(surface_code.hx)
    
    # print(surface_code.lx)
    # surface_code = hgp(h1=surface_code.hz,h2 =surface_code.hz, compute_distance= True)
    surface_code.test()
    p =0.001
    print(surface_code.hz.shape)
    
    # print(ourdecoder.hz_trans)
    test_decoder(num_trials=100000,surface_code=surface_code,p=p)

    
    