
import numpy as np
import numpy as np
from mqt.qecc import *  # UFDecoder

# 高斯消元法（mod 2）
def gauss_elimination_mod2(A):
    n = len(A)
    m = len(A[0])
    Augmented = A.copy()
    col_trans = np.arange(m)
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
    for i in range(n):
        # 对主元所在行进行消元
        if Augmented[i, i] == 1:
            for j in range(0, n):
                if j!=i and Augmented[j, i] == 1:
                    Augmented[j] ^= Augmented[i]
                    syndrome_transpose[j] ^= syndrome_transpose[i]

    return Augmented,col_trans,syndrome_transpose

def calculate_tran_syndrome(syndrome,syndrome_transpose):
    return syndrome_transpose @ syndrome %2

def calculate_original_error(our_result,col_trans):
    trans_results = np.zeros_like(our_result,dtype=int)
    col_trans = col_trans.tolist()
    for i in np.arange(len(col_trans)):
        trans_results[i] = our_result[col_trans.index(i)]
    return trans_results







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
        self.col_trans = col_trans
        self.syndrome_transpose = syndrome_transpose
        B = hz_trans[:,len(hz_trans):len(hz_trans[0])]
        weights = [
            np.log((1 - p) / p) for i in range(H_X.shape[1])
        ]  # 初始每个qubit的对数似然比

        W_f = weights[: H_X.shape[0]]
        W_g = weights[H_X.shape[0] :]

        W_f_B = np.dot(W_f, B)  # W_f * B
        W_g_B_W_f = W_f_B + W_g  # W_f * B + W_g
        # print(f"W_g_B_W_f = {W_g_B_W_f}")

        self.zero_g = np.where(
            W_g_B_W_f > 0,
            0,
            np.where(W_g_B_W_f < 0, 1, np.random.randint(0, 2, size=W_g_B_W_f.shape)),
        )
        # print(f"g = {g}")

        self.B_g = np.dot(B, self.zero_g)  # B * g
        # print(f"B_g = {B_g}")
        
    
    def decode(self,syndrome):
        syndrome_copy = calculate_tran_syndrome(syndrome.copy(),self.syndrome_transpose)
        f = (self.B_g + syndrome_copy)%2
        our_result = np.hstack((f, self.zero_g))
        assert ((self.hz_trans @ our_result)%2 == syndrome_copy).all()
        trans_results = calculate_original_error(our_result,self.col_trans)
        assert ((self.hz @ trans_results)%2 == syndrome).all(), (trans_results)
        return trans_results
    
def test_decoder(num_trials,surface_code,p,ourdecoder):
    num_trials = 10000
    from ldpc import bposd_decoder
    p = 0.1  # 错误率

    # BP+OSD
    bposddecoder = bposd_decoder(
        surface_code.hz,
        error_rate=p,
        channel_probs=[None],
        max_iter=surface_code.N,
        bp_method="ms",
        ms_scaling_factor=0,
        osd_method="osd_cs",
        osd_order=7,
    )

    # UFDecoder
    code = Code(surface_code.hx, surface_code.hz)
    uf_decoder = UFHeuristic()
    uf_decoder.set_code(code)
    bposd_num_success = 0
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
        # 1. BP+OSD
        bposddecoder.decode(syndrome)
        bposd_result =  bposddecoder.osdw_decoding
        bposd_residual_error = (bposddecoder.osdw_decoding + error) % 2
        bpflag = (surface_code.lz @ bposd_residual_error % 2).any()
        if bpflag == 0:
            bposd_num_success += 1

        # 2. UFDecoder
        uf_decoder.decode(syndrome)
        uf_result = np.array(uf_decoder.result.estimate).astype(int)
        uf_residual_error = (uf_result + error) % 2
        flag = (surface_code.lz @ uf_residual_error % 2).any()
        if flag == 0:
            uf_num_success += 1
        # 3. Our Decoder
        our_predicates = ourdecoder.decode(syndrome)
        our_residual_error = (our_predicates + error) % 2
        flag = (surface_code.lz @ our_residual_error % 2).any()
        if flag == 0:
            our_num_success += 1

        # g_index = col_trans[len(hz_trans):len(hz_trans[0])]
        # bposd_g = [bposd_result[idx] for idx in g_index]
        # True_g = [error[idx] for idx in g_index ]
        # random_g = np.zeros_like(zero_g)
        # for idx in range(len(random_g)):
        #     if np.random.rand() < p:
        #         random_g[idx] = 1
        
        
            

    bposd_success_rate = bposd_num_success / num_trials
    uf_success_rate = uf_num_success / num_trials
    our_success_rate = our_num_success / num_trials
    print(f"\nTotal trials: {num_trials}")
    print(f"BP+OSD Success rate: {bposd_success_rate * 100:.2f}%")
    print(f"UF Success rate: {uf_success_rate * 100:.2f}%")
    print(f"Our Success rate: {our_success_rate * 100:.2f}%")

if __name__ == "__main__":
    from ldpc.codes import rep_code
    from bposd.hgp import hgp
    h = rep_code(5)
    surface_code = hgp(h1=h, h2=h, compute_distance=True)
    surface_code.test()
    p =0.05
    ourdecoder = guass_decoder(surface_code.hz,error_rate=p)
    ourdecoder.pre_decode()
    test_decoder(num_trials=1000,surface_code=surface_code,p=p,ourdecoder=ourdecoder)
    
    