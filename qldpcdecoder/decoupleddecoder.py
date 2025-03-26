from .gauss_decoder import guass_decoder
from .decoder import Decoder
from .basis_compute import compute_basis_complement
import numpy as np
import os


        

class DecoupledBBDecoder(Decoder):
    def __init__(self, code,p, **kwargs):
        super().__init__("Decoupled_BB_Decoder_"+kwargs.get("decoders_mode", "both"))
        self.code = code
        self.p = p
        self.Adecoders = []
        self.Bdecoders = []
        self.decoders_mode = kwargs.get("decoders_mode", "both")
        self.load_decoupled_matrix()
    
    def load_decoupled_matrix(self):
        pathdir = "results/"+self.code.name+"/"
        if not os.path.exists(pathdir):
            print("No decoupled matrix found, please run decouple matrix first")
        self.A_T = np.load(pathdir+"A_T.npy")
        self.A_C = np.load(pathdir+"A_C.npy")
        self.A_THC = np.load(pathdir+"A_THC.npy")
        self.A_row_window = np.load(pathdir+"A_row_part.npy")
        self.A_col_window = np.load(pathdir+"A_col_part.npy")
        ## find the block window
        self.A_anchors = []
        start_row = 0
        start_col = 0
        
        for r,c in zip(self.A_row_window,self.A_col_window):
            block = self.A_THC[start_row:start_row+r,start_col:start_col+c]
            self.Adecoders.append(guass_decoder(mode=self.decoders_mode))
            h = np.hstack((block,np.identity(block.shape[0])))
            self.Adecoders[-1].set_h(h,prior=[None],p=self.p)
            self.A_anchors.append([(start_row,start_col),(start_row+r,start_col+c)])
            start_row += r
            start_col += c
    
        self.B_T = np.load(pathdir+"B_T.npy")
        self.B_C = np.load(pathdir+"B_C.npy")
        self.B_THC = np.load(pathdir+"B_THC.npy")
        self.B_row_window = np.load(pathdir+"B_row_part.npy")
        self.B_col_window = np.load(pathdir+"B_col_part.npy")
        self.B_anchors = []
        start_row = 0
        start_col = 0
        for r,c in zip(self.B_row_window,self.B_col_window):
            block = self.B_THC[start_row:start_row+r,start_col:start_col+c]
            self.B_anchors.append([(start_row,start_col),(start_row+r,start_col+c)])
            self.Bdecoders.append(guass_decoder(mode=self.decoders_mode))
            h = np.hstack((block,np.identity(block.shape[0])))
            self.Bdecoders[-1].set_h(h,prior=[None],p=self.p)
            start_row += r
            start_col += c
        
    def decode(self, syndrome):
        # return self.exhaustive_decode(syndrome)
        return self.adaptiveSearch(syndrome, 3)[0]

    def reshape_decode(self, l_syndrome, r_syndrome):
        corrections = []
        left_syndrome = self.A_T @ l_syndrome
        for i, anchor in enumerate(self.A_anchors):
            syndrome_block = left_syndrome[anchor[0][0]:anchor[1][0]]
            error_correction = self.Adecoders[i].decode(syndrome_block)
            corrections.append(error_correction[:anchor[1][1]-anchor[0][1]])
        left_correction = np.hstack(corrections)
        left_correction = self.A_C @ left_correction
        corrections = []
        right_syndrome = self.B_T @ r_syndrome
        for j, anchor in enumerate(self.B_anchors):
            syndrome_block = right_syndrome[anchor[0][0]:anchor[1][0]]
            error_correction = self.Bdecoders[j].decode(syndrome_block)
            corrections.append(error_correction[:anchor[1][1]-anchor[0][1]])
        right_correction = np.hstack(corrections)
        
        right_correction = self.B_C @ right_correction
        syn_correction = np.hstack((left_correction, right_correction))
        return syn_correction
    
    def decode_right(self, syndrome):
        """ 解码右半部分 """
        corrections = []
        right_syndrome = self.B_T @ syndrome
        for j, anchor in enumerate(self.B_anchors):
            syndrome_block = right_syndrome[anchor[0][0]:anchor[1][0]]
            error_correction = self.Bdecoders[j].decode(syndrome_block)
            corrections.append(error_correction[:anchor[1][1]-anchor[0][1]])
        right_correction = np.hstack(corrections)
        right_correction = self.B_C @ right_correction
        return right_correction

    def decode_left(self, syndrome):
        """ 解码左半部分 """
        corrections = []
        left_syndrome = self.A_T @ syndrome
        for i, anchor in enumerate(self.A_anchors):
            syndrome_block = left_syndrome[anchor[0][0]:anchor[1][0]]
            error_correction = self.Adecoders[i].decode(syndrome_block)
            corrections.append(error_correction[:anchor[1][1]-anchor[0][1]])
        left_correction = np.hstack(corrections)
        left_correction = self.A_C @ left_correction
        return left_correction
    
    def count_conflicts(self, syndrome, correction):
        """ 计算修正后的错误分布中有多少个错误 """
        return np.sum(syndrome ^ (self.code.hx @ correction) % 2) + np.sum(correction)
    
    def adaptiveSearch(self, syndrome, order):
        """ 自适应搜索
        对于 AL + BR = S, 先针对 L 从 0 到 order 进行贪心搜索， 再针对R 进行解码
        另一种情况，先针对 R 从 0 到 order 进行贪心搜索， 再针对L 进行解码
        两种情况分别并行计算，最后合并结果
        """
        m,n = self.code.hx.shape
        Apart = self.code.hx[:, :n//2]
        Bpart = self.code.hx[:, n//2:]
        ## gredy search for L
        cur_guess = np.zeros(n//2,dtype=int)
        cur_right_syndrome = syndrome ^ (Apart @ cur_guess) % 2
        cur_right_correction = self.decode_right(cur_right_syndrome)
        cur_correction = np.hstack((cur_guess, cur_right_correction))
        cur_conflicts = self.count_conflicts(syndrome,cur_correction)
        best_conflicts = cur_conflicts
        best_correction = cur_correction
        best_guess = cur_guess.copy()
        for _ in range(1,order+1):
            for i in range(n//2):
                if cur_guess[i] == 0:
                    try_guess = cur_guess.copy()
                    try_guess[i] = 1
                    try_right_syndrome = syndrome ^ ((Apart @ try_guess) %2)
                    try_right_correction = self.decode_right(try_right_syndrome)
                    try_correction = np.hstack((try_guess, try_right_correction))
                    try_conflicts = self.count_conflicts(syndrome,try_correction)
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
        ## greddy search for R
        cur_guess = np.zeros(n//2,dtype=int)
        cur_left_syndrome = syndrome ^ (Bpart @ cur_guess) % 2
        cur_left_correction = self.decode_left(cur_left_syndrome)
        cur_correction = np.hstack((cur_left_correction, cur_guess))
        cur_conflicts = self.count_conflicts(syndrome,cur_correction)
        best_conflicts = cur_conflicts
        best_correction = cur_correction
        best_guess = cur_guess.copy()
        for _ in range(1,order+1):
            for i in range(n//2):
                if cur_guess[i] == 0:
                    try_guess = cur_guess.copy()
                    try_guess[i] = 1
                    try_left_syndrome = syndrome ^ ((Bpart @ try_guess) %2)
                    try_left_correction = self.decode_left(try_left_syndrome)
                    try_correction = np.hstack((try_left_correction, try_guess))
                    try_conflicts = self.count_conflicts(syndrome,try_correction)
                    if try_conflicts < best_conflicts:
                        best_conflicts = try_conflicts
                        best_correction = try_correction
                        best_guess = try_guess.copy()
            if best_conflicts == cur_conflicts:
                break
            else:
                cur_conflicts = best_conflicts
                cur_guess = best_guess
        ## select the best solution
        if best_conflicts_left < best_conflicts:
            best_correction = best_correction_left
            best_conflicts = best_conflicts_left
            
        err_measure = (self.code.hx @ best_correction) % 2 ^ syndrome
        err_correction = np.hstack((best_correction, err_measure))
        return err_correction, best_conflicts
       
        
    
    def exhaustive_decode(self, syndrome):
        """exhaustive search"""
        ## 遍历所有可能的逻辑错误分布
        intersects = np.nonzero(syndrome)[0]
        conbinations = []
        if len(intersects) == 0:
            return np.zeros(self.code.hx.shape[0]+self.code.hx.shape[1],dtype=int)
        for i in range(2**len(intersects)):
            left_syndrome = np.zeros_like(syndrome)
            left_syndrome[intersects] = [int(x) for x in bin(i)[2:].zfill(len(intersects))]
            right_syndrome = syndrome.copy()
            right_syndrome = syndrome ^ left_syndrome
            conbinations.append((left_syndrome, right_syndrome))
        
        # 遍历所有可能的错误分布
        best_hw = np.inf
        best_sol = None
        for left_syndrome, right_syndrome in conbinations:
            # 左部分
            err = self.reshape_decode(left_syndrome, right_syndrome)
            
            err_measure = (self.code.hx @ err) % 2 ^ syndrome
            err_correction = np.hstack((err, err_measure))
            if err_correction is not None:
                hw = np.sum(err_correction)
                if hw < best_hw:
                    best_hw = hw
                    best_sol = err_correction
        return best_sol

      
                
class DecoupledHGDecoder(Decoder):
    def __init__(self,code,p, **kwargs):
        super().__init__("Decouple_HG_Decoder_"+kwargs.get("decoders_mode", "both"))
        self.code = code
        self.p = p
        self.deltaA = code.h1
        self.mA, self.nA = self.deltaA.shape
        self.decoders_mode = kwargs.get("decoders_mode", "both")
        self.deltaB = code.h2 # 经典码生成矩阵 δB
        self.mB, self.nB = self.deltaB.shape
        self.decoderA = guass_decoder(mode=self.decoders_mode)
        self.decoderA.set_h(np.hstack((self.deltaA, np.identity(self.mA))),prior=[None],p=p)
        _, deltaA_complete_basis = compute_basis_complement(self.deltaA)
        self.deltaA_complete_cols = [np.nonzero(basis)[0][0] for basis in deltaA_complete_basis]
        self.decoderB_T = guass_decoder(mode=self.decoders_mode)  # 解码器 D_δB^T
        self.decoderB_T.set_h(np.hstack((self.deltaB.T[:self.mB, :],np.identity(self.mB))),prior=[None],p=p)
        _, deltaBT_complete_basis = compute_basis_complement(self.deltaB.T)
        self.deltaBT_complete_cols = [np.nonzero(basis)[0][0] for basis in deltaBT_complete_basis]
    def decode(self, syndrome):
        # return self.exhaustive_decode(syndrome)
        return self.adaptiveSearch(syndrome, 3)[0]
    
    def count_conflicts(self, syndrome, correction):
        """ 计算修正后的错误分布中有多少个错误 """
        return np.sum(syndrome ^ (self.code.hx @ correction) % 2) + np.sum(correction)
    
    def adaptiveSearch(self, syndrome, order):
        """ 自适应搜索
        对于 AL + BR = S, 先针对 L 从 0 到 order 进行贪心搜索， 再针对R 进行解码
        另一种情况，先针对 R 从 0 到 order 进行贪心搜索， 再针对L 进行解码
        两种情况分别并行计算，最后合并结果
        """
        Apart = self.code.hx[:, :self.nA*self.nB]
        Bpart = self.code.hx[:, self.nA*self.nB:]
        ## gredy search for L
        cur_guess = np.zeros(self.nA*self.nB,dtype=int)
        cur_right_syndrome = syndrome ^ (Apart @ cur_guess) % 2
        cur_right_correction = self.decode_right(cur_right_syndrome)
        cur_correction = np.hstack((cur_guess, cur_right_correction))
        cur_conflicts = self.count_conflicts(syndrome,cur_correction)
        best_conflicts = cur_conflicts
        best_correction = cur_correction
        best_guess = cur_guess.copy()
        for _ in range(1,order+1):
            for i in range(self.nA*self.nB):
                if cur_guess[i] == 0:
                    try_guess = cur_guess.copy()
                    try_guess[i] = 1
                    try_right_syndrome = syndrome ^ ((Apart @ try_guess) %2)
                    try_right_correction = self.decode_right(try_right_syndrome)
                    try_correction = np.hstack((try_guess, try_right_correction))
                    try_conflicts = self.count_conflicts(syndrome,try_correction)
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
        ## greddy search for R
        cur_guess = np.zeros(self.mA*self.mB,dtype=int)
        cur_left_syndrome = syndrome ^ (Bpart @ cur_guess) % 2
        cur_left_correction = self.decode_left(cur_left_syndrome)
        cur_correction = np.hstack((cur_left_correction, cur_guess))
        cur_conflicts = self.count_conflicts(syndrome,cur_correction)
        best_conflicts = cur_conflicts
        best_correction = cur_correction
        best_guess = cur_guess.copy()
        for _ in range(1,order+1):
            for i in range(self.mA*self.mB):
                if cur_guess[i] == 0:
                    try_guess = cur_guess.copy()
                    try_guess[i] = 1
                    try_left_syndrome = syndrome ^ ((Bpart @ try_guess) %2)
                    try_left_correction = self.decode_left(try_left_syndrome)
                    try_correction = np.hstack((try_left_correction, try_guess))
                    try_conflicts = self.count_conflicts(syndrome,try_correction)
                    if try_conflicts < best_conflicts:
                        best_conflicts = try_conflicts
                        best_correction = try_correction
                        best_guess = try_guess.copy()
            if best_conflicts == cur_conflicts:
                break
            else:
                cur_conflicts = best_conflicts
                cur_guess = best_guess
        ## select the best solution
        if best_conflicts_left < best_conflicts:
            best_correction = best_correction_left
            best_conflicts = best_conflicts_left
            
        err_measure = (self.code.hx @ best_correction) % 2 ^ syndrome
        err_correction = np.hstack((best_correction, err_measure))
        return err_correction, best_conflicts
    
    def decode_left(self, syndrome):
        """ 解码左部分 """
        syndrome = syndrome.reshape(self.mA,self.nB)
        L = np.zeros((self.nA, self.nB), dtype=int)
        for col in range(self.nB):
            syndrome_col = syndrome[:, col]
            correction = self.decoderA.decode(syndrome_col)  # 调用经典解码器
            L[:, col] = correction[:self.nA]
        return L.reshape(-1)
    
    def decode_right(self, syndrome):
        """ 解码右部分 """
        syndrome = syndrome.reshape(self.mA,self.nB)
        R = np.zeros((self.mA, self.mB), dtype=int)
        for row in range(self.mA):
            syndrome_row = syndrome[row, :]
            correction = self.decoderB_T.decode(syndrome_row[:self.mB])  # 调用转置解码器
            R[row, :] = correction[:self.mB]
        return R.reshape(-1)


            

        
        
        
      
                
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