from .gauss_decoder import guass_decoder
from .decoder import Decoder
from .basis_compute import compute_basis_complement
import numpy as np

class ReShapeBBDecoder(Decoder):
    def __init__(self,hx,p,lz, deltaA, deltaB):
        super().__init__("Reshape_BB_Decoder")
        self.hx = hx
        self.lz = lz
        self.deltaA = deltaA  # 经典码生成矩阵 δA
        self.mA, self.nA = deltaA.shape
        
        self.deltaB = deltaB  # 经典码生成矩阵 δB
        self.mB, self.nB = deltaB.shape
        self.decoderA = guass_decoder(deltaA,p)
        _, deltaA_complete_basis = compute_basis_complement(deltaA)
        self.deltaA_complete_cols = [np.nonzero(basis)[0][0] for basis in deltaA_complete_basis]
        self.decoderB_T = guass_decoder(deltaB.T[:self.mB, :],p)  # 解码器 D_δB^T
        _, deltaBT_complete_basis = compute_basis_complement(deltaB.T)
        self.deltaBT_complete_cols = [np.nonzero(basis)[0][0] for basis in deltaBT_complete_basis]
        self.pre_decoder = guass_decoder(hx,p)

    def logical_split(self, matrix, logical_matrix):
        logic_part = logical_matrix* matrix
        free_part = matrix - logic_part
        return free_part, logic_part
        
    def split(self, matrix,is_left=True):
        """分解矩阵为自由部分和逻辑部分"""
        # 简化的基变换：实际需计算 (Im δ^T_B)^• 和 (Im δ_A)^•
        # 此处假设已预计算基变换矩阵
        if is_left:
            # 对行进行分解（左部分）
            cols = self.deltaBT_complete_cols
            logical_part = np.zeros_like(matrix)
            for col in cols:
                logical_part[:, col] = self.deltaB.T[:, col]
            free_part = matrix - logical_part
        else:
            # 对列进行分解（右部分）
            rows = self.deltaA_complete_cols
            logical_part = np.zeros_like(matrix)
            for row in rows:
                logical_part[row, :] = self.deltaA[row, :]
            free_part = matrix - logical_part
        return free_part, logical_part


    def decode(self, syndrome):
        return self.exhaustive_decode(syndrome)
    
    def logical_decode(self, syndrome):
        
        initial_solution = self.pre_decoder.decode(syndrome,order=0)
        L, R = initial_solution[:self.nA*self.nB].reshape(self.nA,self.nB),initial_solution[self.nA*self.nB:].reshape(self.mA,self.mB)
        logical_R = np.zeros_like(R)
        logical_L = np.zeros_like(L)
        for lz_arr in self.lz:
            logical_L += lz_arr[:self.nA*self.nB].reshape(self.nA,self.nB)
            logical_R += lz_arr[self.nA*self.nB:].reshape(self.mA,self.mB)
        
        # Split左部分
        ML, OL = self.logical_split(L, logical_L)
        # 处理左逻辑部分
        decoded_L = np.zeros_like(L)
        for col in range(OL.shape[1]):
            column = OL[:, col]
            syndrome_col = self.deltaA@column %2
            correction = self.decoderA.decode(syndrome_col)  # 调用经典解码器
            if np.sum(column) > np.sum(correction):
                decoded_L[:, col] = correction
                print(f"improvemented col {col} {column} -> {correction}")
            else:
                decoded_L[:, col] = column
        L_tilde = ML + decoded_L
        
        # Split右部分
        MR, OR = self.logical_split(R, logical_R)
        # 处理右逻辑部分
        decoded_R = np.zeros_like(R)
        for row in range(OR.shape[0]):
            row_vec = OR[row, :]
            syndrome_col = self.deltaB.T@row_vec % 2
            correction = self.decoderB_T.decode(syndrome_col[:self.mB])  # 调用转置解码器
            
            if np.sum(row_vec) > np.sum(correction):
                decoded_R[row, :] = correction
                print(f"improvemented row {row} {row_vec} -> {correction}")
            else:
                decoded_R[row, :] = row_vec
        R_tilde = MR + decoded_R
        
        syn_correction = np.hstack((L_tilde.reshape(-1),R_tilde.reshape(-1)))
        assert ((self.hx @ syn_correction) % 2 == syndrome).all(), syn_correction
        return syn_correction


    def reshape_decode(self, left_syndrome, right_syndrome):
        left_syndrome = left_syndrome.reshape(self.mA,self.nB)
        right_syndrome = right_syndrome.reshape(self.mA,self.nB)
        L = np.zeros((self.nA, self.nB))
        
        for col in range(self.nB):
            syndrome_col = left_syndrome[:, col]
            correction = self.decoderA.decode(syndrome_col)  # 调用经典解码器
            L[:, col] = correction
        
        R = np.zeros((self.mA, self.mB))
        for row in range(self.mA):
            syndrome_row = right_syndrome[row, :]
            correction = self.decoderB_T.decode(syndrome_row[:self.mB])  # 调用转置解码器
            R[row, :] = correction
        if np.any(R @ self.deltaB % 2 != right_syndrome):
            return None
        syn_correction = np.hstack((L.reshape(-1),R.reshape(-1)))
        return syn_correction

    def exhaustive_decode(self, syndrome):
        """exhaustive search"""
        ## 遍历所有可能的逻辑错误分布
        intersects = np.nonzero(syndrome)[0]
        conbinations = []
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
            err_correction = self.reshape_decode(left_syndrome, right_syndrome)
            if err_correction is not None:
                hw = np.sum(err_correction)
                if hw < best_hw:
                    best_hw = hw
                    best_sol = err_correction
        return best_sol


                
                
   
class ReShapeHGDecoder(Decoder):
    def __init__(self,hx,p,lz, deltaA, deltaB, mode= 'exhaustive'):
        super().__init__("Reshape"+mode)
        self.hx = hx
        self.lz = lz
        self.deltaA = deltaA  # 经典码生成矩阵 δA
        self.mA, self.nA = deltaA.shape
        
        self.deltaB = deltaB  # 经典码生成矩阵 δB
        self.mB, self.nB = deltaB.shape
        self.decoderA = guass_decoder(deltaA,p)
        _, deltaA_complete_basis = compute_basis_complement(deltaA)
        self.deltaA_complete_cols = [np.nonzero(basis)[0][0] for basis in deltaA_complete_basis]
        self.decoderB_T = guass_decoder(deltaB.T[:self.mB, :],p)  # 解码器 D_δB^T
        _, deltaBT_complete_basis = compute_basis_complement(deltaB.T)
        self.deltaBT_complete_cols = [np.nonzero(basis)[0][0] for basis in deltaBT_complete_basis]
        self.pre_decoder = guass_decoder(hx,p)
        self.mode = mode

    def logical_split(self, matrix, logical_matrix):
        logic_part = logical_matrix* matrix
        free_part = matrix - logic_part
        return free_part, logic_part
        
    def split(self, matrix,is_left=True):
        """分解矩阵为自由部分和逻辑部分"""
        # 简化的基变换：实际需计算 (Im δ^T_B)^• 和 (Im δ_A)^•
        # 此处假设已预计算基变换矩阵
        if is_left:
            # 对行进行分解（左部分）
            cols = self.deltaBT_complete_cols
            logical_part = np.zeros_like(matrix)
            for col in cols:
                logical_part[:, col] = matrix[:, col]
            free_part = matrix - logical_part
        else:
            # 对列进行分解（右部分）
            rows = self.deltaA_complete_cols
            logical_part = np.zeros_like(matrix)
            for row in rows:
                logical_part[row, :] = matrix[row, :]
            free_part = matrix - logical_part
        return free_part, logical_part


    def decode(self, syndrome):
        if self.mode == 'logical':
            return self.logical_decode(syndrome)
        elif self.mode == 'homology':
            return self.homology_decode(syndrome)
        elif self.mode =='exhaustive':
            return self.exhaustive_decode(syndrome)
        elif self.mode == 'logical_exhaustive':
            return self.logical_exhaustive_decode(syndrome)
    
    def logical_decode(self, syndrome):
        initial_solution = self.pre_decoder.decode(syndrome,order=0)
        L, R = initial_solution[:self.nA*self.nB].reshape(self.nA,self.nB),initial_solution[self.nA*self.nB:].reshape(self.mA,self.mB)
        logical_R = np.zeros_like(R)
        logical_L = np.zeros_like(L)
        for lz_arr in self.lz:
            logical_L += lz_arr[:self.nA*self.nB].reshape(self.nA,self.nB)
            logical_R += lz_arr[self.nA*self.nB:].reshape(self.mA,self.mB)
        
        # Split左部分
        ML, OL = self.logical_split(L, logical_L)
        assert np.all(OL+ML == L)
        # 处理左逻辑部分
        decoded_L = np.zeros_like(L)
        for col in range(OL.shape[1]):
            column = OL[:, col]
            syndrome_col = self.deltaA@column %2
            correction = self.decoderA.decode(syndrome_col)  # 调用经典解码器
            if np.sum(column) > np.sum(correction):
                decoded_L[:, col] = correction
                print(f"improvemented col {col} {column} -> {correction}")
            else:
                decoded_L[:, col] = column
        L_tilde = ML + decoded_L
        
        # Split右部分
        MR, OR = self.logical_split(R, logical_R)
        
        # 处理右逻辑部分
        decoded_R = np.zeros_like(R)
        for row in range(OR.shape[0]):
            row_vec = OR[row, :]
            syndrome_col = self.deltaB.T@row_vec % 2
            correction = self.decoderB_T.decode(syndrome_col[:self.mB])  # 调用转置解码器
            
            if np.sum(row_vec) > np.sum(correction):
                decoded_R[row, :] = correction
                print(f"improvemented row {row} {row_vec} -> {correction}")
            else:
                decoded_R[row, :] = row_vec
        R_tilde = MR + decoded_R
        
        syn_correction = np.hstack((L_tilde.reshape(-1),R_tilde.reshape(-1)))
        assert ((self.hx @ syn_correction) % 2 == syndrome).all(), syn_correction
        return syn_correction
    
    
    def homology_decode(self, syndrome):
        
        initial_solution = self.pre_decoder.decode(syndrome,order=0)
        L, R = initial_solution[:self.nA*self.nB].reshape(self.nA,self.nB),initial_solution[self.nA*self.nB:].reshape(self.mA,self.mB)
        
        # Split左部分
        ML, OL = self.split(L, is_left=True)
        # 处理左逻辑部分
        decoded_L = np.zeros_like(L)
        for col in range(OL.shape[1]):
            column = OL[:, col]
            syndrome_col = self.deltaA@column %2
            correction = self.decoderA.decode(syndrome_col)  # 调用经典解码器
            if np.sum(column) > np.sum(correction):
                decoded_L[:, col] = correction
                print(f"improvemented col {col} {column} -> {correction}")
            else:
                decoded_L[:, col] = column
        L_tilde = ML + decoded_L
        
        # Split右部分
        MR, OR = self.split(R, is_left=False)
        # 处理右逻辑部分
        decoded_R = np.zeros_like(R)
        for row in range(OR.shape[0]):
            row_vec = OR[row, :]
            syndrome_col = self.deltaB.T@row_vec % 2
            correction = self.decoderB_T.decode(syndrome_col[:self.mB])  # 调用转置解码器
            
            if np.sum(row_vec) > np.sum(correction):
                decoded_R[row, :] = correction
                print(f"improvemented row {row} {row_vec} -> {correction}")
            else:
                decoded_R[row, :] = row_vec
        R_tilde = MR + decoded_R
        
        syn_correction = np.hstack((L_tilde.reshape(-1),R_tilde.reshape(-1)))
        assert ((self.hx @ syn_correction) % 2 == syndrome).all(), syn_correction
        return syn_correction

    def reshape_decode(self, left_syndrome, right_syndrome):
        left_syndrome = left_syndrome.reshape(self.mA,self.nB)
        right_syndrome = right_syndrome.reshape(self.mA,self.nB)
        L = np.zeros((self.nA, self.nB))
        
        for col in range(self.nB):
            syndrome_col = left_syndrome[:, col]
            correction = self.decoderA.decode(syndrome_col)  # 调用经典解码器
            L[:, col] = correction
        
        R = np.zeros((self.mA, self.mB))
        for row in range(self.mA):
            syndrome_row = right_syndrome[row, :]
            correction = self.decoderB_T.decode(syndrome_row[:self.mB])  # 调用转置解码器
            R[row, :] = correction
        if np.any(R @ self.deltaB % 2 != right_syndrome):
            return None
        syn_correction = np.hstack((L.reshape(-1),R.reshape(-1)))
        return syn_correction

    def exhaustive_decode(self, syndrome):
        """exhaustive search"""
        ## 遍历所有可能的逻辑错误分布
        # logical_R = np.zeros((self.nA,self.nB))
        # logical_L = np.zeros((self.mA,self.mB))
        # for lz_arr in self.lz:
        #     logical_L += lz_arr[:self.nA*self.nB].reshape(self.nA,self.nB)
        #     logical_R += lz_arr[self.nA*self.nB:].reshape(self.mA,self.mB)
        # if logical_R.sum() == 0:
        #     logical_qubits = np.nonzero(logical_L.reshape(-1))[0]
        #     intersects = set(np.nonzero(syndrome)[0]).intersection(set(logical_qubits))
        intersects = np.nonzero(syndrome)[0]
        if len(intersects) > 8:
            print("too many intersects for exhaustive search")
        conbinations = []
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
            err_correction = self.reshape_decode(left_syndrome, right_syndrome)
            if err_correction is not None:
                hw = np.sum(err_correction)
                if hw < best_hw:
                    best_hw = hw
                    best_sol = err_correction
        return best_sol
    
    def logical_exhaustive_decode(self, syndrome):
        """exhaustive search"""
        intersects = np.nonzero(syndrome)[0]
        conbinations = []
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
            err_correction = self.reshape_decode(left_syndrome, right_syndrome)
            if err_correction is not None:
                hw = np.sum(self.lz,axis=0) @ err_correction
                if hw < best_hw:
                    best_hw = hw
                    best_sol = err_correction
        return best_sol
                
            

        
        
        
            