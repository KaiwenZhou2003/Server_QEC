from z3 import *
import numpy as np
from functools import reduce

# 计算多个项的异或链（GF(2)中的加法）
def gf2_xor(terms):
    return reduce(lambda a,b: Xor(a,b), terms, False)

def determinant(matrix):
    """递归计算方阵的行列式（Z3表达式）"""
    n = len(matrix)
    if n == 1:
        return matrix[0][0]
    elif n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    else:
        det = 0
        for col in range(n):
            submatrix = [row[:col] + row[col+1:] for row in matrix[1:]]
            sub_det = determinant(submatrix)
            det += ((-1) ** col) * matrix[0][col] * sub_det
        return det

def solve_flexible_block_transform(H_np, row_partition, col_partition):
    """
    允许自定义行和列分块，求解变换矩阵 T，使得 TH 的前 m=sum(col_partition) 列满足分块约束。
    
    参数:
        H_np: 输入矩阵 (l x 2l), NumPy数组
        row_partition: 行分块列表（如 [3,3] 表示分为两个3行的块）
        col_partition: 列分块列表（如 [6,4] 表示分为6列和4列的块）
        k: 最后 k 列不受约束（总列数需满足 sum(col_partition) + k = 2l）
    
    返回:
        T: 变换矩阵 (l x l), NumPy数组
        TH: 分块矩阵 (l x 2l), NumPy数组
    """
    l, n = H_np.shape
    m = sum(col_partition)
    print(m)
    assert len(row_partition) == len(col_partition), "行分片与列分片长度需一致"
    k = n -m
    p = l - sum(row_partition)
    print(k)
    # 转换 H 为 Z3 布尔矩阵
    H = [[BoolVal(bool(H_np[i,j])) for j in range(n)] for i in range(l)]
    
    # 初始化求解器和变量
    opt = Solver()
    T = [[Bool(f"T_{i}_{j}") for j in range(l)] for i in range(l)]  # 变换矩阵
    
    for i in range(l):
        opt.add(Sum([T[i][j] for j in range(l)]) == 1)
        opt.add(Sum([T[j][i] for j in range(l)]) == 1)
    
    col_tran = [[Bool(f"col_tran_{i}_{j}") for i in range(n)] for j in range(n)]  # 列变换
    ## 置换矩阵col_tran特点： 每一行每一列都只有一个 1
    for i in range(n):
        opt.add(Sum([col_tran[i][j] for j in range(n)]) == 1)
        opt.add(Sum([col_tran[j][i] for j in range(n)]) == 1)
    
    
    TH = []
    ## get TH
    for i in range(l):
        THrow = []
        for j in range(n):
            THrow.append(gf2_xor([And(T[i][ki],H[ki][j]) for ki in range(l)]))
        TH.append(THrow)
    
    THC = []
    for i in range(l):
        THCrow = []
        for j in range(n):
            THCrow.append(gf2_xor([And(TH[i][ki],col_tran[ki][j]) for ki in range(n)]))
        THC.append(THCrow)
        
    current_row, current_col = 0, 0
    for r, c in zip(row_partition, col_partition):
        # 当前列不能表示为其他列的线性组合（XOR）
        for i in range(current_row, current_row + r):
            for j in range(current_col + c, n-k):
                opt.add(THC[i][j] == False)
        for i in range(current_row + r,l-p):
            for j in range(current_col, current_col + c):
                opt.add(THC[i][j] == False)
        current_row += r
        current_col += c


    # 求解并提取结果
    if opt.check() == sat:
        model = opt.model()
        T_result = np.array([[1 if is_true(model.evaluate(T[i][j])) else 0 for j in range(l)] for i in range(l)])
        C_result = np.array([[1 if is_true(model.evaluate(col_tran[i][j])) else 0 for j in range(n)] for i in range(n)])
        THC_result = np.array([[1 if is_true(model.evaluate(THC[i][j])) else 0 for j in range(n)] for i in range(l)])
        return T_result,C_result,THC_result
    else:
        print(opt.check())
        return None, None,None

# 测试案例
if __name__ == "__main__":
    # 示例矩阵 (6x12)
    from ldpc.codes import rep_code, ring_code,hamming_code, create_bivariate_bicycle_codes
    
    bb_code, A_list, B_list = create_bivariate_bicycle_codes(
            6, 6, [3], [1, 2], [1, 2], [3]
        )  # 72

    # print(bb_code.hz)
    print(bb_code.lx)
    row_part = [12,12,12]
    col_part = [12,12,12]

    T,C,THC = solve_flexible_block_transform(bb_code.hz[:,:36], row_part, col_part)
    with open(f"results/T_BBleft_{bb_code.name}.txt", "w") as file:
        for row in T:
            file.write(",".join(map(str, row)) + "\n")
    with open(f"results/C_BBleft_{bb_code.name}.txt", "w") as file:
        for row in C:
            file.write(",".join(map(str, row)) + "\n")
    with open(f"results/THC_BBleft_{bb_code.name}.txt", "w") as file:
        for row in THC:
            file.write(",".join(map(str, row)) + "\n")