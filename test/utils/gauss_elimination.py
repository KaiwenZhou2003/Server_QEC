import numpy as np
import math
SELECT_COL = True

# 高斯消元法（mod 2）
def gauss_elimination_mod2(A):

    with open("origin_bbcode.txt", "w") as file:
        for row in A:
            file.write(" ".join(map(str, row)) + "\n")
        file.write("\n\n")
    print("nozero counts:",np.sum(A,axis=1))

    n = len(A)  # 行数
    m = len(A[0])  # 列数
    Augmented = A.copy()
    col_trans = np.arange(m)

    syndrome_transpose = np.identity(n, dtype=int)
    zero_row_counts = 0
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
    
    for i in range(n):
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

    with open("Augmented.txt", "w") as file:
        for row in Augmented:
            file.write(" ".join(map(str, row)) + "\n")
        file.write("\n\n")

    """
    后处理，找到孤立的主元，把它们和前面对角线上的主元拼接到一起
    """
    start_idx = -1
    for i in range(n):
        if Augmented[i, i] == 1:
            continue
        start_idx = i
        break

    current_idx = start_idx  # 表示当前全0行的位置
    for i in range(start_idx, n):
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
    print("find zero rows:", zero_row_counts)
    # syndrome_transpose = syndrome_transpose[:n-zero_row_counts,:]
    return Augmented, col_trans, syndrome_transpose


def calculate_tran_syndrome(syndrome, syndrome_transpose):
    return syndrome_transpose @ syndrome % 2


def calculate_original_error(our_result, col_trans):
    trans_results = np.zeros_like(our_result, dtype=int)
    col_trans = col_trans.tolist()
    for i in np.arange(len(col_trans)):
        trans_results[i] = our_result[col_trans.index(i)]
    return trans_results
