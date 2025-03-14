import numpy as np
import math

SELECT_COL = False


# 高斯消元法（mod 2）
def gauss_elimination_mod2(A):
    n = len(A)
    m = len(A[0])
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
                for j in range(i + 1, m):
                    if Augmented[i, j] == 1:
                        nonzero_counts = np.sum(Augmented[:, j])
                        if nonzero_counts < min_nonzero_counts:
                            prior_jdx = j
                            min_nonzero_counts = nonzero_counts
                j = prior_jdx
                col_trans[i], col_trans[j] = col_trans[j], col_trans[i]
                temp = Augmented[:, i].copy()
                Augmented[:, i] = Augmented[:, j]
                Augmented[:, j] = temp
            elif Augmented[i, i] == 1:
                # 如果主元为0，寻找下面一行有1的列交换
                # print(i,i)
                prior_jdx = i
                min_nonzero_counts = np.sum(Augmented[:, i])
                for j in range(i + 1, m):
                    if Augmented[i, j] == 1:
                        nonzero_counts = np.sum(Augmented[:, j])
                        if nonzero_counts < min_nonzero_counts:
                            prior_jdx = j
                            min_nonzero_counts = nonzero_counts
                j = prior_jdx
                if i == j:
                    continue
                col_trans[i], col_trans[j] = col_trans[j], col_trans[i]
                temp = Augmented[:, i].copy()
                Augmented[:, i] = Augmented[:, j]
                Augmented[:, j] = temp

    syndrome_transpose = np.identity(n, dtype=int)
    zero_row_counts = 0
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
