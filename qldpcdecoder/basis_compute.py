import numpy as np

def row_echelon(mat, reduced=False):
    r"""Converts a binary matrix to (reduced) row echelon form via Gaussian Elimination,
    also works for rank-deficient matrix. Unlike the make_systematic method,
    no column swaps will be performed.

    Input
    ----------
    mat : ndarry
        A binary matrix in numpy.ndarray format.
    reduced: bool
        Defaults to False. If true, the reduced row echelon form is returned.

    Output
    -------
    row_ech_form: ndarray
        The row echelon form of input matrix.
    rank: int
        The rank of the matrix.
    transform: ndarray
        The transformation matrix such that (transform_matrix@matrix)=row_ech_form
    pivot_cols: list
        List of the indices of pivot num_cols found during Gaussian elimination
    """

    m, n = np.shape(mat)
    # Don't do "m<=n" check, allow over-complete matrices
    mat = np.copy(mat)
    # Convert to bool for faster arithmetics
    mat = mat.astype(bool)
    transform = np.identity(m).astype(bool)
    pivot_row = 0
    pivot_cols = []

    # Allow all-zero column. Row operations won't induce all-zero columns, if they are not present originally.
    # The make_systematic method will swap all-zero columns with later non-all-zero columns.
    # Iterate over cols, for each col find a pivot (if it exists)
    for col in range(n):
        # Select the pivot - if not in this row, swap rows to bring a 1 to this row, if possible
        if not mat[pivot_row, col]:
            # Find a row with a 1 in this column
            swap_row_index = pivot_row + np.argmax(mat[pivot_row:m, col])
            # If an appropriate row is found, swap it with the pivot. Otherwise, all zeroes - will loop to next col
            if mat[swap_row_index, col]:
                # Swap rows
                mat[[swap_row_index, pivot_row]] = mat[[pivot_row, swap_row_index]]
                # Transformation matrix update to reflect this row swap
                transform[[swap_row_index, pivot_row]] = transform[
                    [pivot_row, swap_row_index]
                ]

        if mat[pivot_row, col]:  # will evaluate to True if this column is not all-zero
            if not reduced:  # clean entries below the pivot
                elimination_range = [k for k in range(pivot_row + 1, m)]
            else:  # clean entries above and below the pivot
                elimination_range = [k for k in range(m) if k != pivot_row]
            for idx_r in elimination_range:
                if mat[idx_r, col]:
                    mat[idx_r] ^= mat[pivot_row]
                    transform[idx_r] ^= transform[pivot_row]
            pivot_row += 1
            pivot_cols.append(col)

        if pivot_row >= m:  # no more rows to search
            break

    rank = pivot_row
    row_ech_form = mat.astype(int)

    return [row_ech_form, rank, transform.astype(int), pivot_cols]

def compute_basis_complement(delta):
    """
    输入：矩阵 delta（类型为 numpy.ndarray 或 sympy.Matrix）
    输出：补集空间的基矩阵 basis_complement
    """
    # 转置以处理行空间
    m, n = delta.shape
    [_,rank, _, _] = row_echelon(delta.T)
    if m == rank:
        return np.identity(m), []
    if m > rank:
        # 构造增广矩阵 [delta^T | I_n]
        augmented = np.vstack([delta.T, np.eye(m)])
        # 计算 RREF
        [row_ech_form, _, transform, pivot_cols] = row_echelon(augmented)
        # 提取非主元列对应的单位向量
        basis_complement = row_ech_form[rank:m]
        basis = row_ech_form[:rank]
    
        return basis,basis_complement

if __name__ == '__main__':
    # 示例计算 basisB_free
    deltaB = np.array([[1,0,1], [0,1,1]])  # 假设 δB 是 2×3 矩阵
    basisB_free = compute_basis_complement(deltaB)
    print("BasisB_free:\n", basisB_free)

    # 类似地计算 basisA_free（需替换为 δA 的列空间补集）
    deltaA = np.array([[1,1,0], [0,1,1]])
    basisA_free = compute_basis_complement(deltaA.T)  # 转置以处理列空间
    print("BasisA_free:\n", basisA_free)
