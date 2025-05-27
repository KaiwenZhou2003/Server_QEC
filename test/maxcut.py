import numpy as np
import os
import matplotlib.pyplot as plt
from qldpcdecoder.decoupling.maxcutall import split_ldpc_with_maxcut

# 示例矩阵
mat =np.loadtxt('results/lastmat.csv',delimiter=',',dtype=int)

# 使用 MaxCut 算法进行分割
mat_, _,_, row_lens, col_lens = split_ldpc_with_maxcut(mat, k=2)
plt.imshow(mat_, cmap='gist_yarg')
plt.savefig('results/lastmat_maxcut.png')
# left_mat = mat_[:row_lens[0],:col_lens[0]]
# matleft_, _,_, row_lens, col_lens = split_ldpc_with_maxcut(left_mat, k=2)
# plt.imshow(matleft_, cmap='gist_yarg')
# plt.savefig('results/lastmat_left_maxcut.png')