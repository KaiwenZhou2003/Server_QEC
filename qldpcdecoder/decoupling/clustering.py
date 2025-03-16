import numpy as np
from scipy import sparse
from sklearn.cluster import SpectralClustering
import networkx as nx
from bposd.hgp import hgp
from ldpc.codes import rep_code, ring_code,hamming_code, create_bivariate_bicycle_codes
def build_bipartite_graph(H):
    """构建二部图的邻接表表示"""
    m, n = H.shape
    bipartite_graph = {
        'checks': [[] for _ in range(m)],
        'variables': [[] for _ in range(n)]
    }
    rows, cols = H.nonzero()
    for i, j in zip(rows, cols):
        bipartite_graph['checks'][i].append(j)
        bipartite_graph['variables'][j].append(i)
    return bipartite_graph
import numpy as np
from scipy import sparse
from sklearn.cluster import SpectralClustering

def split_ldpc(H, k=2):
    H_sparse = sparse.csr_matrix(H)
    
    # 构建变量投影图
    projection = H_sparse.T.dot(H_sparse)
    projection.setdiag(0)  # 移除自环
    
    # 谱聚类分割变量节点
    sc = SpectralClustering(n_clusters=k, affinity='precomputed', random_state=0)
    var_clusters = sc.fit_predict(projection.toarray())
    
    # 分配校验节点
    clusters = [{'vars': set(), 'checks': set()} for _ in range(k)]
    for var_idx, cid in enumerate(var_clusters):
        clusters[cid]['vars'].add(var_idx)
    
    check_connections = H_sparse.tolil().rows
    check_to_cluster = {}  # 新增：校验节点到簇的映射
    for check_idx, conn_vars in enumerate(check_connections):
        cluster_counts = np.zeros(k)
        for var in conn_vars:
            cluster_counts[var_clusters[var]] += 1
        assigned = np.argmax(cluster_counts)
        clusters[assigned]['checks'].add(check_idx)
        check_to_cluster[check_idx] = assigned  # 记录校验节点分配
    
    # 构建子图
    subgraphs = []
    for cid in range(k):
        sub_vars = sorted(clusters[cid]['vars'])
        sub_checks = sorted(clusters[cid]['checks'])
        sub_H = H_sparse[sub_checks, :][:, sub_vars]
        subgraphs.append({
            'variables': sub_vars,
            'checks': sub_checks,
            'H_sub': sub_H.toarray()
        })
    
    # 新增：检测共享边
    shared_edges = []
    rows, cols = H_sparse.nonzero()
    for ci, vj in zip(rows, cols):
        c_cluster = check_to_cluster.get(ci, -1)
        v_cluster = var_clusters[vj]
        if c_cluster != v_cluster:
            shared_edges.append({
                'check': ci,
                'var': vj,
                'check_cluster': c_cluster,
                'var_cluster': v_cluster
            })
    
    return subgraphs, shared_edges

from sklearn.cluster import AgglomerativeClustering
import numpy as np
def row_clustering(H, k=2):
    """对校验矩阵的行进行层次聚类"""
    # 计算行之间的Jaccard相似度
    similarity = np.zeros((H.shape[0], H.shape[0]))
    for i in range(H.shape[0]):
        for j in range(i, H.shape[0]):
            intersection = np.sum(H[i] & H[j])
            union = np.sum(H[i] | H[j])
            similarity[i,j] = intersection / union if union !=0 else 0
            similarity[j,i] = similarity[i,j]
    
    # 层次聚类
    cluster = AgglomerativeClustering(n_clusters=k, linkage='average')
    row_labels = cluster.fit_predict(1 - similarity)  # 将相似度转换为距离
    return row_labels

def reorder_matrix(H, row_labels):
    """根据行聚类结果重排矩阵"""
    sorted_indices = np.argsort(row_labels)
    return H[sorted_indices, :]

def optimized_split(H, k=2):
    # 步骤1: 行聚类与重排
    row_labels = row_clustering(H, k)
    H_reordered = reorder_matrix(H, row_labels)
    
    # 步骤2: 对重排后的矩阵进行变量节点分割
    subgraphs, shared_edges = split_ldpc(H_reordered, k)
    
    # 步骤3: 记录原始索引映射关系
    row_perm = np.argsort(row_labels)  # 记录行置换顺序
    return subgraphs, shared_edges, row_perm

# 测试用例
if __name__ == "__main__":
    # 示例矩阵 (与问题描述一致)
    p = 0.01
    h1 = ring_code(6)
    h2 = ring_code(5)
    css_code = hgp(h1=h1, h2=h2, compute_distance=True)
    N = 72
    if N == 72:
        bb_code, A_list, B_list = create_bivariate_bicycle_codes(
            6, 6, [3], [1, 2], [1, 2], [3]
        )  # 72
    elif N == 90:
        bb_code, A_list, B_list = create_bivariate_bicycle_codes(
            15, 3, [9], [1, 2], [2, 7], [0]
        )  # 90
    elif N == 108:
        bb_code, A_list, B_list = create_bivariate_bicycle_codes(
            9, 6, [3], [1, 2], [1, 2], [3]
        )  # 108
    elif N == 144:
        bb_code, A_list, B_list = create_bivariate_bicycle_codes(
            12, 6, [3], [1, 2], [1, 2], [3]
        )  # 144
    elif N == 288:
        bb_code, A_list, B_list = create_bivariate_bicycle_codes(
            12, 12, [3], [2, 7], [1, 2], [3]
        )  # 288
    elif N == 360:
        bb_code, A_list, B_list = create_bivariate_bicycle_codes(
            30, 6, [9], [1, 2], [25, 26], [3]
        )  # 360
    elif N == 756:
        bb_code, A_list, B_list = create_bivariate_bicycle_codes(
            21, 18, [3], [10, 17], [3, 19], [5]
        )  # 756
    else:
        print("unsupported N")
    
    subgraphs, shared_edges = split_ldpc(bb_code.hz, k=12)
    
    print("【子图结构】")
    for i, sub in enumerate(subgraphs):
        print(f"子图{i}: 变量{sub['variables']} 校验{sub['checks']}")
        print(f"校验矩阵:\n{sub['H_sub']}\n")
    
    print("【跨子图连接】")
    for edge in shared_edges:
        print(f"校验c{edge['check']}(子图{edge['check_cluster']}) -> "
              f"变量v{edge['var']}(子图{edge['var_cluster']})")
    print(f"共享的边共{len(shared_edges)}条")