import gurobipy as gp
from gurobipy import GRB
import numpy as np

def maxcut_partition(weights):
    """
    使用 MaxCut 算法对图进行分割
    :param graph: 图的邻接表表示
    :param weights: 边的权重
    :return: 节点的划分结果（0 或 1）
    """
    # 创建模型
    model = gp.Model("maxcut")
    
    # 添加变量
    nodes = list(range(weights.shape[0]))
    x = model.addVars(nodes, vtype=GRB.BINARY, name="x")
    
    # 添加目标函数：最大化跨子图的边权重之和
    obj = gp.quicksum(weights[i, j] * (x[i] - x[j])**2 for i in nodes for j in nodes)
    ## 约束条件 1：每个节点至少属于一个子图
    model.addConstr(sum(x[i] for i in nodes) >= len(nodes)/3)
    model.addConstr(sum(1-x[i] for i in nodes) >= len(nodes)/3)
    ## 约束条件 2：每个子图至少包含一个节点
    model.setObjective(obj, GRB.MINIMIZE)
    
    # 求解
    model.optimize()
    
    # 提取结果
    partition = {i: int(x[i].X) for i in nodes}
    print(f"最优分割结果：{partition}")
    return partition

def split_ldpc_with_maxcut(H, k=2):
    """
    使用 MaxCut 算法对 LDPC 码进行分割
    :param H: 校验矩阵
    :param k: 分割的子图数量
    :return: 子图和共享边
    """
    from itertools import combinations
    # 构建图模型
    weights = {}
    num_rows, num_cols = H.shape
    rows, cols = H.nonzero()
    weights = np.zeros((num_cols, num_cols))
    for i in range(num_rows):
        nonzerocols = np.nonzero(H[i])[0]
        for j, k in combinations(nonzerocols, 2):
            weights[j, k] += 1
            weights[k, j] += 1
        
    # 使用 MaxCut 算法进行分割
    partition = maxcut_partition(weights)
    
    # 分配变量节点和校验节点到子图
    clusters = [{'vars': set(), 'checks': set()} for _ in range(k)]
    for node, cid in partition.items():
        if node < H.shape[1]:  # 变量节点
            clusters[cid]['vars'].add(node)
        else:  # 校验节点
            clusters[cid]['checks'].add(node - H.shape[1])
    
    # 构建子图
    subgraphs = []
    for cid in range(k):
        sub_vars = sorted(clusters[cid]['vars'])
        sub_checks = sorted(clusters[cid]['checks'])
        sub_H = H[sub_checks, :][:, sub_vars]
        subgraphs.append({
            'variables': sub_vars,
            'checks': sub_checks,
            'H_sub': sub_H
        })
    
    # 检测共享边
    shared_edges = []
    for i, j in zip(rows, cols):
        c_cluster = partition.get(i + H.shape[1], -1)
        v_cluster = partition.get(j, -1)
        if c_cluster != v_cluster:
            shared_edges.append({
                'check': i,
                'var': j,
                'check_cluster': c_cluster,
                'var_cluster': v_cluster
            })
    
    return subgraphs, shared_edges


if __name__ == "__main__":
    # 示例矩阵
    from bposd.hgp import hgp
    from ldpc.codes import rep_code, ring_code,hamming_code, create_bivariate_bicycle_codes
    p = 0.01
    h1 = ring_code(6)
    h2 = ring_code(5)
    css_code = hgp(h1=h1, h2=h2, compute_distance=True)
    bb_code, A_list, B_list = create_bivariate_bicycle_codes(
            6, 6, [3], [2, 1], [1, 2], [3]
        ) 
    
    print('BB code K,L,Q', bb_code.K, bb_code.L, bb_code.Q)
    print('Logical Qubits', bb_code.lz)
    exit()
    # 使用 MaxCut 算法进行分割
    subgraphs, shared_edges = split_ldpc_with_maxcut(bb_code.hz, k=2)
    
    print("【子图结构】")
    for i, sub in enumerate(subgraphs):
        print(f"子图{i}: 变量{sub['variables']} 校验{sub['checks']}")
        print(f"校验矩阵:\n{sub['H_sub']}\n")
    
    print("【跨子图连接】")
    for edge in shared_edges:
        print(f"校验c{edge['check']}(子图{edge['check_cluster']}) -> "
              f"变量v{edge['var']}(子图{edge['var_cluster']})")
    print(f"共享的边共{len(shared_edges)}条")