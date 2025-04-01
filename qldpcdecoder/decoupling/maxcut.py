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
    model.addConstr(sum(x[i] for i in nodes) >= len(nodes)/5)
    model.addConstr(sum(1-x[i] for i in nodes) >= len(nodes)/5)
    ## 约束条件 2：每个子图至少包含一个节点
    model.setObjective(obj, GRB.MINIMIZE)
    
    # 求解
    model.optimize()
    
    # 提取结果
    partition = { i: int(x[i].X) for i in nodes}
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
    weights = np.zeros((num_cols, num_cols))
    for i in range(num_rows):
        nonzerocols = np.nonzero(H[i])[0]
        for j, k in combinations(nonzerocols, 2):
            # if weights[j, k] == 0:
            #     weights[j, k] = 1
            #     weights[k, j] = 1
            # else:
            #     continue
            weights[j, k] += 1
            weights[k, j] += 1
        
    # 使用 MaxCut 算法进行分割
    partition = maxcut_partition(weights)
    
    # 分配变量节点和校验节点到子图
    graphs = {i: [] for i in range(k)}
    for node, cid in partition.items():
        graphs[cid].append(node)
    ## order the matrix by the graph id
    new_matrix = np.zeros((num_rows, num_cols))
    col_idx = 0
    mapping = {}
    for _,nodes in graphs.items():
        for j in nodes:
            new_matrix[:, col_idx] = H[:, j]
            col_idx += 1
            mapping[col_idx] = j
    
    return new_matrix, mapping

