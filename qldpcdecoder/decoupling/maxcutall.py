import gurobipy as gp
from gurobipy import GRB
import numpy as np

def maxcut_partition(weights, K):
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
    if K == 2:
        x = model.addVars(nodes, vtype=GRB.BINARY, name="x")
        obj = gp.quicksum(weights[i, j] * (x[i] - x[j])**2 for i in nodes for j in nodes)
        ## 约束条件 1：每个节点至少属于一个子图
        model.addConstr(gp.quicksum(x[i] for i in nodes) >= len(nodes)//3)
        model.addConstr(gp.quicksum(1-x[i] for i in nodes)  >= len(nodes)//3)
        ## 约束条件 2：每个子图至少包含一个节点
        model.setObjective(obj, GRB.MINIMIZE)
    else:
        x = model.addVars(nodes, K, vtype=GRB.BINARY, name="x")

        obj = gp.quicksum(weights[i, j] * (x[i,k] - x[j,k])**2 for i in nodes for j in nodes for k in range(K))
        ## 约束条件 1：每个节点至少属于一个子图
        for j in range(K):
            model.addConstr(gp.quicksum(x[i,j] for i in nodes) >= len(nodes)//(2*K))
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
    num_nodes = num_rows + num_cols
    weights = np.zeros((num_cols+num_rows, num_cols+num_rows))
    for i in range(num_rows):
        nonzerocols = np.nonzero(H[i])[0]
        for j in nonzerocols:
            weights[i, j+num_rows] = 1
            weights[j+num_rows, i] = 1
           
        
    # 使用 MaxCut 算法进行分割
    partition = maxcut_partition(weights, k)
    
    # 分配变量节点和校验节点到子图
    graphrows = {i: [] for i in range(k)}
    graphcols = {i: [] for i in range(k)}
    for node, cid in partition.items():
        if node < num_rows:
            graphrows[cid].append(node)
        else:
            graphcols[cid].append(node-num_rows)
    ## order the matrix by the graph id
    new_matrix_row = np.zeros((num_rows, num_cols))
    row_idx = 0
    row_mapping = {}
    for _,nodes in graphrows.items():
        for j in nodes:
            new_matrix_row[ row_idx,:] = H[ j,:]
            row_mapping[row_idx] = j
            row_idx += 1
            
    new_matrix_col = np.zeros((num_rows, num_cols))
    col_idx = 0
    col_mapping = {}
    for _,nodes in graphcols.items():
        for j in nodes:
            new_matrix_col[:, col_idx] = new_matrix_row[:, j]
            col_mapping[col_idx] = j
            col_idx += 1
            
    
         
            
    
    return new_matrix_col, row_mapping, col_mapping, [len(row) for row in graphrows.values()],[len(col) for col in graphcols.values()]

