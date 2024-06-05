# -*- coding: utf-8 -*-
# @Author  : 杜承泽
# @Email   : Monickar@foxmail.com

from Topology_zoo_Analog import *
from gurobipy import Model, GRB, quicksum

def Netscope_a_round(Y, A_rm):
    '''
    :param y: 观测的路径拥塞状态，列向量；如为矩阵，横纬度为时间
    :param y_l: 观测的路径丢包率，列向量；如为矩阵，横纬度为时间
    :param A_rm: routing matrix,矩阵，纵维度为路径，横维度为链路
    '''
    model = Model()
    model.setParam('OutputFlag', 0)
    w = 0.15
    
    Y = Y.reshape(-1, 1)

    cols = A_rm.shape[1]
    Y_rows = Y.shape[0]
    Y_cols = Y.shape[1]

    # 定义X_vars为决策变量，非负
    X_vars = model.addVars(cols, Y_cols, vtype=GRB.CONTINUOUS, lb=0, name="X")

    # 为了表示Y - RX和X的绝对值，引入额外变量
    abs_diff_vars = model.addVars(Y_rows, Y_cols, lb=0, name="AbsDiff")
    abs_X_vars = model.addVars(cols, Y_cols, lb=0, name="AbsX")

    # 添加约束以确保 abs_diff_vars 确实代表了差值的绝对值
    for i in range(Y_rows):
        for j in range(Y_cols):
            model.addConstr(abs_diff_vars[i, j] >= quicksum(A_rm[i,k] * X_vars[k,j] for k in range(cols)) - Y[i,j], name=f"abs_diff_pos_{i}_{j}")
            model.addConstr(abs_diff_vars[i, j] >= -(quicksum(A_rm[i,k] * X_vars[k,j] for k in range(cols)) - Y[i,j]), name=f"abs_diff_neg_{i}_{j}")

    # 添加约束以确保 abs_X_vars 确实代表了X的绝对值
    for k in range(cols):
        for j in range(Y_cols):
            model.addConstr(abs_X_vars[k, j] >= X_vars[k,j], name=f"abs_X_pos_{k}_{j}")
            model.addConstr(abs_X_vars[k, j] >= -X_vars[k,j], name=f"abs_X_neg_{k}_{j}")

    # 目标函数是最小化 abs_diff_vars 的和 加上 w 乘以 abs_X_vars 的和
    obj = quicksum(abs_diff_vars[i, j] for i in range(Y_rows) for j in range(Y_cols)) + w * quicksum(abs_X_vars[k, j] for k in range(cols) for j in range(Y_cols))

    model.setObjective(obj, GRB.MINIMIZE)

    # 求解模型
    model.optimize()

    # 提取解决方案
    X_solution = np.zeros((cols, Y_cols))
    for k in range(cols):
        for j in range(Y_cols):
            X_solution[k, j] = X_vars[k, j].X
    

    x_boolean = np.zeros((cols, Y_cols))
    for l in range(cols):
        if X_vars[l, 0].x > 0.02:
            x_boolean[l, 0] = 1

    return x_boolean, X_solution

def Netscope(Y_l, A_rm):
    '''
    :param Y: 观测的路径拥塞状态，列向量；如为矩阵，横纬度为时间
    :param Y_l: 观测的路径丢包率，列向量；如为矩阵，横纬度为时间
    :param A_rm: routing matrix,矩阵，纵维度为路径，横维度为链路
    '''
    times_n = Y_l.shape[1] 
    x_identified = np.zeros((A_rm.shape[1], times_n))
    x_loss_rate_identified = np.zeros((A_rm.shape[1], times_n))
    
    for t in range(times_n):
        # print(f'--------------第{t}轮-------------')
        y_l = Y_l[:, t]
        x, x_l = Netscope_a_round(y_l, A_rm)
        x_identified[:, t] = x.flatten()
        x_loss_rate_identified[:, t] = x_l.flatten()
        
    # print('x_identified', x_identified)
    # print('x_loss_rate_identified', x_loss_rate_identified)

    return x_identified, x_loss_rate_identified


def test():
    
    # 设置随机种子
    
    # np.random.seed(0)
    
    net = 网络基类()
    name = 'easy'
    net.配置拓扑(f"./topology_zoo/topology_zoo数据集/{name}.gml")
    sp = [get_source_nodes(name)[0][0]]
    print(sp)
    net.部署测量路径(源节点列表=sp)
    
    net.配置参数(异构链路先验拥塞概率 = 0.5)
    print(net.路由矩阵.shape)

    net.运行网络(运行的总时间 = 5000)
    观测矩阵_ = net.运行日志['路径状态']
    链路观测矩阵_ = net.运行日志['链路状态']
    丢包率矩阵 = net.运行日志['链路丢包率状态']
    路径丢包率观测矩阵 = net.运行日志['路径丢包率状态']

    Y = np.where(观测矩阵_, 0, 1)
    X = np.where(链路观测矩阵_, 0, 1)
    路由矩阵_ = net.路由矩阵
    True_X = np.where(链路观测矩阵_, 0, 1)
    A_rm = np.where(路由矩阵_, 1, 0)

    print('路由矩阵')
    print(A_rm)
    
    print('真实链路状态')
    print(True_X)
    
    print('路径布尔状态')
    print(Y)
    print('路径观测丢包率')
    print(路径丢包率观测矩阵)
    Netscope(路径丢包率观测矩阵, A_rm)


if __name__ == '__main__':
    test()