# -*- coding: utf-8 -*-
# @Author  : 杜承泽
# @Email   : Monickar@foxmail.com

from Topology_zoo_Analog import *


def range_a_round(y, y_l, A_rm):
    '''
    :param y: 观测的路径拥塞状态，列向量；如为矩阵，横纬度为时间
    :param y_l: 观测的路径丢包率，列向量；如为矩阵，横纬度为时间
    :param A_rm: routing matrix,矩阵，纵维度为路径，横维度为链路
    '''

    # print('路径布尔状态', y)
    # print('路径观测丢包率', y_l)

    alpha = 0.3
    x_loss_rate = {l:0 for l in range(A_rm.shape[1])} # x_c -> 待推测的链路丢包率
    x_bool_status = {l:0 for l in range(A_rm.shape[1])} # x_c -> 待推测的链路状态

    R = [] # R -> 未修正的路径集和
    G_P = [] # G_P -> 未拥塞的路径
    for i in range(len(y)):
        if y[i] == 1:
            R.append(i)
        else:
            G_P.append(i)
    # print('R', R)
    # print('G_P', G_P)

    U = [] # U -> 未标记的链路集和
    for p in R:
        for l in range(A_rm.shape[1]):
            if A_rm[p][l] == 1:
                U.append(l)
    U = list(set(U))

    for p in G_P:
        for l in range(A_rm.shape[1]):
            if A_rm[p][l] == 1 and l in U:
                U.remove(l)

    # print('U', U)

    while len(U) > 0 and len(R) > 0: # 当 U集为空 且 R集为空
        # 从 R 中选出丢包率最低的路径
        min_loss = 1
        p_b = -1
        for p in R:
            if y_l[p] < min_loss:
                min_loss = y_l[p]
                p_b = p

        # print('p_b', p_b)

        # 从 R 中选出 丢包率与 p_b 相似的路径
        Omega = []
        for p in R:
            if abs(y_l[p] - y_l[p_b]) / min(y_l[p], y_l[p_b]) < alpha:
                Omega.append(p)
        
        # print('Omega', Omega)

        # 计算每条链路上经过故障路径的个数
        Num = {l: 0 for l in U}
        for l in U:
            n = 0
            for p in R:
                if A_rm[p][l] == 1:
                    n += 1
            Num[l] = n

        # 计算每条链路对应的 score
        scores = {l: 0 for l in U}
        for l in U:
            # 链路 l 经过 Omega 中的路径的个数
            n = 0
            for p in Omega:
                if A_rm[p][l] == 1:
                    n += 1
            scores[l] = n 

        # 选出有最大分数的链路集和
        max_score = 0
        L_m = []
        for l in U:
            if scores[l] > max_score:
                max_score = scores[l]
                L_m = [l]
            elif scores[l] == max_score:
                L_m.append(l)

        # 在 L_m 中选出 Num中值最大的链路 l_m
        max_num = 0
        l_m = -1
        for l in L_m:
            if Num[l] > max_num:
                max_num = Num[l]
                l_m = l

        # print('l_m', l_m)
        # 计算 Omega 中经过 l_m 的路径集和 的平均丢包率
        sum_loss = 0
        count = 0
        for p in Omega:
            if A_rm[p][l_m] == 1:
                sum_loss += y_l[p]
                count += 1
        r_bar = sum_loss / count

        # 计算 link l_m 的 范围 [r_bar *(1 / (1 + alpha)), r_bar * (1 + alpha)]
        r_min = r_bar * (1 / (1 + alpha))
        r_max = r_bar * (1 + alpha)

        # print('l_m', l_m, 'r_bar', r_bar, 'r_min', r_min, 'r_max', r_max)

        # 从 R 中移除丢包率在 r_min 和 r_max 之间的路径
        R_new = []
        for p in R:
            if A_rm[p][l_m] == 1:
                if r_min <= y_l[p] <= r_max:
                    R_new.append(p)
        R = [p for p in R if p not in R_new]

        # 更新其他包含l_m路径的丢包率 p_loss = max(0, p_loss - r_bar)
        for p in R:
            if A_rm[p][l_m] == 1:
                y_l[p] = max(0, y_l[p] - r_bar)
        
        # 从 U 中移除 l_m
        U.remove(l_m)

        x_loss_rate[l_m] = r_bar

        # print('update R', R, 'update U', U)

    # 若链路的丢包率大于 0.005，则认为链路故障
    for l in range(A_rm.shape[1]):
        if x_loss_rate[l] > 0.005:
            x_bool_status[l] = 1

    # 创建链路状态矩阵
    x = np.zeros((A_rm.shape[1], 1))
    x_l = np.zeros((A_rm.shape[1], 1))
    for l in range(A_rm.shape[1]):
        x[l] = x_bool_status[l]
        x_l[l] = x_loss_rate[l]
        
    # print('x', x)
    # print('x_l', x_l)
    # print('shape of x', x.shape)
    return x, x_l
    

def Range_Tomo_sum(Y, Y_l, A_rm):
    '''
    :param Y: 观测的路径拥塞状态，列向量；如为矩阵，横纬度为时间
    :param Y_l: 观测的路径丢包率，列向量；如为矩阵，横纬度为时间
    :param A_rm: routing matrix,矩阵，纵维度为路径，横维度为链路
    '''
    times_n = Y.shape[1] 
    x_identified = np.zeros((A_rm.shape[1], times_n))
    x_loss_rate_identified = np.zeros((A_rm.shape[1], times_n))
    
    for t in range(times_n):
        # print(f'--------------第{t}轮-------------')
        y = Y[:, t]
        y_l = Y_l[:, t]
        x, x_l = range_a_round(y, y_l, A_rm)
        x_identified[:, t] = x.flatten()
        x_loss_rate_identified[:, t] = x_l.flatten()
        
    # print('x_identified', x_identified)
    # print('x_loss_rate_identified', x_loss_rate_identified)

    return x_identified, x_loss_rate_identified


def test():
    
    # 设置随机种子
    
    np.random.seed(2)
    
    net = 网络基类()
    name = 'easy'
    net.配置拓扑(f"./topology_zoo/topology_zoo数据集/{name}.gml")
    sp = [get_source_nodes(name)[0][0]]
    print(sp)
    net.部署测量路径(源节点列表=sp)
    
    net.配置参数(异构链路先验拥塞概率 = 0.8)
    print(net.路由矩阵.shape)

    net.运行网络(运行的总时间 = 5)
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
    x, x_l = Range_Tomo_sum(Y, 路径丢包率观测矩阵, A_rm)

    print('x', x)


if __name__ == '__main__':
    test()