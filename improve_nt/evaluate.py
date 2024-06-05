from alg_clink_2007 import alg_clink_2007
from alg_clink_2007_pcs import alg_clink_2007_pcs


from alg_range_sum import Range_Tomo_sum
from alg_range_sum_pcs import Range_Tomo_sum_pcs

from alg_netscope import Netscope
from alg_netscope_pcs import Netscope_pcs

import json, os
from time import time, strftime, localtime

from Topology_zoo_Analog import *

def test(tp_name, avg_prob, index, noise_= 0, ternary_flag=False, print_flag=False):
    
    # 设置随机种子
    np.random.seed(index)
    net = 网络基类()
    name = tp_name
    net.配置拓扑(f"./topology_zoo/topology_zoo数据集/{name}.gml")
    sp = [get_source_nodes(name)[0][0]]
    # print(sp)
    net.部署测量路径(源节点列表=sp)
    
    net.配置参数(异构链路先验拥塞概率 = avg_prob)
            
    pri_cong_prob = np.array([i.先验拥塞概率 for i in net.链路集和])
    print(net.路由矩阵.shape)

    N = 3000
    net.运行网络(运行的总时间 = N)
    观测矩阵_ = net.运行日志['路径状态']
    链路观测矩阵_ = net.运行日志['链路状态']
    链路丢包率矩阵 = net.运行日志['链路丢包率状态']
    路径丢包率观测矩阵 = net.运行日志['路径丢包率状态']
    PCS = net.运行日志['路径拥塞链路数']

    PCS_noise = PCS.copy()
    print('PCS', PCS.shape)
    Y = np.where(观测矩阵_, 0, 1)
    X = np.where(链路观测矩阵_, 0, 1)
    路由矩阵_ = net.路由矩阵
    True_X = np.where(链路观测矩阵_, 0, 1)
    A_rm = np.where(路由矩阵_, 1, 0)

    noise = noise_
    # 路由矩阵每行 1 的个数
    paths_len = np.sum(A_rm, axis=1)
    # print(paths_len)
    for t in range(N):
        for p in range(A_rm.shape[0]):
            if random.random() < (noise * 2) and Y[p][t] == 1:
                # print('yes')
                pcs_noise = np.random.randint(1, paths_len[p])
                PCS_noise[p][t] = pcs_noise + PCS[p][t]
                if PCS_noise[p][t] > paths_len[p]:
                    PCS_noise[p][t] = paths_len[p]
    
    r_err = 0
    for t in range(N):
        for p in range(A_rm.shape[0]):
            r_err += (abs(PCS_noise[p][t] - PCS[p][t]) / paths_len[p]) ** 0.5

    print('PCS noise:', r_err / (N * A_rm.shape[0]))
    
    if ternary_flag:
        PCS_noise[PCS_noise >= 2] = 2
        PCS = PCS_noise.copy()

    time_start = time()

    c_x = alg_clink_2007(Y.copy(), A_rm, pri_cong_prob)
    c_x_pcs = alg_clink_2007_pcs(Y.copy(), A_rm, pri_cong_prob, PCS.copy())

    c_b_metrics = Evaluate_boolean_performance(True_X, c_x)
    c_b_metrics_pcs = Evaluate_boolean_performance(True_X, c_x_pcs)
    time_end_1 = time()
    if print_flag:
        print('clink:', c_b_metrics)
        print('clink_pcs', c_b_metrics_pcs)
        print('spending time(s):', time_end_1 - time_start)
    

    r_x, r_x_l = Range_Tomo_sum(Y.copy(), 路径丢包率观测矩阵.copy(), A_rm)
    # r_x_pcs, r_x_l_pcs = Range_Tomo_sum(Y.copy(), 路径丢包率观测矩阵.copy(), A_rm)
    r_x_pcs, r_x_l_pcs = Range_Tomo_sum_pcs(Y.copy(), 路径丢包率观测矩阵.copy(), A_rm, PCS.copy())
    range_b_metrics = Evaluate_boolean_performance(True_X, r_x)
    range_l_metrics = Evaluate_loss_performance(链路丢包率矩阵, r_x_l)
    range_b_metrics_pcs = Evaluate_boolean_performance(True_X, r_x_pcs)
    range_l_metrics_pcs = Evaluate_loss_performance(链路丢包率矩阵, r_x_l_pcs)
    time_end_2 = time()
    if print_flag:
        print('range_sum:', range_b_metrics)
        print('range_sum_pcs:', range_b_metrics_pcs)
        print('range_sum_loss:', range_l_metrics)
        print('range_sum_loss_pcs:', range_l_metrics_pcs)
        print('spending time(s):', time_end_2 - time_end_1)


    n_x, n_x_l = Netscope(路径丢包率观测矩阵.copy(), A_rm)
    n_x_pcs, n_x_l_pcs = Netscope_pcs(路径丢包率观测矩阵.copy(), A_rm, PCS.copy())

    nets_b_metrics = Evaluate_boolean_performance(True_X, n_x)
    nets_l_metrics = Evaluate_loss_performance(链路丢包率矩阵, n_x_l)

    nets_b_metrics_pcs = Evaluate_boolean_performance(True_X, n_x_pcs)
    nets_l_metrics_pcs = Evaluate_loss_performance(链路丢包率矩阵, n_x_l_pcs)
    time_end_3 = time()
    if print_flag:
        print('netscope:', nets_b_metrics)
        print('netscope_pcs:', nets_b_metrics_pcs)
        print('netscope_loss:', nets_l_metrics)
        print('netscope_loss_pcs:', nets_l_metrics_pcs)
        print('spending time(s):', time_end_3 - time_end_2)

    result = {}
    result['index'] = index
    result['clink'] = c_b_metrics
    result['clink_pcs'] = c_b_metrics_pcs
    result['range_sum'] = range_b_metrics
    result['range_sum_pcs'] = range_b_metrics_pcs
    result['range_sum_loss'] = range_l_metrics
    result['range_sum_loss_pcs'] = range_l_metrics_pcs
    result['netscope'] = nets_b_metrics
    result['netscope_pcs'] = nets_b_metrics_pcs
    result['netscope_loss'] = nets_l_metrics
    result['netscope_loss_pcs'] = nets_l_metrics_pcs
    result['probs'] = pri_cong_prob.tolist()

    # 保存json 文件
    file_path = f'./result/{tp_name}/new/'
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    time_strap = strftime("%Y-%m-%d-%H-%M-%S", localtime())
    result['time'] = time_strap
    with open(f'./result/{tp_name}/new/{avg_prob}_{index}_{ternary_flag}.json', 'w') as f:
        json.dump(result, f)

if __name__ == '__main__':
    test('Agis', 0.9, 2, 0, True, True)