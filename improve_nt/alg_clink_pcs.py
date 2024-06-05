'''
[INFOCOM'07] The Boolean Solution to the Congested IP Link Location Problem Theory and Practice
'''

import numpy as np
import copy


def alg_clink_pcs(y: np.ndarray, A_rm: np.ndarray, x_pc: np.ndarray, pcs: np.ndarray):
    """
    clink算法的调用接口
    :param y: 观测的路径拥塞状态，列向量；如为矩阵，横纬度为时间
    :param A_rm: routing matrix,矩阵，纵维度为路径，横维度为链路
    :param x_pc: ’probability of congestion' 列向量
    :return: x_identified 为返回的已识别的链路拥塞状态；列向量，如为矩阵，横维度为时间，纵维度为链路推测状态
    """
    # print("\rClink is inferring...", end="")
    if np.ndim(y) <= 1:  # 强制转换为列向量
        y = y.reshape((-1, 1))

    m, n = A_rm.shape
    _, num_time = y.shape

    x_identified = np.zeros((n, num_time))

    for i in range(num_time):
        # print(f'---------{i}---------')
        paths_state_obv = y[:, i]
        pcs_t = pcs[:, i]
        links_state_infered = clink_a_groub(paths_state_obv, A_rm, x_pc, pcs_t)
        x_identified[:, i] = links_state_infered

    return np.int8(x_identified)


def clink_a_groub(y, A_rm, x_pc, pcs):
    """
    clink 算法测试一组数据
    :param y: np.ndarray  观测路径的拥塞状态，普通向量
    :param A_rm: np。ndarray  路由矩阵
    :param x_pc: 链路拥塞概率，np.ndarray  普通向量
    :return:links_state_inferred: np.ndarray 普通
    """
    tree_vector = rm_to_vector(A_rm)
    num_links = len(tree_vector)
    links_congest_pro = list(x_pc.flatten())

    route_matrix = A_rm

    paths_state_obv = copy.deepcopy(y)  # 观测路径状态数组

    links_state_inferred = diagnose(paths_state_obv, route_matrix, num_links, links_congest_pro, pcs)

    return links_state_inferred


def diagnose(paths_state_obv, route_matrix: np.ndarray, num_links, links_cong_pro: list, pcs):
    """
    clink算法核心部分
    :param paths_state_obv: array(m,)  路径的观测状态
    :param route_matrix:  array(m,n)   路由矩阵
    :param num_links: int  链路数量
    :param links_cong_pro: list 链路的拥塞概率
    :return:
    """
    paths_cong_obv, paths_no_cong_obv = cal_cong_path_info(paths_state_obv)
    # print('链路的拥塞概率:', links_cong_pro)
    congested_path = (paths_cong_obv - 1).tolist()
    un_congested_path = (paths_no_cong_obv - 1).tolist()
    # print("congested_path",congested_path)
    # print("un_congested_path",un_congested_path)

    # 生成正常链路和不确定链路
    good_link, uncertain_link = get_link_state_class(un_congested_path, route_matrix, num_links)
    # print('位于不拥塞路径中的链路:', good_link)
    # print('不确定拥塞状态的链路:', uncertain_link)

    # 获取经过一条链路的所有路径domain
    domain_dict = {}
    for i in uncertain_link:
        domain_dict[i] = [j for j in get_paths(i + 1, route_matrix) if j in congested_path]
    # print("domain_dict")
    # print(domain_dict)

    links_state_inferred = np.zeros(num_links)
    links_cong_inferred = []
    # 计算所有的链路
    temp_state = [1e8 for _ in range(len(uncertain_link))]
    # print('temp_state:', temp_state)

    if not temp_state and len(congested_path):  # 如果存在无解的情形
        links_state_inferred = links_state_inferred + np.nan

    while temp_state and len(congested_path) > 0:
        # 找到最小的值对应的链路
        for index, i in enumerate(uncertain_link):
            # print(self._congestion_prob_links)
            # 方法1 公式(log((1-p)/p))|domain(x)|
            a = np.log((1 - links_cong_pro[i]) / (links_cong_pro[i]))
            b = len(domain_dict[i])
            if b == 0:
                temp_state[index] = 1e8
            else:
                temp_state[index] = a / b

        # print(temp_state)
        index = temp_state.index(min(temp_state))
        links_state_inferred[uncertain_link[index]] = 1
        links_cong_inferred.append(uncertain_link[index] + 1)
        # print("推断的链路", uncertain_link[index]+1)

        for item in domain_dict[uncertain_link[index]]:
            if item in congested_path:
                # print('congested_path', congested_path)
                # print('item:', item)
                congested_path.remove(item)

        domain_dict.pop(uncertain_link[index])
        uncertain_link.remove(uncertain_link[index])
        temp_state.remove(temp_state[index])

        links_state_inferred = np.int8(links_state_inferred)
        # print('link_state', links_state_inferred)

        path_state_now = route_matrix.dot(links_state_inferred)
        # print(route_matrix.shape, links_state_inferred.shape, path_state_now.shape)
        # print('path_state_now', path_state_now)
        # print('path_state_obv', paths_state_obv)

        diff = pcs - path_state_now
        # print('diff', diff)
        fitting_paths = np.where(diff == 0)[0]

        # print('fitting_paths', fitting_paths)

        f_path_links = []
        for p in fitting_paths:
            for l in range(num_links):
                if route_matrix[p][l] == 1:
                    f_path_links.append(l)
        
        for l in f_path_links:
            if l in uncertain_link:
                domain_dict.pop(l)
                temp_state.remove(temp_state[uncertain_link.index(l)])
                uncertain_link.remove(l)
                
                links_state_inferred[l] = 0


        for k, v in domain_dict.items():
            temp = []
            for i in v:
                if i in congested_path:
                    temp.append(i)
            domain_dict[k] = copy.deepcopy(temp)

        # print("domain_dict")
        # print(domain_dict)
        # print("uncertain_link",uncertain_link)
        # print("congest_path",congested_path)
    return links_state_inferred
    # print("真实的链路拥塞",self._links_congested)
    # print('推测的链路拥塞:', self.link_state_inferred)


def get_paths(link: int, route_matrix):
    """
    获取经过指定链路的所有路径。

    在路由矩阵中，第 0 列代表链路 1，第 1 列代表链路 2。依次类推。
    第 0 行代表路径 1，第 1 行代表路径 2。依次类推。
    :param link: 链路的编号
    :return:
    """
    assert link > 0
    paths, = np.where(route_matrix[:, link - 1] > 0)
    return paths.tolist()


def get_link_state_class(un_congested_path: list, route_matrix, num_links):
    """
    根据非拥塞路径，返回正常链路列表，和拥塞链路列表
    :param un_congested_path:list
    :return:good_link:list ,uncertain_link:list   存储链路下标
    """
    # 所有经过了不拥塞路径的链路
    good_link = []

    for i in un_congested_path:
        for index, item in enumerate(route_matrix[i]):
            if int(item) == 1 and index not in good_link:
                good_link.append(index)

    all_links = [i for i in range(num_links)]
    # 排除那些肯定不拥塞的链路
    uncertain_link = []
    for item in all_links:
        if item not in good_link:
            uncertain_link.append(item)
    return good_link, uncertain_link


def cal_cong_path_info(paths_state_obv):
    """
    根据路径的观测信息，计算拥塞路径和非拥塞路径
    :param paths_state_obv:
    :return:
    """
    paths_cong = []
    paths_no_cong = []
    for index in range(len(paths_state_obv)):
        if int(paths_state_obv[index]) == 1:
            # if int(self.path_states[index]) == 1:
            paths_cong.append(index + 1)
        else:
            paths_no_cong.append(index + 1)
    return np.array(paths_cong), np.array(paths_no_cong)


def rm_to_vector(A_rm: np.ndarray):
    """
    将路由矩阵转换为树向量
    :param A_rm:
    :return:
    """

    tree_vector = [0] * (A_rm.shape[1])

    for i in range(A_rm.shape[0]):
        path = A_rm[i]
        pre_node = 0
        for j in range(path.shape[0]):
            if path[j] == 1:
                tree_vector[j] = pre_node
                pre_node = j + 1

    return tree_vector


def test_clink_alg():
    # 测试一
    # tree_vector=[0, 1, 1, 1, 1, 1, 6, 6, 6, 6, 6, 6, 6, 9, 9, 12, 12, 15, 15]
    # net = Cong_net()
    # net.auto_init(tree_vector, "loss_model_1", 0.015, 0.1)
    # y=np.array([0 ,1 ,1 ,0 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1]).reshape((-1,1))
    # A_rm=net.route_matrix
    # x_pc=np.array([0.549, 0.715, 0.603, 0.545, 0.424, 0.646, 0.438, 0.892, 0.964, 0.383, 0.792, 0.529, 0.568, 0.926,
    #                   0.071, 0.087, 0.02, 0.833, 0.778])

    # 测试二
    # tree_vector = [0, 1, 1, 3, 3, 5, 5, 6, 6, 6]

    # y = np.array([1, 0, 1, 1, 1, 1]).reshape((-1, 1))
    # # y=np.array([[1,0,1,1,1,1],[1,1,1,1,1,1],[0,0,1,1,1,1]]).transpose()
    # A_rm = np.array([
    #     [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [1, 0, 1, 1, 0, 0, 0, 0, 0, 0],
    #     [1, 0, 1, 0, 1, 0, 1, 0, 0, 0],
    #     [1, 0, 1, 0, 1, 1, 0, 1, 0, 0],
    #     [1, 0, 1, 0, 1, 1, 0, 0, 1, 0],
    #     [1, 0, 1, 0, 1, 1, 0, 0, 0, 1]])

    # x_pc = np.array([0.1] * 10)

    # links_state_infered = alg_clink(y, A_rm, x_pc)
    # print(links_state_infered)

    # 测试三
    # y = np.array([[0, 0, 0],
    #               [0, 0, 1],
    #               [0, 1, 0],
    #               [0, 1, 1],
    #               [1, 0, 0],
    #               [1, 0, 1],
    #               [1, 1, 0],
    #               [1, 1, 1]], dtype=np.int8).T

    # A_rm = np.array([
    #     [1, 0, 1, 1, 0],
    #     [1, 0, 1, 0, 1],    
    #     [0, 1, 1, 0, 1]], dtype=np.int8)

    Y = np.array([[1, 1, 1]]).T
    PCS = np.array([[1, 1, 1]]).T

    A_rm = np.array([[1, 1, 0, 0],
                [1, 0, 1, 0],
                [1, 0, 0, 1]], dtype=np.int8)




    x_pc = np.array([0.05, 0.1, 0.35, 0.1])

    links_state_inferred = alg_clink_pcs(Y, A_rm, x_pc, PCS)
    print(links_state_inferred)


if __name__ == '__main__':
    test_clink_alg()





# '''
# [INFOCOM'07] The Boolean Solution to the Congested IP Link Location Problem Theory and Practice
# '''

# import numpy as np
# import copy
# import warnings

# def alg_clink(y: np.ndarray, A_rm: np.ndarray, x_pc: np.ndarray):
#     print("\rClink is inferring...", end="")
#     """
#     clink算法的调用接口
#     :param y: 观测的路径拥塞状态，列向量；如为矩阵，横纬度为时间
#     :param A_rm: routing matrix,矩阵，纵维度为路径，横维度为链路
#     :param x_pc: 'probability of congestion' 列向量
#     :return: x_identified 为返回的已识别的链路拥塞状态；列向量，如为矩阵，横维度为时间，纵维度为链路推测状态
#     """

#     if np.ndim(y) <= 1:  # 强制转换为列向量
#         y = y.reshape((-1, 1))

#     m, n = A_rm.shape
#     _, num_time = y.shape

#     x_identified = np.zeros((n, num_time))

#     for i in range(num_time):
#         paths_state_obv = y[:, i]
#         links_state_infered = clink_a_groub(paths_state_obv, A_rm, x_pc)
#         x_identified[:, i] = links_state_infered

#     print("\rclink inferred!       ", end="")
#     return np.int8(x_identified)


# def clink_a_groub(y, A_rm, x_pc):
#     """
#     clink 算法测试一组数据
#     :param y: np.ndarray  观测路径的拥塞状态，普通向量
#     :param A_rm: np。ndarray  路由矩阵
#     :param x_pc: 链路拥塞概率，np.ndarray  普通向量
#     :return:links_state_inferred: np.ndarray 普通
#     """
#     tree_vector = rm_to_vector(A_rm)
#     num_links = len(tree_vector)
#     links_congest_pro = list(x_pc.flatten())

#     route_matrix = A_rm

#     paths_state_obv = copy.deepcopy(y)  # 观测路径状态数组

#     links_state_inferred = diagnose(paths_state_obv, route_matrix, num_links, links_congest_pro)
#     return links_state_inferred


# def diagnose(paths_state_obv, route_matrix: np.ndarray, num_links, links_cong_pro: list):
#     """
#     clink算法核心部分
#     :param paths_state_obv: array(m,)  路径的观测状态
#     :param route_matrix:  array(m,n)   路由矩阵
#     :param num_links: int  链路数量
#     :param links_cong_pro: list 链路的拥塞概率
#     :return:
#     """
#     paths_cong_obv, paths_no_cong_obv = cal_cong_path_info(paths_state_obv)
#     # print('链路的拥塞概率:', links_cong_pro)
#     congested_path = (paths_cong_obv - 1).tolist()
#     un_congested_path = (paths_no_cong_obv - 1).tolist()
#     # print("congested_path",congested_path)
#     # print("un_congested_path",un_congested_path)

#     # 生成正常链路和不确定链路
#     good_link, uncertain_link = get_link_state_class(un_congested_path, route_matrix, num_links)
#     # print('位于不拥塞路径中的链路:', good_link)
#     # print('不确定拥塞状态的链路:', uncertain_link)

#     # 获取经过一条链路的所有路径domain
#     domain_dict = {}
#     for i in uncertain_link:
#         domain_dict[i] = [j for j in get_paths(i + 1, route_matrix) if j in congested_path]
#     # print("domain_dict")
#     # print(domain_dict)

#     links_state_inferred = np.zeros(num_links)
#     links_cong_inferred = []
#     # 计算所有的链路
#     temp_state = [1e8 for _ in range(len(uncertain_link))]
#     # print('temp_state:', temp_state)

#     if not temp_state and len(congested_path):  # 如果存在无解的情形
#         links_state_inferred = links_state_inferred + np.nan
#     warnings.filterwarnings('ignore')
#     while temp_state and len(congested_path) > 0:
#         # 找到最小的值对应的链路
#         for index, i in enumerate(uncertain_link):
#             # print(self._congestion_prob_links)
#             # 方法1 公式(log((1-p)/p))|domain(x)|
#             try:
#                 a = np.log((1 - links_cong_pro[i]) / (links_cong_pro[i]))
#             except (ValueError, ZeroDivisionError, RuntimeWarning):
#                 a = np.nan  # 替换无效值为 NaN
#             b = len(domain_dict[i])
#             if b == 0:
#                 temp_state[index] = 1e8
#             else:
#                 temp_state[index] = a / b

#             # 方法2 公式log((1-p)/p/|domain(x)|)
#             # b=len(domain_dict[i])
#             # if b==0:
#             #     temp_state[index]=1e8
#             # else:
#             #     a=np.log((1 - self._congestion_prob_links[i]) / (self._congestion_prob_links[i])/b)
#             #     temp_state[index]=a

#         # print(temp_state)
#         index = temp_state.index(min(temp_state))
#         links_state_inferred[uncertain_link[index]] = 1
#         links_cong_inferred.append(uncertain_link[index] + 1)
#         # print("推断的链路",uncertain_link[index]+1)
#         for item in domain_dict[uncertain_link[index]]:
#             if item in congested_path:
#                 # print('congested_path', congested_path)
#                 # print('item:', item)
#                 congested_path.remove(item)
#         domain_dict.pop(uncertain_link[index])
#         uncertain_link.remove(uncertain_link[index])
#         temp_state.remove(temp_state[index])

#         for k, v in domain_dict.items():
#             temp = []
#             for i in v:
#                 if i in congested_path:
#                     temp.append(i)
#             domain_dict[k] = copy.deepcopy(temp)

#         # print("domain_dict")
#         # print(domain_dict)
#         # print("uncertain_link",uncertain_link)
#         # print("congest_path",congested_path)
#     print("\r      Clink inferred !", end="")
#     return links_state_inferred
#     # print("真实的链路拥塞",self._links_congested)
#     # print('推测的链路拥塞:', self.link_state_inferred)


# def get_paths(link: int, route_matrix):
#     """
#     获取经过指定链路的所有路径。

#     在路由矩阵中，第 0 列代表链路 1，第 1 列代表链路 2。依次类推。
#     第 0 行代表路径 1，第 1 行代表路径 2。依次类推。
#     :param link: 链路的编号
#     :return:
#     """
#     assert link > 0
#     paths, = np.where(route_matrix[:, link - 1] > 0)
#     return paths.tolist()


# def get_link_state_class(un_congested_path: list, route_matrix, num_links):
#     """
#     根据非拥塞路径，返回正常链路列表，和拥塞链路列表
#     :param un_congested_path:list
#     :return:good_link:list ,uncertain_link:list   存储链路下标
#     """
#     # 所有经过了不拥塞路径的链路
#     good_link = []

#     for i in un_congested_path:
#         for index, item in enumerate(route_matrix[i]):
#             if int(item) == 1 and index not in good_link:
#                 good_link.append(index)

#     all_links = [i for i in range(num_links)]
#     # 排除那些肯定不拥塞的链路
#     uncertain_link = []
#     for item in all_links:
#         if item not in good_link:
#             uncertain_link.append(item)
#     return good_link, uncertain_link


# def cal_cong_path_info(paths_state_obv):
#     """
#     根据路径的观测信息，计算拥塞路径和非拥塞路径
#     :param paths_state_obv:
#     :return:
#     """
#     paths_cong = []
#     paths_no_cong = []
#     for index in range(len(paths_state_obv)):
#         if int(paths_state_obv[index]) == 1:
#             # if int(self.path_states[index]) == 1:
#             paths_cong.append(index + 1)
#         else:
#             paths_no_cong.append(index + 1)
#     return np.array(paths_cong), np.array(paths_no_cong)


# def rm_to_vector(A_rm: np.ndarray):
#     """
#     将路由矩阵转换为树向量
#     :param A_rm:
#     :return:
#     """

#     tree_vector = [0] * (A_rm.shape[1])

#     for i in range(A_rm.shape[0]):
#         path = A_rm[i]
#         pre_node = 0
#         for j in range(path.shape[0]):
#             if path[j] == 1:
#                 tree_vector[j] = pre_node
#                 pre_node = j + 1

#     return tree_vector


# def test_clink_alg():
#     # 测试一
#     # tree_vector=[0, 1, 1, 1, 1, 1, 6, 6, 6, 6, 6, 6, 6, 9, 9, 12, 12, 15, 15]
#     # net = Cong_net()
#     # net.auto_init(tree_vector, "loss_model_1", 0.015, 0.1)
#     # y=np.array([0 ,1 ,1 ,0 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1]).reshape((-1,1))
#     # A_rm=net.route_matrix
#     # x_pc=np.array([0.549, 0.715, 0.603, 0.545, 0.424, 0.646, 0.438, 0.892, 0.964, 0.383, 0.792, 0.529, 0.568, 0.926,
#     #                   0.071, 0.087, 0.02, 0.833, 0.778])

#     # 测试二
#     # tree_vector = [0, 1, 1, 3, 3, 5, 5, 6, 6, 6]

#     # y = np.array([1, 0, 1, 1, 1, 1]).reshape((-1, 1))
#     # # y=np.array([[1,0,1,1,1,1],[1,1,1,1,1,1],[0,0,1,1,1,1]]).transpose()
#     # A_rm = np.array([
#     #     [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#     #     [1, 0, 1, 1, 0, 0, 0, 0, 0, 0],
#     #     [1, 0, 1, 0, 1, 0, 1, 0, 0, 0],
#     #     [1, 0, 1, 0, 1, 1, 0, 1, 0, 0],
#     #     [1, 0, 1, 0, 1, 1, 0, 0, 1, 0],
#     #     [1, 0, 1, 0, 1, 1, 0, 0, 0, 1]])

#     # x_pc = np.array([0.1] * 10)

#     # links_state_infered = alg_clink(y, A_rm, x_pc)
#     # print(links_state_infered)

#     # 测试三
#     y = np.array([[0, 0, 0],
#                 #   [0, 0, 1],
#                 #   [0, 1, 0],
#                 #   [0, 1, 1],
#                 #   [1, 0, 0],
#                 #   [1, 0, 1],
#                 #   [1, 1, 0],
#                 # [1, 1, 1]
#                   ], dtype=np.int8).T

#     A_rm = np.array([
#         [1, 0, 1, 1, 0],
#         [1, 0, 1, 0, 1],
#         [0, 1, 1, 0, 1]], dtype=np.int8)
    
#     # A_rm = np.array([
#     #     [1,0,1,1,0],
#     #     [1,0,1,0,1],
#     #     [0,1,1,0,1]
#     #     ])
#     # print(y)
#     x_pc = np.array([0.1] * 5)
#     # print(x_pc)
#     links_state_inferred = alg_clink(y, A_rm, x_pc)
#     print(links_state_inferred)


# if __name__ == '__main__':
#     test_clink_alg()
