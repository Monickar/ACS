import itertools, statistics
import sys, random, json
from time import time
import types, re
from _ctypes import PyObj_FromPtr 

import warnings
from copy import deepcopy
from functools import reduce
from itertools import product
from scipy import stats

import matplotlib.pylab as plt
import networkx as nx
import numpy as np
from ordered_set import OrderedSet
from tqdm import trange



# noinspection NonAsciiCharacters
class 链路基类:
    def __init__(self, 名称: tuple):
        self.名称: tuple = 名称  # 元组类型的名称
        self.状态: bool = True  # 默认处于正常状态
        self.当前时刻: int = 0  # 记录当前运行的时刻
        self.先验拥塞概率: float = 0.0  # 默认不发生拥塞
        self.丢包率: float = 0.0  # 默认不发生丢包
        self.怪异: bool = False  # 默认为同构链路
        

    def __str__(self):
        return f'链路基类对象 {self.名称}'

    def __repr__(self):
        return f'链路基类对象 {self.名称}'

    def 生成状态(self, 运行时刻: int = 0):
        self.当前时刻 = deepcopy(运行时刻)  # 更新当前时刻
        self.状态 = False if np.random.rand() <= self.先验拥塞概率 else True
        return deepcopy(self.状态)

    def 生成丢包率(self, 运行时刻: int = 0):
        self.当前时刻 = deepcopy(运行时刻)  # 更新当前时刻

        丢包率 = np.random.uniform(0.03, 0.05)
        if self.怪异 == False:
            self.丢包率 = 丢包率 * (1 + self.先验拥塞概率) # 丢包率与先验拥塞概率有关
        else:
            self.丢包率 = 丢包率 * (1 + self.先验拥塞概率) * 2

        if self.状态 == True:
            self.丢包率 = 0.0

        self.丢包率 = np.round(self.丢包率, 3)
        return deepcopy(self.丢包率)

    def 攻击记录(self, 运行时刻: int = 0):
        pass

# noinspection NonAsciiCharacters
class 路径基类(链路基类):
    def __init__(self, 名称: tuple):
        super().__init__(名称)

        self.路由: list = []  # 集和元素为 链路基类 实例
        self.和状态: int = 0  # 0 为好；=1 为单条， > 2 为有多条坏链路
        self.拥塞链路数: int = 0  # 记录拥塞链路数

    def __str__(self):
        return f'路径基类对象 {self.名称}'

    def __repr__(self):
        return f'路径基类对象 {self.名称}'

    def 计算先验拥塞概率(self):
        self.先验拥塞概率 = 1.0 - reduce(lambda i, j: i * j, [1.0 - link.先验拥塞概率 for link in self.路由])
        return deepcopy(self.先验拥塞概率)

    def 生成状态(self, 运行时刻: int = 0):
        self.当前时刻 = deepcopy(运行时刻)  # 更新当前时刻
        # print("路径基类 生成状态")
        self.状态 = reduce(lambda i, j: i and j, [link.状态 for link in self.路由])  # True 为好；False 为坏
        return deepcopy(self.状态)

    def 生成和状态(self, 运行时刻: int = 0):
        self.当前时刻 = deepcopy(运行时刻)
        self.和状态 = len(self.路由) - sum(link.状态 for link in self.路由)
        if self.和状态 > 1:
            self.和状态 = 2
        return deepcopy(self.和状态)
        
    def 生成探测路径丢包率(self, 运行时刻: int = 0):
        self.当前时刻 = deepcopy(运行时刻)
        self.丢包率 = sum(link.丢包率 for link in self.路由)

        error_ratio = np.random.uniform(0.95, 1.05)
        self.丢包率 = np.round(self.丢包率 * error_ratio, 3)

        return deepcopy(self.丢包率)
    
    def 生成链路拥塞数(self, 运行时刻: int = 0):
        self.当前时刻 = deepcopy(运行时刻)
        self.拥塞链路数 = sum([1 for link in self.路由 if link.状态 == False])
        return deepcopy(self.拥塞链路数)
        
# noinspection NonAsciiCharacters
class 网络基类:
    def __init__(self):
        self.图文件名 = ''
        self.图: nx.DiGraph = nx.DiGraph([])

        self.名称: str = ''

        self.节点字典: dict = {
            '所有节点集和': OrderedSet([]),
            '边缘节点集和': OrderedSet([]),
            '内部节点集和': OrderedSet([])
        }

        self.链路集和: OrderedSet = OrderedSet([])
        self.路径集和: OrderedSet = OrderedSet([])

        self.路由矩阵: np.ndarray = np.array([])

        self.运行日志: dict = {
            '链路状态': np.empty_like([]),
            '路径状态': np.empty_like([]),
            '路径和状态': np.empty_like([]),
            '运行总时间': 0
        }

    def __str__(self):
        return f'网络基类对象 {self.名称}'
    
    def clear_(self):
        self.图文件名 = ''
        self.图: nx.DiGraph = nx.DiGraph([])

        self.名称: str = ''

        self.节点字典: dict = {
            '所有节点集和': OrderedSet([]),
            '边缘节点集和': OrderedSet([]),
            '内部节点集和': OrderedSet([])
        }

        self.链路集和: OrderedSet = OrderedSet([])
        self.路径集和: OrderedSet = OrderedSet([])

        self.路由矩阵: np.ndarray = np.array([])

        self.运行日志: dict = {
            '链路状态': np.empty_like([]),
            '路径状态': np.empty_like([]),
            '运行总时间': 0
        }


    def 配置拓扑(self, graph_gml: str):
        # 通过 topology zoo 提供的 gml 网络文件格式读取网络拓扑信息
        self.图文件名 = graph_gml + '.gml' if graph_gml[-4:] != '.gml' else graph_gml
        self.图 = nx.read_gml(self.图文件名)  # 最开始按照无向图来读取，最后再转成有向图

        # 读取拓扑名称
        try:
            self.名称 = self.图.graph['label'] if 'label' in self.图.graph.keys() else self.图.graph['Network']
        except KeyError:
            self.名称 = self.图文件名[:-4]

        # for node, degree in dict(self.图.degree()).items():
        #     print("node",node,"degree",degree)


        # 移除度为 2 的点：会造成该点相邻的两条链路，不具有"端到端"可区分性
        link_src_list = [link[0] for link in self.图.edges.keys()]
        link_dst_list = [link[1] for link in self.图.edges.keys()]
        for node, degree in dict(self.图.degree()).items():
            # print("node",node,"degree",degree)
            if not self.图._node[node]['Internal']:  # 移除外部的虚拟节点
                self.图.remove_node(node)
                # print(node)

            elif degree == 2 and node in link_src_list and node in link_dst_list:
                self.图.add_edge(*self.图.degree._nodes[node])
                #print("node",node,"degree",degree)
                self.图.remove_node(node)

        # 判断是否为连通图
        if not nx.is_connected(self.图):
            warnings.warn("注意: 当前裁剪得到的图属于一个非连通图！", category=Warning)

        # 获取节点字典
        self.节点字典['所有节点集和'] = OrderedSet([node for node in self.图.nodes])
        self.节点字典['边缘节点集和'] = OrderedSet([node for node, degree in self.图.degree if degree < 2])
        self.节点字典['内部节点集和'] = self.节点字典['所有节点集和'] - self.节点字典['边缘节点集和']

        # 转为有向图
        self.图 = self.图.to_directed()

        self.图.paths = []  # 向图实例中，新添加一个 paths 的列表属性

    def 部署测量路径(self, 源节点列表: list = [], 目的节点列表: list = []):
        """
        - 根据源目的节点列表，选定测量路径，以执行端到端测量
        - 输出测量覆盖的链路集和
        - 输出路由矩阵
        """

        if not OrderedSet(源节点列表).issubset(self.节点字典['边缘节点集和']):
            try:
                源节点列表 = [self.节点字典['所有节点集和'][i] for i in 源节点列表]
            except:
                warnings.warn("注意: 请检查所指定的源节点列表！", category=Warning)
                return
        else:
            warnings.warn("注意: 请检查所指定的源节点列表！", category=Warning)
            return

        if 目的节点列表 == []:
            目的节点列表 = list(self.节点字典['边缘节点集和'] - set(源节点列表))
        else:
            目的节点列表 = [net.节点字典['所有节点集和'][i] for i in 目的节点列表]


        path_set, self.图.paths, link_ref = dict(nx.all_pairs_shortest_path(self.图)), [], OrderedSet([])

        self.链路集和, self.路径集和 = OrderedSet([]), OrderedSet([])
        tmp_graph=nx.DiGraph()
        node_dict={}
        for src, dst in product(源节点列表, 目的节点列表):
            self.图.paths += [(src, dst)]
            self.路径集和.append(路径基类((src, dst)))

            route = path_set[src][dst]
            #print("route", route)
            for link in [(i, j) for i, j in zip(route[:-1], route[1:])]:
                node_dict[link[0]]=node_dict.get(link[0],0)+1
                node_dict[link[1]]=node_dict.get(link[1],0)+1
                if link in link_ref:
                    no = link_ref.index(link)
                    self.路径集和[-1].路由.append(self.链路集和[no])
                elif link[::-1] in link_ref:
                    no = link_ref.index(link[::-1])
                    self.路径集和[-1].路由.append(self.链路集和[no])
                else:
                    tmp_graph.add_edge(link[0],link[1])
                    link_ref.append(link)
                    #print("link_ref", link_ref)
                    self.链路集和.append(链路基类(link))
                    #print("链路集和", self.链路集和)
                    self.路径集和[-1].路由.append(self.链路集和[-1])

           

        # 获取路径矩阵
        self.路由矩阵 = np.zeros((len(self.路径集和), len(self.链路集和)), dtype=bool)
        for i, path in enumerate(self.路径集和):
            for j, link in enumerate(self.链路集和):
                self.路由矩阵[i, j] = True if link in path.路由 else False

        index = 0  # 去掉"不具有NT可识别性"的链路，即对于承载了相同路径集和的所有链路而言，仅保留其中一条
        while index < self.路由矩阵.shape[1] - 1:
            to_remove = []
            for i in range(index + 1, self.路由矩阵.shape[1]):
                if all(self.路由矩阵[:, index] == self.路由矩阵[:, i]):
                    to_remove.append(i)
            new_des=None
            new_source=None
            for i in to_remove:
                if tmp_graph.degree(self.链路集和[i].名称[1])!=2:
                    new_des=self.链路集和[i].名称[1]
                if tmp_graph.degree(self.链路集和[i].名称[0])>2 :
                    new_source=self.链路集和[i].名称[0]

            index_old=None
            if to_remove:
                index_old=deepcopy(self.链路集和[index])

            if not new_des:
                new_des=self.链路集和[index].名称[1]
            if not new_source:
                new_source=self.链路集和[index].名称[0]
            if new_des!=new_source:
                self.链路集和[index].名称=(new_source,new_des)

            if to_remove:
                for i in range(self.路由矩阵.shape[0]):
                    if self.路由矩阵[i,index]==True:
                        #self.路径集和[i].路由.remove(index_old)
                        #self.路径集和[i].路由.append(self.链路集和[index])
                        for j in range(len(to_remove)):
                            self.路径集和[i].路由.remove(self.链路集和[to_remove[j]])
            
            for i in range(len(to_remove)):
                self.图.remove_edge(self.链路集和[to_remove[i]].名称[1],self.链路集和[to_remove[i]].名称[0])
                self.图.remove_edge(self.链路集和[to_remove[i]].名称[0],self.链路集和[to_remove[i]].名称[1])

            if to_remove:
                self.图.remove_edge(index_old.名称[0],index_old.名称[1])
                self.图.remove_edge(index_old.名称[1],index_old.名称[0])
                self.图.add_edge(self.链路集和[index].名称[0],self.链路集和[index].名称[1])
                self.图.add_edge(self.链路集和[index].名称[1],self.链路集和[index].名称[0])

            #删除所有的度为0的点
            for node, degree in dict(self.图.degree()).items():
                if degree==0:
                    self.图.remove_node(node)
                    # print("删除节点:",node)
            
            self.路由矩阵 = np.delete(self.路由矩阵, to_remove, 1)
            for j in to_remove[::-1]:  # 通过倒序的方式来进行链路删除
                # print("删除链路:",self.链路集和[j].名称)
                self.链路集和.pop(j)
                # print(j)

            index += 1


    def 导出拓扑(self, img_name: str = None, with_path=False, with_name=False, with_geo=True, **kwargs):
        """
        - 默认只画拓扑，导出为 PNG 格式；若指定画出路径，则将所有的"端到端"路径全部高亮画出，并导出为 GIF 格式
        - 默认用节点对应的数字编号而非名称来标示每个节点
        - 默认使用节点的经纬度来画网络拓扑
        """

        with_attention = kwargs.get('with_attention', None)
        attention_link = kwargs.get('attention_link', [])
        attention_link_label = kwargs.get('attention_link_label', [])

        img_name = deepcopy(self.名称) if img_name is None else img_name

        pos = {}
        if with_geo:
            for node, attr in self.图._node.items():
                pos[node] = np.array([attr['Longitude'], attr['Latitude']])
        else:
            pos = nx.kamada_kawai_layout(self.图)

        # print("attention_link", attention_link)
        # print("attention_label", attention_link_label)
        edge_labels = {}


        # print(edge_labels)

        if with_path and self.图.paths != []:  # 画网络拓扑，并高亮测量路径
            import os, shutil, imageio
            
            if os.path.exists('./img'):
                shutil.rmtree('./img')
            os.makedirs('./img')
            nodes = list(self.图.nodes)
            edges = list(self.图.edges)

            frames = []

            for i, path in enumerate(self.图.paths):
                plt.figure()
                plt.title(f'{self.名称}\nPATH$_{{{i}}}$: {path[0]} --> {path[1]}')  # 使用LaTeX语句实现下标，注意有3层大括号
                nx.draw(self.图, pos,
                        with_labels=True,
                        node_color='white',
                        node_size=100,
                        edge_color='white',
                        labels=None if with_name else {node: idx for idx, node in enumerate(self.图.nodes.keys())})
                
                nx.draw_networkx_nodes(self.图, pos, nodelist=[path[0]], node_color='gold')
                nx.draw_networkx_nodes(self.图, pos, nodelist=[path[1]], node_color='grey', node_shape='s')

                path_index = self.图.paths.index(path)
                route = [link.名称 for link in self.路径集和[path_index].路由]

                nx.draw_networkx_edges(self.图, pos, edgelist=route, edge_color='green', width=2)

                if with_attention:
                    print(attention_link)
                    nx.draw_networkx_edges(self.图, pos, edgelist=attention_link, edge_color='blue', label='12',width=2, alpha=0.4)
                    # for link in attention_link:
                    #     if link in self.图.edges:
                    #         print(link, type(link))
                    #         edge_labels[link] = attention_link_label[link]
                    # nx.draw_networkx_edge_labels(self.图, pos, edge_labels=edge_labels, font_color='blue', font_size=8)
                

                plt.axis('tight')
                plt.savefig(f'./img/img_{i}.png')
                plt.close('all')

                image = imageio.v2.imread(f'./img/img_{i}.png')
                frames.append(image)



            imageio.mimsave('./data/' + img_name + '.gif', frames, duration=500)  # 间隔单位: ms
            shutil.rmtree('./img')

        else:
            plt.figure()
            plt.title(f'{self.名称}')

            nx.draw(self.图, pos,
                    with_labels=True,
                    node_color='white',
                    node_size=100,
                    edge_color='black',
                    arrows=False,  # 按无向边来做图
                    labels=None if with_name else {node: idx for idx, node in enumerate(self.图.nodes.keys())})

            plt.axis('tight')
            print("image_name", img_name)
            plt.savefig('./' + img_name + '.png')
            plt.close('all')

    def 配置参数(self, **kwargs):
        """
        可扩展，以支持配置更多的参数类别
        """
        if '异构链路先验拥塞概率' in kwargs.keys():
            if len([kwargs['异构链路先验拥塞概率']]) >= 1:
                given_c_prob = deepcopy(kwargs['异构链路先验拥塞概率'])
                prop = 0
                # 如果 kwargs['cov'] 存在
                if 'cov' in kwargs.keys():
                    prop = kwargs['cov']
                
                kwargs['异构链路先验拥塞概率'] = []
                for _ in self.链路集和:
                    # c_prob = np.float16(np.random.normal(loc=given_c_prob, scale=0.05))
                    # c_prob =  np.random.exponential(given_c_prob)
                    c_prob = np.random.uniform(0, given_c_prob)  
                    kwargs['异构链路先验拥塞概率'].append(c_prob if  0 < c_prob < 1
                                                       else 0 )

            for link, c_prob in zip(self.链路集和, kwargs['异构链路先验拥塞概率']):
                link.先验拥塞概率 = deepcopy(c_prob)

            for path in self.路径集和:
                path.计算先验拥塞概率()

        elif '同构链路先验拥塞概率' in kwargs.keys():
            if len([kwargs['同构链路先验拥塞概率']]) != 1:
                warnings.warn("注意: 当前只能指定同一先验拥塞概率值！", category=Warning)
                sys.exit(0)

            c_prob = kwargs['同构链路先验拥塞概率'] if 0.0 <= kwargs['同构链路先验拥塞概率'] <= 1.0 \
                else (1.0 if kwargs['同构链路先验拥塞概率'] > 1.0 else 0.0)
            for link in self.链路集和:
                link.先验拥塞概率 = deepcopy(c_prob)

            for path in self.路径集和:
                path.计算先验拥塞概率()

    def 运行网络(self, 运行的总时间: int = 1):
        self.运行日志['运行总时间'] = 运行的总时间

        self.运行日志['链路状态'] = np.zeros((len(self.链路集和), 运行的总时间), dtype=bool)
        self.运行日志['链路丢包率状态'] = np.zeros((len(self.链路集和), 运行的总时间), dtype=float)
        self.运行日志['路径状态'] = np.zeros((len(self.路径集和), 运行的总时间), dtype=bool)
        self.运行日志['路径和状态'] = np.zeros((len(self.路径集和), 运行的总时间), dtype=int)
        self.运行日志['路径丢包率状态'] = np.zeros((len(self.路径集和), 运行的总时间), dtype=float)
        self.运行日志['路径拥塞链路数'] = np.zeros((len(self.路径集和), 运行的总时间), dtype=int)

        for t in range(运行的总时间):# , desc=f'网络-{self.名称}', leave=True):
            self.运行日志['链路状态'][:, t] = np.array([link.生成状态(运行时刻=t) for link in self.链路集和])
            self.运行日志['链路丢包率状态'][:, t] = np.array([link.生成丢包率(运行时刻=t) for link in self.链路集和])
            n_ = [link.攻击记录(运行时刻=t) for link in self.链路集和]
            self.运行日志['路径状态'][:, t] = np.array([path.生成状态(运行时刻=t) for path in self.路径集和])
            self.运行日志['路径和状态'][:, t] = np.array([path.生成和状态(运行时刻=t) for path in self.路径集和])
            self.运行日志['路径拥塞链路数'][:, t] = np.array([path.生成链路拥塞数(运行时刻=t) for path in self.路径集和])
            self.运行日志['路径丢包率状态'][:, t] = np.array([path.生成探测路径丢包率(运行时刻=t) for path in self.路径集和])

    def 链路详细信息(self):
        '''
        - reutrn 返回 链路的总深度 & 链路的承载的路径数
        '''
        路径集和 = self.路径集和
        链路集和 = self.链路集和
        link_depth = []   # 链路的总深度
        link_carry = []   # 链路的承载路径数
        for link in 链路集和:
            depth = 0; carry = 0
            for path in 路径集和:
                if link in path.路由:
                    depth += (path.路由.index(link) + 1)  # 确定路由深度
                    carry += 1                          # 确定链路承载路径数
                    # print("depth", link.名称, depth)
            link_depth.append(depth)
            link_carry.append(carry)
        
        return link_depth, link_carry
    
    def 路径详细信息(self):
        '''
        - reutrn 返回 路径的Hops
        '''
        路径集和 = self.路径集和
        path_mes = []
        for path in 路径集和:
            path_mes.append(len(path.路由))
        
        return path_mes
        
# noinspection NonAsciiCharacters
class 攻击者基类:
    def __init__(self):
        self.被入侵的路径对象集和: OrderedSet = OrderedSet([])

        self.攻击日志: dict = {
            '攻击位置': [],
            '攻击时刻': []
        }

    @staticmethod
    def 随机攻击策略函数(路径状态_受攻击前: bool, 攻击概率: float = 0.1):
        """
        - 攻击概率，通过 __defaults__ 可以动态修改
        """

        路径状态_受攻击后 = 路径状态_受攻击前 ^ True if np.random.rand() <= 攻击概率 else deepcopy(路径状态_受攻击前)

        return 路径状态_受攻击后

    @staticmethod
    def 生成状态(self: 路径基类, 运行时刻: int = 0, 攻击策略函数: callable = lambda x: x, 攻击日志: dict = None):
        
        """
        - 通过"反转"路径的观测结果，完成攻击动作
        - 攻击策略函数，默认不攻击；通过 __defaults__ 可以动态修改
        - 攻击日志，默认不记录；通过 __defaults__ 可以动态修改
        """

        self.当前时刻 = deepcopy(运行时刻)  # 更新当前时刻

        路径状态_受攻击前 = reduce(lambda i, j: i and j, [link.状态 for link in self.路由])
        路径状态_受攻击后 = 攻击策略函数(路径状态_受攻击前)

        print("路径状态-攻击前-后", 路径状态_受攻击前, 路径状态_受攻击后)

        if 攻击日志 is not None and 路径状态_受攻击后 != 路径状态_受攻击前:
            攻击日志['攻击位置'].append(self.名称)
            攻击日志['攻击时刻'].append(运行时刻)

        self.状态 = deepcopy(路径状态_受攻击后)  # 回写攻击结果
        # return deepcopy(self.状态)

    def 执行入侵(self, 受攻击的路径对象: 路径基类 = None, **kwargs):
        if 受攻击的路径对象 not in self.被入侵的路径对象集和:
            self.被入侵的路径对象集和.add(受攻击的路径对象)
        print("执行入侵")
        受攻击的路径对象.生成状态 = types.MethodType(self.生成状态, 受攻击的路径对象)

    def 清空(self, **kwargs):
        if "被入侵的路径对象集和" in kwargs.keys():
            self.被入侵的路径对象集和 = OrderedSet([])

        elif "攻击日志" in kwargs.keys():
            for i in self.攻击日志.keys():
                self.攻击日志[i] = []

        else:
            pass



class 链路攻击者:
    def __init__(self):
        self.被入侵的链路对象集和: OrderedSet = OrderedSet([])
        self.随机数组 = []
        self.攻击日志: dict = {
            '攻击位置': [],
            '攻击时刻': []
        }

    @staticmethod
    def 链路攻击策略函数(链路状态_受攻击前: bool, 攻击概率: float = 0.1, 随机数组: list = []):
        """
        - 链路攻击策略函数：只能将链路状态从 True 变成 False
        """
        if 链路状态_受攻击前:
            # 只有链路状态为 False 时才执行攻击，以确保只将链路状态从 False 变成 True
            if 随机数组.pop() <= 攻击概率:
                return False
        return 链路状态_受攻击前

    @staticmethod
    def 攻击记录(self: 链路基类, 运行时刻: int = 0, 攻击策略函数: callable = lambda x: x, 攻击日志: dict = None):
        
        """
        - 通过"攻击"(制造拥塞0->1) 链路的观测结果，完成攻击
        - 攻击策略函数，默认不攻击；通过 __defaults__ 可以动态修改
        - 攻击日志，默认不记录；通过 __defaults__ 可以动态修改
        """

        # print("链路状态(前)",self.状态)

        链路状态_受攻击前 = deepcopy(self.状态)

        链路状态_受攻击后 = 攻击策略函数(链路状态_受攻击前)

        # print("链路状态-攻击前-后", self.名称, self.当前时刻, 链路状态_受攻击前, 链路状态_受攻击后)

        if 攻击日志 is not None and 链路状态_受攻击后 != 链路状态_受攻击前:
            攻击日志['攻击位置'].append(self.名称)
            攻击日志['攻击时刻'].append(运行时刻)

        self.状态 = deepcopy(链路状态_受攻击后)  # 回写攻击结果

        return deepcopy(self.状态)

    def 执行入侵(self, 受攻击的链路对象: 链路基类 = None, **kwargs):
        if 受攻击的链路对象 not in self.被入侵的链路对象集和:
            self.被入侵的链路对象集和.add(受攻击的链路对象)
        # print("执行入侵")

        受攻击的链路对象.攻击记录 = types.MethodType(self.攻击记录, 受攻击的链路对象)

    def 清空(self, **kwargs):
        if "被入侵的链路对象集和" in kwargs.keys():
            self.被入侵的链路对象集和 = OrderedSet([])

        elif "攻击日志" in kwargs.keys():
            for i in self.攻击日志.keys():
                self.攻击日志[i] = []

        else:
            pass
    
    def 修正记录(self, 攻击日志, 链路集和, 运行日志):
        
        链路集和 = [i.名称 for i in 链路集和]
        # print(运行日志)
        for i in range(len(攻击日志['攻击位置'])):
            index_ = 链路集和.index(攻击日志['攻击位置'][i])
            运行日志[index_, 攻击日志['攻击时刻'][i]] = False
        
        return 运行日志

def round_(x):
    return round(x, 2)



class Attack_model(网络基类):

    def __init__(self, topology_name: str, 源节点列表: list, n_samples: int, prob: float):
        super().__init__()
        self.源节点列表 = 源节点列表
        self.topology_name = topology_name
        self.n_samples = n_samples
        self.prob = prob

    def sim_data_paths(self):

        self.配置拓扑(f"./topology_zoo/topology_zoo数据集/{self.topology_name}.gml")
        self.部署测量路径(源节点列表=self.源节点列表)
        self.配置参数(异构链路先验拥塞概率 = self.prob)

        运行时间 = self.n_samples + self.n_samples
        self.运行网络(运行的总时间 = 运行时间)

        观测矩阵_ = self.运行日志['路径状态']
        观测结果矩阵 = np.where(观测矩阵_, 0, 1)
        
        B = self.add_noise(观测结果矩阵)

        return B[:self.n_samples], B[self.n_samples:]

    def sim_data_links(self):

        self.配置拓扑(f"./topology_zoo/topology_zoo数据集/{self.topology_name}.gml")
        self.部署测量路径(源节点列表=self.源节点列表)
        self.配置参数(异构链路先验拥塞概率 = self.prob)

        运行时间 = self.n_samples + self.n_samples

        current_time = int(time())

        # 使用当前时间戳作为随机数生成器的种子
        np.random.seed(current_time)

        self.运行网络(运行的总时间 = 运行时间)

        观测矩阵_ = self.运行日志['路径状态']

        路由矩阵_ = self.路由矩阵
        观测矩阵_ = self.运行日志['路径状态']
        路由矩阵 = np.where(路由矩阵_, 1, 0)
        观测矩阵 = np.where(观测矩阵_, 0, 1)

        B = self.add_noise(观测矩阵)

        return B[:self.n_samples], B[self.n_samples:]




    def link_gank(self,  攻击链路集和: list, 攻击概率: float):
        """
        - flag: 是否被攻击, true为要攻击
        """

        self.配置拓扑(f"./topology_zoo/topology_zoo数据集/{self.topology_name}.gml")
        self.部署测量路径(源节点列表=self.源节点列表)
        self.配置参数(异构链路先验拥塞概率 = self.prob)
        alpha = [i.先验拥塞概率  for  i in self.链路集和]
        
        attacker = 链路攻击者()
        # 采用随机攻击
        
        随机数组 = list([np.random.rand() for _ in range(self.n_samples * len(攻击链路集和))])

        分攻击概率 = 攻击概率 / len(攻击链路集和)
        if 分攻击概率 > 1:
            分攻击概率 = 0.99
        attacker = 链路攻击者()
        attacker.链路攻击策略函数.__defaults__ = (分攻击概率, 随机数组)  # 设置随机攻击机率为 0.5
        attacker.攻击记录.__defaults__ = (attacker.链路攻击策略函数, attacker.攻击日志, )  # 植入随机攻击策略


        for link_id in 攻击链路集和:
            attacker.执行入侵(self.链路集和[link_id])  # 入侵路径

        self.运行网络(运行的总时间 = self.n_samples)

        路由矩阵_ = self.路由矩阵
        观测矩阵_ = self.运行日志['路径状态']
        路由矩阵 = np.where(路由矩阵_, 1, 0)
        观测矩阵 = np.where(观测矩阵_, 0, 1)


        B = self.add_noise(观测矩阵)

        return B

    def get_topology_mes(self):

        self.配置拓扑(f"./topology_zoo/topology_zoo数据集/{self.topology_name}.gml")
        self.部署测量路径(源节点列表=self.源节点列表)

        y_, x_ = self.路由矩阵.shape
        
        A = np.where(self.路由矩阵, 1, 0)
        
        return y_, x_, A

    def add_noise(self, array):
        X = array.T

        mean = 0
        stddev = 0.1  # 标准差越大，噪声越强
        X = X + np.random.normal(mean, stddev, X.shape) # 加噪方便高斯拟合
   
        # Z = (X - X.mean(axis=0)) / np.std(X, axis=0)  # z = (x - mu_x) / \sigma_x
        U = stats.norm.cdf(X)  # fits to copula, j-dist r.v. with uniform marginals
        B = stats.beta.ppf(U, a=0.5, b=0.5)  # inverse CDF (percent point function)

    
        return B

def shuffle_data(Y_hat, index, ap):
    '''
    - ap 为攻击频率
    '''
        
    Y = Y_hat.copy()  
    second_column = Y[:, index]
    size = int(len(second_column) * ap) 
    # 打乱第index列的顺序
    np.random.shuffle(second_column[:size])

    # 将打乱后的列重新放回原数组
    Y[:, index] = second_column
    return Y


def 生成笛卡尔积(lst: list, num: int = 3):
    cartesian_product = list(product(lst, repeat=num))

    # 去除重复项
    unique_cartesian_product = list(set(cartesian_product))
    ans = []
    for x in unique_cartesian_product:
        x = list(set(list(x)))
        if x not in ans:
            ans.append(x)
    return ans

def 统计链路承载路径数(net):

    路由矩阵_ = net.路由矩阵
    路由矩阵 = np.where(路由矩阵_, 1, 0)
    # 统计路由矩阵中 每一列中 1 的个数
    ans = np.sum(路由矩阵, axis=0)
    max_ = np.max(ans)
    min_ = np.min(ans)
    means = np.mean(ans)
    variance = statistics.variance(ans)
    # print("max", max_, "min", min_, "means", means,"variance", variance)
    
    return max_, min_, means, variance

def 统计节点出入度(net):

    out_degrees = net.图.out_degree()
    in_degrees = net.图.in_degree()

    out_degrees_values = [x[1] for x in out_degrees]
    in_degrees_values = [x[1] for x in in_degrees]

    # 计算出度的平均值、最大值和最小值
    avg_out_degree = sum(out_degrees_values) / len(out_degrees_values)
    max_out_degree = max(out_degrees_values)
    min_out_degree = min(out_degrees_values)

    # 计算入度的平均值、最大值和最小值
    avg_in_degree = sum(in_degrees_values) / len(in_degrees_values)
    max_in_degree = max(in_degrees_values)
    min_in_degree = min(in_degrees_values)

    # 计算出度和入度的方差
    variance_out_degree = statistics.variance(out_degrees_values)
    variance_in_degree = statistics.variance(in_degrees_values)
    # 打印结果
    # print("出度平均值:", avg_out_degree,"出度最大值:", max_out_degree,"出度最小值:", min_out_degree,"出度方差:", variance_out_degree)
    
def get_attacked_links(name: str):
    
    attacked_links = {'Agis': [0], 'Geant': [11], 'Canada': [0], 'Japan': [0, 11], 'Chinanet': [2], 'Y': [0]}
    
    return attacked_links[name]

def get_random_attacked_links(n, m):
    # 生成包含1到n之间m个不重复数字的列表
    random_numbers = random.sample(range(1, n), m)

    return random_numbers


def get_source_nodes(name: str):
    
    tp_source_nodes = {'Agis': [0, 7], 'Geant': [8, 17, 15], 'Canada': [8, 20], 'Japan': [2, 4], 'Y': [0], 'boom': [0, 1], 'Chinanet': [2, 0], 'easy': [0]}
    return [tp_source_nodes[name]]
    
    
    # net = 网络基类()
    # net.配置拓扑(f"./topology_zoo/topology_zoo数据集/{name}.gml")

    # 统计节点出入度(net)

    # 边缘节点集和 = net.节点字典['边缘节点集和']
    # 所有节点集和 = net.节点字典['所有节点集和']

    # index_ = [所有节点集和.index(i) for i in 边缘节点集和]
    # # print("边缘节点个数", len(index_), index_)
    # # print("总节点数", len(net.节点字典['所有节点集和']))

    # max_hop = 1
    # min_hop = 100
    # sum_hop = []

    # for i in index_:
    #     net.clear_()
    #     net.配置拓扑(f"./topology_zoo/topology_zoo数据集/{name}.gml")
    #     net.部署测量路径(源节点列表=[i])
    #     路径集和 = net.路径集和
    #     sum_hops = []

    #     for 路径 in 路径集和:
    #         sum_hops.append( len(路径.路由) )

    #     max_hop = max(max_hop, max(sum_hops))
    #     min_hop = min(min_hop, min(sum_hops))
    #     sum_hop.append(sum(sum_hops)/len(sum_hops))
        
    # Hops = {"边缘节点个数":len(index_),"节点数":len(net.节点字典['所有节点集和']),"max": max_hop, "min": min_hop, "mean": sum(sum_hop)/len(sum_hop)}
    # # print("HOPS: ","max", max_hop,"min", min_hop,"mean", sum(sum_hop)/len(sum_hop))
    
    # 源节点列表 = 生成笛卡尔积(index_)
    
    # answer = []
    
    # max_hop = 1
    # min_hop = 100
    # sum_hop = []
    # variance_hop = []
    # 链路数 = []
    # 路径数 =  []

    # max_carry = 1
    # min_carry = 100
    # means_carry = []
    # variance_carry = []

    # for i in range(len(源节点列表)):
    #     net.clear_()
    #     net.配置拓扑(f"./topology_zoo/topology_zoo数据集/{name}.gml")
    #     net.部署测量路径(源节点列表=源节点列表[i])
    #     y_, x_ = net.路由矩阵.shape
    #     sum_hops, 路径集和 = [], net.路径集和
    #     if y_ < x_ + 5:
    #         answer.append([y_, x_, 源节点列表[i]])
    #         链路数.append(x_); 路径数.append(y_)
    #         max_, min_, means_, variance = 统计链路承载路径数(net)

    #         max_carry = max(max_carry, max_)
    #         min_carry = min(min_carry, min_)
    #         variance_carry.append(variance)
    #         means_carry.append(means_)    

    #         for 路径 in 路径集和:
    #             sum_hops.append( len(路径.路由) )
    #         max_hop = max(max_hop, max(sum_hops))
    #         min_hop = min(min_hop, min(sum_hops))
            
    #         sum_hop.append(sum(sum_hops)/len(sum_hops))
            
    # print("Carry: ", "max", max_carry, "min", min_carry, "mean", sum(means_carry)/len(means_carry), "variance", sum(variance_carry)/len(variance_carry))
    # print("T-HOPS: ","max", max_hop,"min", min_hop,"mean", sum(sum_hop)/len(sum_hop))
    
    # # print(len(源节点列表), 源节点列表)
    
    # print("符合条件的(Y<X)的情况数量：",len(answer))
    # print("  平均链路数", sum(链路数)/len(链路数), "平均路径数", sum(路径数)/len(路径数))
    answer.sort(key=lambda x: (x[1] - x[0]), reverse=True)
    answer_ = [(i[0], i[1], i[2]) for i in answer]

    # samples = random.sample(answer, int(len(answer) / 2))
    samples = [i[2] for i in answer if len(i[2]) >= 2]
    
    
    num_ = 1
    if len(samples) <= num_:
        return samples
    else:
        return samples[0:num_]


def Evaluate_boolean_performance(True_X, infer_X):
    #inferred_array由算法得出,real_array即真实信息,rounds表示测试了几轮
    ave_FPR=0
    ave_DR=0
    ave_F1=0
    ave_F2=0; ave_Prec=0
    rounds = infer_X.shape[1]; real_array = True_X; inferred_array = infer_X
    for round in range(rounds):
        TP=0
        FP=0
        FN=0
        TN=0
        for link_index in range(len(real_array.T[round])):
            if real_array.T[round][link_index]==0 and inferred_array.T[round][link_index]==0:
                TN+=1
            elif real_array.T[round][link_index]==0 and inferred_array.T[round][link_index]==1:
                FP+=1
            elif real_array.T[round][link_index]==1 and inferred_array.T[round][link_index]==1:
                TP+=1
            elif real_array.T[round][link_index]==1 and inferred_array.T[round][link_index]==0:
                FN+=1
        if TP+FN == 0:
            DR=1
        else:
            DR=TP/(TP+FN)
            
        if FP + TN == 0:
            FPR=0
        else:
            FPR=FP/(FP+TN)
        if TP+FP==0:
            precision=1
        else:
            precision=TP/(TP+FP)
        if TP+FN==0:
            recall=1
        else:
            recall=TP/(TP+FN)
        if TP == 0:
            if sum(real_array.T[round])==0:
                F1=1;F2=1
            else:
                F1=0;F2=0
        else:
            F1 = 2*precision*recall/(precision+recall)
            F2 = 5*precision*recall/(4*precision+recall)
        ave_FPR += FPR
        ave_DR += DR
        ave_F1 += F1
        ave_F2 += F2; ave_Prec += precision

    ave_FPR /=rounds
    ave_DR /=rounds
    ave_F1 /=rounds; ave_F2 /=rounds; ave_Prec /=rounds   

    return [ave_DR, ave_FPR, ave_Prec, ave_F1, ave_F2]

def Evaluate_loss_performance(y_true, y_pred):

    epsilon = 1e-8  # 用于避免除以零

    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))
    range_y = np.max(y_true, axis=0) - np.min(y_true, axis=0)
    nrmse = rmse / (range_y + epsilon)

    mape =  np.mean(np.abs((y_true - y_pred) / (y_true + epsilon)), axis=0) * 100

    # 计算所有时刻的平均 NRMSE 和 MAPE
    average_nrmse = np.mean(nrmse)
    average_mape = np.mean(mape)

    relative_acc = round((1 - average_nrmse) * 1e2, 4)

    return relative_acc, average_nrmse, average_mape


class NoIndent(object):
    """ Value wrapper. """
    def __init__(self, value):
        # 如果 value 为 numpy 列表
        if not isinstance(value, (list, tuple, np.ndarray)):
            raise TypeError('Only lists and tuples can be wrapped')
        self.value = value

class MyEncoder(json.JSONEncoder):
    FORMAT_SPEC = '@@{}@@'  # Unique string pattern of NoIndent object ids.
    regex = re.compile(FORMAT_SPEC.format(r'(\d+)'))  # compile(r'@@(\d+)@@')

    def __init__(self, **kwargs):
        # Keyword arguments to ignore when encoding NoIndent wrapped values.
        ignore = {'cls', 'indent'}

        # Save copy of any keyword argument values needed for use here.
        self._kwargs = {k: v for k, v in kwargs.items() if k not in ignore}
        super(MyEncoder, self).__init__(**kwargs)

    def default(self, obj):
        return (self.FORMAT_SPEC.format(id(obj)) if isinstance(obj, NoIndent)
                    else super(MyEncoder, self).default(obj))

    def iterencode(self, obj, **kwargs):
        format_spec = self.FORMAT_SPEC  # Local var to expedite access.

        # Replace any marked-up NoIndent wrapped values in the JSON repr
        # with the json.dumps() of the corresponding wrapped Python object.
        for encoded in super(MyEncoder, self).iterencode(obj, **kwargs):
            match = self.regex.search(encoded)
            if match:
                id = int(match.group(1))
                no_indent = PyObj_FromPtr(id)
                json_repr = json.dumps(no_indent.value, **self._kwargs)
                # Replace the matched id string with json formatted representation
                # of the corresponding Python object.
                encoded = encoded.replace(
                            '"{}"'.format(format_spec.format(id)), json_repr)

            yield encoded



if __name__ == '__main__':
    net = 网络基类()
    name = 'easy'
    net.配置拓扑(f"./topology_zoo/topology_zoo数据集/{name}.gml")
    sp = get_source_nodes(name)[0]
    print(sp)
    net.部署测量路径(源节点列表=sp)
    
    net.配置参数(异构链路先验拥塞概率 = 0.2)
    print(net.路由矩阵.shape)

    net.运行网络(运行的总时间 = 5)
    观测矩阵_ = net.运行日志['路径状态']
    链路观测矩阵_ = net.运行日志['链路状态']
    丢包率矩阵 = net.运行日志['链路丢包率状态']
    丢包率观测矩阵 = net.运行日志['路径丢包率状态']

    Y = np.where(观测矩阵_, 0, 1)
    X = np.where(链路观测矩阵_, 0, 1)
    True_X = np.where(链路观测矩阵_, 0, 1)

    print("观测矩阵")
    print(Y)
    print("链路观测矩阵")
    print(X)


    print("丢包率观测矩阵")
    print(丢包率观测矩阵)
    print("丢包率矩阵")
    print(丢包率矩阵)













    # A = np.where(net.路由矩阵, 1, 0)
    # print("rank", np.linalg.matrix_rank(A))
    # # 对 A 按列求和
    # print(np.mean(np.sum(A, axis=0)))
    # HOPS = 0
    # for path in net.路径集和:
    #     HOPS += len(path.路由)
    # print("HOPS", HOPS/len(net.路径集和))
    # # for link in net.链路集和:
    # #     print(link)
    # index = [0,11]
    # attention_link, attention_link_label = [], []
    # for link in range(len(net.链路集和)):
    #     if link in index:
    #         attention_link.append(net.链路集和[link].名称)
    #         attention_link_label.append(link)

    # net.导出拓扑(with_path=True, with_geo=False, with_attention=True, attention_link= attention_link, attention_link_label=attention_link_label)

    # # 采用随机攻击
    # 攻击链路集和 = [0]
    # 随机数组 = list([np.random.rand() for _ in range(运行时间 * len(攻击链路集和))])


    # attacker = 链路攻击者()
    # attacker.链路攻击策略函数.__defaults__ = (0.5, 随机数组)  # 设置随机攻击机率为 0.5
    # attacker.攻击记录.__defaults__ = (attacker.链路攻击策略函数, attacker.攻击日志, )  # 植入随机攻击策略

    # if flag:
    #     for link_id in 攻击链路集和:
    #         attacker.执行入侵(net.链路集和[link_id])  # 入侵路径 0

    # net.运行网络(运行的总时间= 运行时间)