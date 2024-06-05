import os
import pickle
import numpy as np
import sys
from multiprocessing import Pool

# sys.path.append('./lstm-aae')
from model import *
from val import *
import json
from tqdm import tqdm


def createModelGetAcc(PROBE_CLASS, model_class, Topology_Type, ratio, rate, seq_len, model_id):
    t_dataset = f'/home/dcz/wp/NS3/ns-3-allinone/ns-3-dev-ns-3.34/dataset/{PROBE_CLASS}/{model_class}/{Topology_Type}/dataset_1_{rate}_t.pkl'
    v_dataset = f'/home/dcz/wp/NS3/ns-3-allinone/ns-3-dev-ns-3.34/dataset/{PROBE_CLASS}/{model_class}/{Topology_Type}/dataset_1_{rate}_v.pkl'
    
    model_name = PROBE_CLASS + '_' + model_class + '_' + Topology_Type + '_' + str(ratio) + '_' + str(rate) + '_' + str(seq_len)
    if model_class == 'Single':
        finished = get_model_single(model_name, t_dataset, seq_len, ratio)
    # elif model_class == 'Multi':
    #     finished = get_model_multi(model_name, t_dataset, 3, ratio)
        
    acc, f1, recall, prec, racc, per_acc = get_acc(model_name, v_dataset, model_class, seq_len, ratio) 
    
    print('Acc:', acc, 'F1:', f1, 'Recall:', recall, 'Prec:', prec, 'Relative_acc:', racc, 'Per_acc', per_acc)
    # 转list
    f1, recall, prec, per_acc = f1.tolist(), recall.tolist(), prec.tolist(), per_acc.tolist()
    # 构造json
    json_ = {}
    json_['model_id'] = model_id
    json_['PobeClass'] = PROBE_CLASS
    json_['ModelClass'] = model_class
    json_['Topology_Type'] = Topology_Type
    json_['ratio'] = ratio
    json_['rate'] = rate
    json_['seq_len'] = seq_len
    json_['finished'] = finished
    json_['Acc'] = acc
    json_['F1'] = f1
    json_['Recall'] = recall
    json_['Prec'] = prec
    json_['Relative_Acc'] = racc
    
    # 保存json
    json_name = f'./result/{PROBE_CLASS}_{model_class}_{Topology_Type}_{ratio}_{rate}_{seq_len}_{model_id}.json'
    with open(json_name, 'w') as file:
        json.dump(json_, file)
        
def process_task(mc, fp, tr, ls, mi, Tt):
    Topology_Type = Tt
    # print('Topology_Type:', Topology_Type)
    if MODEL_CLASS[mc] == 'Single':
        print('Create Model:', '探测流种类', PROBE_CLASS[0], '模型类型', MODEL_CLASS[mc], '拓扑结构', Topology_Type, '探测比例', Time_ratio[tr], '探测流强度', RATE_CLASS_LIST[fp], '单路的seq', LEN_sp[ls], '第', mi, '次')
        createModelGetAcc(PROBE_CLASS[0], MODEL_CLASS[mc], Topology_Type, Time_ratio[tr], RATE_CLASS_LIST[fp], LEN_sp[ls], mi)
    else:
        print('Create Model:', '探测流种类', PROBE_CLASS[0], '模型类型', MODEL_CLASS[mc], '拓扑结构', Topology_Type, '探测比例', Time_ratio[tr], '探测流强度', RATE_CLASS_LIST[fp], '多路的seq', 3, '第', mi, '次')
        createModelGetAcc(PROBE_CLASS[0], MODEL_CLASS[mc], Topology_Type, Time_ratio[tr], RATE_CLASS_LIST[fp], 3, mi)

# 设置参数
RATE_CLASS_LIST = [3.0] #

PROBE_CLASS = ['E']
MODEL_CLASS = ['Single']
Time_ratio = [0.8]
LEN_sp = [3]


# 使用进程池来并行处理任务，最大进程数设置为10
if __name__ == '__main__':
    argv = sys.argv 
    Topology_Type = argv[1]
    print('Topology_Type:', Topology_Type)
    
    with Pool(processes=1) as pool:
        tasks = [(mc, fp, tr, ls, mi, Topology_Type) for mc in range(len(MODEL_CLASS))
                                         for fp in range(len(RATE_CLASS_LIST))
                                         for tr in range(len(Time_ratio))
                                         for ls in range(len(LEN_sp)) if MODEL_CLASS[mc] == 'Single'
                                         for mi in range(5)]
        # 总次数
        print('任务共有', len(tasks))
        pool.starmap(process_task, tasks)