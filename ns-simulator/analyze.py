from __future__ import division
import sys
import os
import csv, math
import numpy as np
import bisect
import matplotlib.pyplot as plt
import numpy as np
import datetime
import pickle

try:
    from xml.etree import cElementTree as ElementTree
except ImportError:
    from xml.etree import ElementTree
    
def read_delay(mode, name, probe_flow_ids, 背景流个数):
    
    # 创建all_delays列表, 长度为 len(probe_flow_ids)
    probs_delays_ids = [[] for _ in range(len(probe_flow_ids))]
    probe_gaps_ids = [[] for _ in range(len(probe_flow_ids))]
    probs_times_ids = [[] for _ in range(len(probe_flow_ids))]
    
    # 指定txt文件路径
    file_path = f'./txt/flow_monitor{name}.txt'

    # 用于存储flowid为5的记录的列表
    result_records = []
    cong_records = None
    # 打开文件并读取数据
    with open(file_path, 'r') as file:
        # 逐行读取文件内容
        for line in file:
            # 分割每行数据，假设数据用空格分隔
            data = line.split(',')
            if len(data) == 背景流个数:
                cong_records = [int(data[i][0]) for i in range(背景流个数)]
                continue
            
            # 检查flowid是否为7
            if int(data[0]) in probe_flow_ids:
                index = probe_flow_ids.index(int(data[0]))
                # 提取delay、packetid和packetsize，并添加到结果列表
                flowid, packetid, delay, packetsize, time = map(float, data)
                result_records.append((packetid, delay, packetsize, time))
                probs_delays_ids[index].append(delay)
                probs_times_ids[index].append(time)

    times = [time for _, _, _, time in result_records]
    x_s = [2e-4 * 1 * math.exp(abs(i-5)) for i in range(1, 10)]
    ans = []
    间隔 = []
    # round_n = int(len(probs_times_ids[-1]) / 10)
    min_round = min([len(probs_times_ids[i]) for i in range(len(probe_flow_ids))])
    round_n = min_round // 10
    print("min_round", round_n)
    
    if mode == 1:
        for p in range(len(probs_delays_ids)):
            times = probs_times_ids[p]
            for i in range(round_n):
                delat_time = []
                for j in range(1,10):
                    delat_time.append(times[i*10+j]-times[i*10+j-1])
                delat_time = np.array(delat_time) ; x_s = np.array(x_s)
                # if (delat_time[0] / x_s[0]) > 0.95 and (delat_time[0] / x_s[0]) < 1.05:
                probe_gaps_ids[p].append(delat_time / x_s)

        probe_gaps_ids = np.array(probe_gaps_ids)
        print('probs_gaps_shape', probe_gaps_ids.shape)
        
    elif mode == 2:
        delays = [delay for _, delay, _, _ in result_records]
        ans = delays
        # print(len(ans), ans)
    
    min_lenth = min([len(probs_delays_ids[i]) for i in range(len(probs_delays_ids))])
    probe_delays_ids = [probs_delays_ids[i][:min_lenth] for i in range(len(probs_delays_ids))]
    probe_delays_ids = np.array(probe_delays_ids)
    print(probe_delays_ids.shape)
    # assert probs_delays_ids.shape[1] <= 40 and probs_delays_ids.shape[1] >= 30  
    
    return probe_delays_ids, cong_records, probe_gaps_ids
    
def create_and_pickle_file(my_array, cong_record, abw, pkl, 间隔, probe_rate, tp_name, anomoly_ratio):
    # 获取当前时间戳
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
    anomoly_ratio = int(anomoly_ratio) / 5
    anomoly_ratio = round(anomoly_ratio, 1)
    # 创建 'data' 文件夹，如果它不存在
    folder_name = 'data/' + tp_name + '/' + str(anomoly_ratio) + '/' + 'probe_rate_' + str(probe_rate)
    if not os.path.exists(f'{folder_name}'):
        os.makedirs(f'{folder_name}')

    # 文件名为当前时间戳，保存在 'data' 文件夹下
    filename = os.path.join(f'{folder_name}', f"{timestamp}.pkl")

    # 将数据保存为 pickle 文件
    with open(filename, 'wb') as file:
        data_to_pickle = {'numpy_array': my_array, 'cong': cong_record, 'abw': abw, 'pkl': pkl, 'gaps': 间隔, 'probe_rate': probe_rate}
        pickle.dump(data_to_pickle, file)

    print(f"文件 '{filename}' 已创建并保存为 pickle 文件{cong_record}")
    
    
def parse_time_ns(tm):
    if tm.endswith('ns'):
        return float(tm[:-2])
    raise ValueError(tm)

def entropy(probabilities):
    # 确保概率总和为1
    if abs(sum(probabilities) - 1.0) > 1e-2:
        raise ValueError("概率总和必须等于1")

    # 初始化熵值
    entropy_value = 0

    # 计算每个事件的熵并累加
    for prob in probabilities:
        if prob > 0:
            entropy_value -= prob * math.log2(prob)

    return entropy_value


def deal_with(flow):
    t = flow.delayHistogram
    width = t.bins[0][1]
    bins = t.bins[1:]
    new_bins = [(x[0] - width, x[1], x[2]) for x in bins]
    b_min, b_max = new_bins[0][0], new_bins[-1][0] + new_bins[-1][1]
    width_ = (b_max - b_min) / 9
    hist = [i for i in np.arange(b_min, b_max + width_, width_)]
    list_ = np.array([0 for i in range(10)])
    for bin in bins:
        index = bisect.bisect_right(hist, bin[0])
        list_[index - 1] += bin[2]
    
    probabilities = list_ / sum(list_)

    return entropy(probabilities)

def ternary_to_digits(ternaryNumber, n):
    digits = []  # 存储恢复的数字
    current = ternaryNumber  # 当前处理的数
    
    # 循环n次或直到current为0
    for _ in range(n):
        digit = current % 3  # 获取当前最低位的数字
        digits.append(digit)  # 添加到数组中
        current = current // 3  # 准备处理下一位
        
        if current == 0:
            break  # 如果已经处理完所有位，则退出循环
    
    # 如果digits长度小于n，添加0直到长度为n
    while len(digits) < n:
        digits.append(0)
    
    # 因为是从最低位开始恢复的，所以需要反转数组以匹配原始顺序
    return digits[::-1]


## FiveTuple
class FiveTuple(object):
    ## class variables
    ## @var sourceAddress 
    #  source address
    ## @var destinationAddress 
    #  destination address
    ## @var protocol 
    #  network protocol
    ## @var sourcePort 
    #  source port
    ## @var destinationPort 
    #  destination port
    ## @var __slots_ 
    #  class variable list
    __slots_ = ['sourceAddress', 'destinationAddress', 'protocol', 'sourcePort', 'destinationPort']
    def __init__(self, el):
        '''! The initializer.
        @param self The object pointer.
        @param el The element.
        '''
        self.sourceAddress = el.get('sourceAddress')
        self.destinationAddress = el.get('destinationAddress')
        self.sourcePort = int(el.get('sourcePort'))
        self.destinationPort = int(el.get('destinationPort'))
        self.protocol = int(el.get('protocol'))
        
## Histogram
class Histogram(object):
    ## class variables
    ## @var bins
    #  histogram bins
    ## @var nbins
    #  number of bins
    ## @var number_of_flows
    #  number of flows
    ## @var __slots_
    #  class variable list
    __slots_ = 'bins', 'nbins', 'number_of_flows'
    def __init__(self, el=None):
        '''! The initializer.
        @param self The object pointer.
        @param el The element.
        '''
        self.bins = []
        if el is not None:
            self.nbins = int(el.get('nBins'))
            for bin in el.findall('bin'):
                self.bins.append( (float(bin.get("start")), float(bin.get("width")), int(bin.get("count"))) )

## Flow
class Flow(object):
    ## class variables
    ## @var flowId
    #  delay ID
    ## @var delayMean
    #  mean delay
    ## @var packetLossRatio
    #  packet loss ratio
    ## @var rxBitrate
    #  receive bit rate
    ## @var txBitrate
    #  transmit bit rate
    ## @var fiveTuple
    #  five tuple
    ## @var packetSizeMean
    #  packet size mean
    ## @var probe_stats_unsorted
    #  unsirted probe stats
    ## @var hopCount
    #  hop count
    ## @var flowInterruptionsHistogram
    #  flow histogram
    ## @var rx_duration
    #  receive duration
    ## @var __slots_
    #  class variable list
    __slots_ = ['flowId', 'delayMean', 'packetLossRatio', 'rxBitrate', 'txBitrate',
                'fiveTuple', 'packetSizeMean', 'probe_stats_unsorted',
                'hopCount', 'delayHistogram', 'rx_duration']
    def __init__(self, flow_el):
        '''! The initializer.
        @param self The object pointer.
        @param flow_el The element.
        '''
        self.flowId = int(flow_el.get('flowId'))
        rxPackets = float(flow_el.get('rxPackets'))
        txPackets = float(flow_el.get('txPackets'))
        self.rxPackets = rxPackets
        self.txPackets = txPackets

        tx_duration = (parse_time_ns (flow_el.get('timeLastTxPacket')) - parse_time_ns(flow_el.get('timeFirstTxPacket')))*1e-9
        rx_duration = (parse_time_ns (flow_el.get('timeLastRxPacket')) - parse_time_ns(flow_el.get('timeFirstRxPacket')))*1e-9
        self.rx_duration = rx_duration
        self.probe_stats_unsorted = []
        if rxPackets:
            self.hopCount = float(flow_el.get('timesForwarded')) / rxPackets + 1
        else:
            self.hopCount = -1000
        if rxPackets:
            self.delayMean = float(flow_el.get('delaySum')[:-2]) / rxPackets * 1e-9
            self.packetSizeMean = float(flow_el.get('rxBytes')) / rxPackets
        else:
            self.delayMean = None
            self.packetSizeMean = None
        if rx_duration > 0:
            self.rxBitrate = float(flow_el.get('rxBytes'))*8 / rx_duration
        else:
            self.rxBitrate = None
        if tx_duration > 0:
            self.txBitrate = float(flow_el.get('txBytes'))*8 / tx_duration
        else:
            self.txBitrate = None
        lost = float(flow_el.get('lostPackets'))
        #print "rxBytes: %s; txPackets: %s; rxPackets: %s; lostPackets: %s" % (flow_el.get('rxBytes'), txPackets, rxPackets, lost)
        if rxPackets == 0:
            self.packetLossRatio = None
        else:
            self.packetLossRatio = (lost / (rxPackets + lost))

        interrupt_hist_elem = flow_el.find("delayHistogram")
        if interrupt_hist_elem is None:
            self.delayHistogram = None
        else:
            self.delayHistogram = Histogram(interrupt_hist_elem)

## ProbeFlowStats
class ProbeFlowStats(object):
    ## class variables
    ## @var probeId
    #  probe ID
    ## @var packets
    #  network packets
    ## @var bytes
    #  bytes
    ## @var delayFromFirstProbe
    #  delay from first probe
    ## @var __slots_
    #  class variable list
    __slots_ = ['probeId', 'packets', 'bytes', 'delayFromFirstProbe']

## Simulation
class Simulation(object):
    ## class variables
    ## @var flows
    #  list of flows
    def __init__(self, simulation_el):
        '''! The initializer.
        @param self The object pointer.
        @param simulation_el The element.
        '''
        self.flows = []
        FlowClassifier_el, = simulation_el.findall("Ipv4FlowClassifier")
        flow_map = {}
        for flow_el in simulation_el.findall("FlowStats/Flow"):
            flow = Flow(flow_el)
            flow_map[flow.flowId] = flow
            self.flows.append(flow)
        for flow_cls in FlowClassifier_el.findall("Flow"):
            flowId = int(flow_cls.get('flowId'))
            flow_map[flowId].fiveTuple = FiveTuple(flow_cls)

        for probe_elem in simulation_el.findall("FlowProbes/FlowProbe"):
            probeId = int(probe_elem.get('index'))
            for stats in probe_elem.findall("FlowStats"):
                flowId = int(stats.get('flowId'))
                s = ProbeFlowStats()
                s.packets = int(stats.get('packets'))
                s.bytes = float(stats.get('bytes'))
                s.probeId = probeId
                if s.packets > 0:
                    s.delayFromFirstProbe =  parse_time_ns(stats.get('delayFromFirstProbeSum')) / float(s.packets)
                else:
                    s.delayFromFirstProbe = 0
                flow_map[flowId].probe_stats_unsorted.append(s)


def main(argv):
    xml_file_name = f'./txt/test_{argv[1]}.xml'
    file_obj = open(xml_file_name)

    sys.stdout.flush()        
    level = 0
    sim_list = []
    for event, elem in ElementTree.iterparse(file_obj, events=("start", "end")):
        if event == "start":
            level += 1
        if event == "end":
            level -= 1
            if level == 0 and elem.tag == 'FlowMonitor':
                sim = Simulation(elem)
                sim_list.append(sim)
                elem.clear() # won't need this any more
                sys.stdout.write(".")
                sys.stdout.flush()
    # print(" done.")

    if len(argv) == 7: # 文件名， 文件id, 背景流个数，带宽
        
        bandwith = argv[3]
        probe_rate = argv[4]
        topology_name = argv[5]
        anomoly_ratio = int(argv[6])
        print('anomoly_ratio', anomoly_ratio)
        print('anomoly_ratio', anomoly_ratio / 5)
        probe_rate = float(probe_rate)
        sum_rxBitrate = 0
        sum_txPackets, sum_rxPackets = 0, 0
        sourceAddress = []
        endAddress = []

    
        for sim in sim_list:
            for flow in sim.flows:
                sum_txPackets += flow.txPackets
                sum_rxPackets += flow.rxPackets
                sum_rxBitrate += flow.rxBitrate
                t = flow.fiveTuple
                if t.destinationPort == 38:
                    sourceAddress.append(t.sourceAddress)
                    endAddress.append(t.destinationAddress)


        PLR = round(((sum_txPackets - sum_rxPackets) / sum_txPackets) * 1e2, 2)

        print(sourceAddress, endAddress)
        # 对 sourceAddress 进行排序，endAddress 根据 sourceAddress 排序
        sourceAddress, endAddress = zip(*sorted(zip(sourceAddress, endAddress)))
        print(sourceAddress, endAddress)
        
        flows = sim_list[0].flows
        背景流个数 = int(argv[2])
        print("背景流个数", 背景流个数, len(endAddress))
        assert 背景流个数 == len(endAddress)
        print("背景流个数", 背景流个数)
        bandwith_res = [0 for i in range(背景流个数)]

        probe_flow_ids = [0 for i in range(背景流个数)] # 记录探测流s的id
        probe_pkl = [0 for i in range(背景流个数)]      # 记录探测流的丢包率
        
        for sim in sim_list:
            for flow in sim.flows:
                
                t = flow.fiveTuple
                proto = {6: 'TCP', 17: 'UDP'} [t.protocol]
                Proporation = 0
                print("FlowID: %i (%s %s/%s --> %s/%i)" % \
                    (flow.flowId, proto, t.sourceAddress, t.sourcePort, t.destinationAddress, t.destinationPort))
                if flow.txBitrate is None:
                    print("\tTX bitrate: None")
                else:
                    print("\tTX bitrate: %.2f Mbit/s" % (flow.txBitrate*1e-6,))
                if flow.rxBitrate is None:
                    print("\tRX bitrate: None")
                else:
                    print("\tRX bitrate: %.2f Mbit/s" % (flow.rxBitrate*1e-6,))
                    Proporation = round((flow.rxBitrate / sum_rxBitrate), 4)
                    
                print("\tProporation", round((Proporation * 100), 2), "%")
                    
                if flow.delayMean is None:
                    print("\tMean Delay: None")
                else:
                    print("\tMean Delay: %.2f ms" % (flow.delayMean*1e3,))
                if flow.packetLossRatio is None:
                    print("\tPacket Loss Ratio: None")
                else:
                    print("\tPacket Loss Ratio: %.2f %%" % (flow.packetLossRatio*100))
                # print("\tPacket Loss / Proporation", round((flow.packetLossRatio * 1e2/ Proporation), 2))
                print("")

                # 记录探测流的flowid
                if t.destinationPort == 38 and t.destinationAddress in endAddress:
                    index_ = endAddress.index(t.destinationAddress)
                    probe_flow_ids[index_] = flow.flowId
                    probe_pkl[index_] = round(flow.packetLossRatio * 100, 2)
                    if t.sourceAddress in sourceAddress and t.destinationAddress in endAddress and sourceAddress.index(t.sourceAddress) == endAddress.index(t.destinationAddress):
                        index_ = sourceAddress.index(t.sourceAddress)
                        bandwith_res[index_] += flow.rxBitrate * 1e-6 / int(bandwith)
        
        
        print("probe_pkl", probe_pkl)
        print("flow_ids", probe_flow_ids)
        CTR = round((sum_rxBitrate / 1e6 / 4) / 0.1, 2)
        probe_delays, cong_record, probe_gaps = read_delay(1, argv[1], probe_flow_ids, int(argv[2]))
        # cong_record = ternary_to_digits(int(cong_record), len(endAddress))
        print("probe_delays", cong_record, type(cong_record))
        create_and_pickle_file(probe_delays, cong_record, np.round(np.array(bandwith_res) * 100, 1), 
                               probe_pkl, probe_gaps, probe_rate, topology_name, anomoly_ratio)
        print("utilize", CTR, "%")
        print("Packet loss rate", PLR, "%")

    if len(argv) == 3: # 文件名， 文件id, 带宽
        
        bandwith = argv[2]
        sum_rxBitrate = 0
        sum_txPackets, sum_rxPackets = 0, 0
        sourceAddress = []
        endAddress = []

    
        for sim in sim_list:
            for flow in sim.flows:
                sum_txPackets += flow.txPackets
                sum_rxPackets += flow.rxPackets
                sum_rxBitrate += flow.rxBitrate


        PLR = round(((sum_txPackets - sum_rxPackets) / sum_txPackets) * 1e2, 2)


        
        flows = sim_list[0].flows
        背景流个数 = 3
        bandwith_res = [0 for i in range(背景流个数)]

        probe_flow_ids = [0 for i in range(背景流个数)] # 记录探测流s的id
        probe_pkl = [0 for i in range(背景流个数)]      # 记录探测流的丢包率
        
        for sim in sim_list:
            for flow in sim.flows:
                
                t = flow.fiveTuple
                proto = {6: 'TCP', 17: 'UDP'} [t.protocol]
                Proporation = 0
                print("FlowID: %i (%s %s/%s --> %s/%i)" % \
                    (flow.flowId, proto, t.sourceAddress, t.sourcePort, t.destinationAddress, t.destinationPort))
                if flow.txBitrate is None:
                    print("\tTX bitrate: None")
                else:
                    print("\tTX bitrate: %.2f Mbit/s" % (flow.txBitrate*1e-6,))
                if flow.rxBitrate is None:
                    print("\tRX bitrate: None")
                else:
                    print("\tRX bitrate: %.2f Mbit/s" % (flow.rxBitrate*1e-6,))
                    Proporation = round((flow.rxBitrate / sum_rxBitrate), 4)
                    
                print("\tProporation", round((Proporation * 100), 2), "%")
                    
                if flow.delayMean is None:
                    print("\tMean Delay: None")
                else:
                    print("\tMean Delay: %.2f ms" % (flow.delayMean*1e3,))
                if flow.packetLossRatio is None:
                    print("\tPacket Loss Ratio: None")
                else:
                    print("\tPacket Loss Ratio: %.2f %%" % (flow.packetLossRatio*100))
                # print("\tPacket Loss / Proporation", round((flow.packetLossRatio * 1e2/ Proporation), 2))
                print("")



if __name__ == '__main__':
    main(sys.argv)



