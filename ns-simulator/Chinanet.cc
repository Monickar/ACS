#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <random>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <chrono>
#include <iomanip>
#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/csma-module.h"
#include "ns3/internet-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/applications-module.h"
#include "ns3/netanim-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/command-line.h"

using namespace ns3;
std::map<int, int> dict_interf;
std::map<int, int> dict_interf_2;

std::vector<std::string> delays(5); // 创建一个有5个元素的向量
std::vector<std::string> bandwidths(5); // 创建一个有5个元素的向量

int start_time = 0;
int end_time = 8;

void setDelays(std::vector<std::string>& delays, int count) {
    // 确保count的值有效
    count = std::min(count, static_cast<int>(delays.size()));
    count = std::max(count, 0);

    // 初始化随机数生成器
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(20, 21 + 10 * count);  // 生成20到30之间的随机整数

    // 创建一个索引的向量
    std::vector<int> indices(delays.size());
    std::iota(indices.begin(), indices.end(), 0); // 填充0到delays.size() - 1

    // 打乱索引
    std::shuffle(indices.begin(), indices.end(), gen);

    // 设置随机分布的delays
    int delay_size = int(delays.size());
    for (int i = 0; i < delay_size; ++i) {
        // 生成20ms到30ms之间的随机延迟
        int randomDelay = dist(gen);
        delays[indices[i]] = std::to_string(randomDelay) + "ms";
    }

    printf("delays: %s %s %s %s %s\n", delays[0].c_str(), delays[1].c_str(), delays[2].c_str(), delays[3].c_str(), delays[4].c_str());
}

void setBandwidths(std::vector<std::string>& bandwidth, int count) {
    // 确保count的值有效
    count = std::min(count, static_cast<int>(bandwidth.size()));
    count = std::max(count, 0);

    // 设置所有值为"10Mbps"
    std::fill(bandwidth.begin(), bandwidth.end(), "10Mbps");

    // 初始化随机数生成器
    std::random_device rd;
    std::mt19937 gen(rd());

    // 创建一个索引的向量
    std::vector<int> indices(bandwidth.size());
    std::iota(indices.begin(), indices.end(), 0); // 填充0到delays.size() - 1

    // 打乱索引
    std::shuffle(indices.begin(), indices.end(), gen);

    // 根据count设置"45ms"
    for(int i = 0; i < count; ++i) {
        bandwidth[indices[i]] = "10Mbps";
    }
}

// 生成1-8随机数
int generateRandomNumbers() {
    std::srand(std::time(0));  
    int randomNum;
    randomNum = std::rand() % (5 - 1) + 1;  
    return randomNum;
}
int max_num(int a, int b, int c, int d, int e) {
    int max = a;

    if (b > max) {
        max = b;
    }

    if (c > max) {
        max = c;
    }

    if (d > max) {
        max = d;
    }

    if (e > max) {
        max = e;
    }

    return max;
}

void appendToFile(const std::string& filename, const std::string& content) {
    // 打开文件，以追加模式打开
    std::ofstream file;
    file.open(filename, std::ios::app);

    // 检查文件是否成功打开
    if (!file.is_open()) {
        std::cerr << "无法打开文件: " << filename << std::endl;
        return;
    }
    // 写入内容到文件
    file << content << std::endl;
    // 关闭文件
    file.close();
}

// 生成随机数
double getRandomDouble(double min, double max) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(min, max);
    return dis(gen);
}
// 生成随机整数
int getRandomInt(int min, int max) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(min, max);
    return dis(gen);
}

// 根据概率p生成0或1
int return_1(double p) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1); // 生成0到1之间的浮点数
    double randomValue = dis(gen);
    return randomValue < p ? 1 : 0; // 如果生成的数小于p，则返回1；否则返回0
}

void probe_(NodeContainer nodes, std::vector<Ipv4InterfaceContainer> interfaces, int PacketSize, int DataRate){
  OnOffHelper onoff ("ns3::UdpSocketFactory", Address (InetSocketAddress (interfaces[3].GetAddress (0), 10)));
  onoff.SetAttribute ("PacketSize", UintegerValue (PacketSize));
  StringValue DataRate_str = std::to_string(DataRate) + "Mbps";
  onoff.SetAttribute ("DataRate", StringValue (DataRate_str));
  onoff.SetAttribute ("OnTime", StringValue ("ns3::ConstantRandomVariable[Constant=1]"));
  onoff.SetAttribute ("OffTime", StringValue ("ns3::ConstantRandomVariable[Constant=0]"));

  ApplicationContainer clientApps = onoff.Install (nodes.Get (3));
  clientApps.Start (Seconds (3.0));
  clientApps.Stop (Seconds (15.0));

  // 接受探测流
  // 创建探测流接收端
  PacketSinkHelper sink("ns3::UdpSocketFactory", Address(InetSocketAddress(Ipv4Address::GetAny(), 10)));

  ApplicationContainer sinkAppsAC = sink.Install(nodes.Get(4));
  sinkAppsAC.Start(Seconds(2));
  sinkAppsAC.Stop(Seconds(9));

}

void bck_(NodeContainer nodes, std::vector<Ipv4InterfaceContainer> interfaces, int sender, int receiver, int PacketSize, double DataRate){

  //UDP 背景流
  OnOffHelper onoff ("ns3::UdpSocketFactory", Address (InetSocketAddress (interfaces[dict_interf[receiver]].GetAddress (dict_interf_2[receiver]), 10)));
  onoff.SetAttribute ("PacketSize", UintegerValue (PacketSize));
  StringValue DataRate_str_0 = std::to_string(DataRate/5+generateRandomNumbers()/400) + "Mbps";
  onoff.SetAttribute ("DataRate", StringValue (DataRate_str_0));
  onoff.SetAttribute ("OnTime", StringValue ("ns3::ConstantRandomVariable[Constant=0.21]"));
  onoff.SetAttribute ("OffTime", StringValue ("ns3::ConstantRandomVariable[Constant=0.2]"));

  ApplicationContainer clientApps = onoff.Install (nodes.Get (sender));
  clientApps.Start (Seconds (start_time+generateRandomNumbers()/15.0));
  clientApps.Stop (Seconds (end_time));

  //UDP 背景流1
  OnOffHelper onoff1 ("ns3::UdpSocketFactory", Address (InetSocketAddress (interfaces[dict_interf[receiver]].GetAddress (dict_interf_2[receiver]), 15)));
  onoff1.SetAttribute ("PacketSize", UintegerValue (PacketSize));
  StringValue DataRate_str_2 = std::to_string(DataRate/3) + "Mbps";
  onoff1.SetAttribute ("DataRate", StringValue (DataRate_str_2));
  onoff1.SetAttribute ("OnTime", StringValue ("ns3::ConstantRandomVariable[Constant=0.2]"));
  onoff1.SetAttribute ("OffTime", StringValue ("ns3::ConstantRandomVariable[Constant=0.2]"));

  ApplicationContainer clientApps1 = onoff1.Install (nodes.Get (sender));
  clientApps1.Start (Seconds (start_time+generateRandomNumbers()/15.0));
  clientApps1.Stop (Seconds (end_time));

  //TCP 背景流
  PPBPHelper ppbp = PPBPHelper ("ns3::TcpSocketFactory", InetSocketAddress (interfaces[dict_interf[receiver]].GetAddress (dict_interf_2[receiver]), 12));
  ppbp.SetAttribute ("PacketSize", UintegerValue (PacketSize));
  StringValue DataRate_str_1 = std::to_string(DataRate/5) + "Mb/s";
  ppbp.SetAttribute ("BurstIntensity", StringValue(DataRate_str_1));
  // ppbp.SetAttribute ("DataRate", StringValue (DataRate_str_1));
  // ppbp.SetAttribute ("OnTime", StringValue ("ns3::ConstantRandomVariable[Constant=0.3]"));
  // ppbp.SetAttribute ("OffTime", StringValue ("ns3::ConstantRandomVariable[Constant=0.1]"));
  ApplicationContainer apps = ppbp.Install (nodes.Get (sender));
  apps.Start (Seconds (start_time+generateRandomNumbers()/100));
  apps.Stop (Seconds (end_time));

  // 接受udp背景流
  PacketSinkHelper sink("ns3::UdpSocketFactory", Address(InetSocketAddress(Ipv4Address::GetAny(), 10)));
  ApplicationContainer sinkAppsAC = sink.Install(nodes.Get(receiver));
  sinkAppsAC.Start(Seconds(start_time+generateRandomNumbers()/100));
  sinkAppsAC.Stop(Seconds(end_time));

  // 接受udp背景流
  PacketSinkHelper sink1("ns3::UdpSocketFactory", Address(InetSocketAddress(Ipv4Address::GetAny(), 15)));
  ApplicationContainer sinkAppsAC1 = sink1.Install(nodes.Get(receiver));
  sinkAppsAC1.Start(Seconds(start_time+generateRandomNumbers()/100));
  sinkAppsAC1.Stop(Seconds(end_time));

  // 接受tcp背景流
  PacketSinkHelper sink2("ns3::TcpSocketFactory", Address(InetSocketAddress(Ipv4Address::GetAny(), 12)));
  ApplicationContainer sinkAppsAC2 = sink2.Install(nodes.Get(receiver));
  sinkAppsAC2.Start(Seconds(start_time));
  sinkAppsAC2.Stop(Seconds(end_time));

}
void bck_1(NodeContainer nodes, std::vector<Ipv4InterfaceContainer> interfaces, int sender, int receiver, int PacketSize, double DataRate){

  //UDP 背景流
  OnOffHelper onoff ("ns3::UdpSocketFactory", Address (InetSocketAddress (interfaces[receiver-1].GetAddress (1), 11)));
  onoff.SetAttribute ("PacketSize", UintegerValue (PacketSize));
  StringValue DataRate_str_0 = std::to_string(DataRate) + "Mbps";
  onoff.SetAttribute ("DataRate", StringValue (DataRate_str_0));
  onoff.SetAttribute ("OnTime", StringValue ("ns3::ConstantRandomVariable[Constant=0.02]"));
  onoff.SetAttribute ("OffTime", StringValue ("ns3::ConstantRandomVariable[Constant=0.2]"));

  ApplicationContainer clientApps = onoff.Install (nodes.Get (sender));
  clientApps.Start (Seconds (start_time));
  clientApps.Stop (Seconds (end_time));

  // 接受udp背景流
  PacketSinkHelper sink("ns3::UdpSocketFactory", Address(InetSocketAddress(Ipv4Address::GetAny(), 11)));
  ApplicationContainer sinkAppsAC = sink.Install(nodes.Get(receiver));
  sinkAppsAC.Start(Seconds(start_time+generateRandomNumbers()/100));
  sinkAppsAC.Stop(Seconds(end_time));

}

void probe_udp_client(NodeContainer nodes, std::vector<Ipv4InterfaceContainer> interfaces, int sender, int receiver, double probe_rate){

 // udp-echo-client.cc
//设置客户端应用层

  double value = probe_rate * pow(10, -3); // 计算 te-4 的值，这里是 2.2e-4
  UdpEchoClientHelper echoClient (interfaces[dict_interf[sender]].GetAddress (dict_interf_2[sender]), 38);   
  echoClient.SetAttribute ("MaxPackets", UintegerValue (4000));       
  echoClient.SetAttribute ("Interval", TimeValue (Seconds (value)));  
  echoClient.SetAttribute ("PacketSize", UintegerValue (400));     
  
//本例中，让客户端发送一个1024byte的数据分组。

  ApplicationContainer clientApps = echoClient.Install (nodes.Get (receiver));
  clientApps.Start (Seconds (start_time+1));
  clientApps.Stop (Seconds (end_time-1));

}



int main (int argc, char *argv[]) {

   double myCustomParam = 0.5;
   double probe_v = 2;
   int anomolylink = 0;

    CommandLine cmd_;
    cmd_.AddValue("myParam", "The value of the congestion prob", myCustomParam);
    cmd_.AddValue("probe_v", "The value of the probe_v", probe_v);
    cmd_.AddValue("anomoly_link", "The value of the anomoly", anomolylink);
    cmd_.Parse(argc, argv);

    // 构造时间字符串
    auto now_1 = std::chrono::high_resolution_clock::now();
    auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(now_1.time_since_epoch()).count();
    time_t now = time(0);
    tm *ltm = localtime(&now);
    std::ostringstream oss_time;
    oss_time << ltm->tm_hour + ltm->tm_min * 2 + ltm->tm_sec;
    std::string time_str = oss_time.str();
    int time_int = std::stoi(time_str) * generateRandomNumbers() + generateRandomNumbers() + millis % 10000;
    printf("%s %d %ld\n", time_str.c_str(), time_int, millis);

    double congestion_prob = myCustomParam;
    double probe_rate = probe_v;
    int anomoly_link = anomolylink;
    std::cout << "The value of the congestion prob is: " << congestion_prob << std::endl;
    std::cout << "The value of the probe_v is: " << probe_rate << std::endl;
    std::cout << "The value of the anomoly is: " << anomoly_link << std::endl;

    int sf_0 = return_1(congestion_prob); int sf_5 = return_1(congestion_prob); int sf_6 = return_1(congestion_prob);
    int sf_1 = return_1(congestion_prob); int sf_7 = return_1(congestion_prob); int sf_8 = return_1(congestion_prob);
    int sf_2 = return_1(congestion_prob); int sf_9 = return_1(congestion_prob); int sf_10 = return_1(congestion_prob);
    int sf_3 = return_1(congestion_prob); int sf_11 = return_1(congestion_prob); int sf_12 = return_1(congestion_prob);
    int sf_4 = return_1(congestion_prob); int sf_13 = return_1(congestion_prob); int sf_14 = return_1(congestion_prob);
    int sf_15 = return_1(congestion_prob); int sf_16 = return_1(congestion_prob); int sf_17 = return_1(congestion_prob);
    int sf_18 = return_1(congestion_prob); int sf_19 = return_1(congestion_prob); int sf_20 = return_1(congestion_prob);
    int sf_21 = return_1(congestion_prob); int sf_22 = return_1(congestion_prob); int sf_23 = return_1(congestion_prob);
    int sf_24 = return_1(congestion_prob); int sf_25 = return_1(congestion_prob); int sf_26 = return_1(congestion_prob);
    int sf_27 = return_1(congestion_prob);

    int m1 = sf_0 + sf_1 + sf_3 + sf_5; 
    int m2 = sf_0 + sf_1 + sf_2;
    int m3 = sf_0 + sf_1 + sf_3 + sf_4 + sf_6;
    int m4 = sf_0 + sf_1 + sf_3 + sf_4 + sf_7;
    int m5 = sf_0 + sf_1 + sf_3 + sf_4 + sf_8 + sf_9;
    int m6 = sf_0 + sf_1 + sf_3 + sf_4 + sf_8 + sf_10;
    int m7 = sf_0 + sf_1 + sf_3 + sf_4 + sf_8 + sf_11;
    int m8 = sf_0 + sf_1 + sf_3 + sf_4 + sf_12 + sf_13;

    int m9 = sf_25 + sf_24 + sf_23 + sf_22;
    int m10 = sf_25 + sf_24 + sf_23 + sf_21 + sf_20;
    int m11 = sf_25 + sf_24 + sf_23 + sf_21 + sf_15 + sf_16 + sf_19;

    int m12 = sf_25 + sf_24 + sf_23 + sf_21 + sf_15 + sf_16 + sf_17;
    int m13 = sf_25 + sf_24 + sf_23 + sf_21 + sf_15 + sf_16 + sf_18 + sf_26 + sf_27;


    std::string ternaryNumber = std::to_string(m1) + ',' + std::to_string(m2) + ',' + std::to_string(m3) + ',' + std::to_string(m4) + ',' + std::to_string(m5) + ',' + std::to_string(m6) + ',' + std::to_string(m7) + ',' + std::to_string(m8) + ',' + std::to_string(m12) + ',' + std::to_string(m13) + ',' + std::to_string(m11) + ',' + std::to_string(m10) + ',' + std::to_string(m9);

    double num0, num1, num2, num3, num4, num5, num6, num7, num8, num9, num10, num11, num12, num13, num14, num15;
    double num16, num17, num18, num19, num20, num21, num22, num23, num24, num25, num26, num27, num28;

    double flow_0 = 0.2; double flow_1 = 3.1; double flow_2 = 5.5; double flow_3 = 9.1;

    num0 = sf_0 == 0 ? getRandomDouble(flow_0, flow_1) : getRandomDouble(flow_2, flow_3);
    num1 = sf_1 == 0 ? getRandomDouble(flow_0, flow_1) : getRandomDouble(flow_2, flow_3); 
    num6 = sf_6 == 0 ? getRandomDouble(flow_0, flow_1) : getRandomDouble(flow_2, flow_3);
    num7 = sf_7 == 0 ? getRandomDouble(flow_0, flow_1) : getRandomDouble(flow_2, flow_3);
    printf("num1: %f num6: %f num7: %f\n", num1, num6, num7);
    num2 = sf_2 == 0 ? getRandomDouble(flow_0, flow_1) : getRandomDouble(flow_2, flow_3);
    num8 = sf_8 == 0 ? getRandomDouble(flow_0, flow_1) : getRandomDouble(flow_2, flow_3);
    num9 = sf_9 == 0 ? getRandomDouble(flow_0, flow_1) : getRandomDouble(flow_2, flow_3);
    printf("num2: %f num8: %f num9: %f\n", num2, num8, num9);
    num3 = sf_3 == 0 ? getRandomDouble(flow_0, flow_1) : getRandomDouble(flow_2, flow_3);
    num10 = sf_10 == 0 ? getRandomDouble(flow_0, flow_1) : getRandomDouble(flow_2, flow_3);
    num11 = sf_11 == 0 ? getRandomDouble(flow_0, flow_1) : getRandomDouble(flow_2, flow_3);
    printf("num3: %f num10: %f num11: %f\n", num3, num10, num11);
    num4 = sf_4 == 0 ? getRandomDouble(flow_0, flow_1) : getRandomDouble(flow_2, flow_3);
    num12 = sf_12 == 0 ? getRandomDouble(flow_0, flow_1) : getRandomDouble(flow_2, flow_3);
    num13 = sf_13 == 0 ? getRandomDouble(flow_0, flow_1) : getRandomDouble(flow_2, flow_3);
    printf("num4: %f num12: %f num13: %f\n", num4, num12, num13);
    num5 = sf_5 == 0 ? getRandomDouble(flow_0, flow_1) : getRandomDouble(flow_2, flow_3);
    num14 = sf_14 == 0 ? getRandomDouble(flow_0, flow_1) : getRandomDouble(flow_2, flow_3);
    num15 = sf_15 == 0 ? getRandomDouble(flow_0, flow_1) : getRandomDouble(flow_2, flow_3);
    printf("num5: %f num14: %f num15: %f\n", num5, num14, num15);

    num16 = sf_16 == 0 ? getRandomDouble(flow_0, flow_1) : getRandomDouble(flow_2, flow_3);
    num17 = sf_17 == 0 ? getRandomDouble(flow_0, flow_1) : getRandomDouble(flow_2, flow_3);
    num18 = sf_18 == 0 ? getRandomDouble(flow_0, flow_1) : getRandomDouble(flow_2, flow_3);
    num19 = sf_19 == 0 ? getRandomDouble(flow_0, flow_1) : getRandomDouble(flow_2, flow_3);
    num20 = sf_20 == 0 ? getRandomDouble(flow_0, flow_1) : getRandomDouble(flow_2, flow_3);
    num21 = sf_21 == 0 ? getRandomDouble(flow_0, flow_1) : getRandomDouble(flow_2, flow_3);
    num22 = sf_22 == 0 ? getRandomDouble(flow_0, flow_1) : getRandomDouble(flow_2, flow_3);
    num23 = sf_23 == 0 ? getRandomDouble(flow_0, flow_1) : getRandomDouble(flow_2, flow_3);
    num24 = sf_24 == 0 ? getRandomDouble(flow_0, flow_1) : getRandomDouble(flow_2, flow_3);
    num25 = sf_25 == 0 ? getRandomDouble(flow_0, flow_1) : getRandomDouble(flow_2, flow_3);
    num26 = sf_26 == 0 ? getRandomDouble(flow_0, flow_1) : getRandomDouble(flow_2, flow_3);
    num27 = sf_27 == 0 ? getRandomDouble(flow_0, flow_1) : getRandomDouble(flow_2, flow_3);

    num28 = num27 + num26 + num25 + num24 + num23 + num22 + num21 + num20 + num19 + num18 + num17 + num16 + num15 + num14 + num13 + num12 + num11 + num10 + num9 + num8 + num7 + num6 + num5 + num4 + num3 + num2 + num1 + num0;
    printf("num28: %f\n", num28);
  // 创建节点
  NodeContainer nodes;
  nodes.Create (44);
  int number_links = 43;
  std::vector<NodeContainer> nodeLinks(number_links);

  // 逻辑链接
  nodeLinks[0] = NodeContainer(nodes.Get(0), nodes.Get(1));
  nodeLinks[1] = NodeContainer(nodes.Get(1), nodes.Get(2));
  nodeLinks[2] = NodeContainer(nodes.Get(2), nodes.Get(5));
  nodeLinks[3] = NodeContainer(nodes.Get(2), nodes.Get(3));
  dict_interf[0] = 0; dict_interf_2[0] = 0;
  dict_interf[1] = 0; dict_interf_2[1] = 1;
  dict_interf[2] = 1; dict_interf_2[2] = 1;
  dict_interf[5] = 2; dict_interf_2[5] = 1;
  dict_interf[3] = 3; dict_interf_2[3] = 1;
  
  
  nodeLinks[4] = NodeContainer(nodes.Get(3), nodes.Get(6));
  nodeLinks[5] = NodeContainer(nodes.Get(3), nodes.Get(4));
  nodeLinks[6] = NodeContainer(nodes.Get(6), nodes.Get(7));
  nodeLinks[7] = NodeContainer(nodes.Get(6), nodes.Get(8));
  nodeLinks[8] = NodeContainer(nodes.Get(6), nodes.Get(9));
  dict_interf[4] = 5; dict_interf_2[4] = 1;
  dict_interf[6] = 4; dict_interf_2[6] = 1;
  dict_interf[7] = 6; dict_interf_2[7] = 1;
  dict_interf[8] = 7; dict_interf_2[8] = 1;
  

  nodeLinks[9] = NodeContainer(nodes.Get(9), nodes.Get(10));
  nodeLinks[10] = NodeContainer(nodes.Get(9), nodes.Get(11));
  nodeLinks[11] = NodeContainer(nodes.Get(9), nodes.Get(12));
  dict_interf[9] = 8; dict_interf_2[9] = 1;
  dict_interf[10] = 9; dict_interf_2[10] = 1;
  dict_interf[11] = 10; dict_interf_2[11] = 1;
  dict_interf[12] = 11; dict_interf_2[12] = 1;

  nodeLinks[12] = NodeContainer(nodes.Get(6), nodes.Get(13));
  nodeLinks[13] = NodeContainer(nodes.Get(13), nodes.Get(14));
  dict_interf[13] = 12; dict_interf_2[13] = 1;
  dict_interf[14] = 13; dict_interf_2[14] = 1;

  nodeLinks[14] = NodeContainer(nodes.Get(14), nodes.Get(15));
  nodeLinks[15] = NodeContainer(nodes.Get(15), nodes.Get(16));
  nodeLinks[16] = NodeContainer(nodes.Get(16), nodes.Get(17));
  dict_interf[15] = 14; dict_interf_2[15] = 1;
  dict_interf[16] = 15; dict_interf_2[16] = 1;
  dict_interf[17] = 16; dict_interf_2[17] = 1;


  nodeLinks[17] = NodeContainer(nodes.Get(17), nodes.Get(18));
  nodeLinks[18] = NodeContainer(nodes.Get(17), nodes.Get(19));
  nodeLinks[19] = NodeContainer(nodes.Get(17), nodes.Get(20));
  dict_interf[18] = 17; dict_interf_2[18] = 1;
  dict_interf[19] = 18; dict_interf_2[19] = 1;


  nodeLinks[20] = NodeContainer(nodes.Get(15), nodes.Get(21));
  nodeLinks[21] = NodeContainer(nodes.Get(23), nodes.Get(15));
  dict_interf[20] = 19; dict_interf_2[20] = 1;
  dict_interf[21] = 20; dict_interf_2[21] = 1;


  nodeLinks[22] = NodeContainer(nodes.Get(23), nodes.Get(22));
  nodeLinks[23] = NodeContainer(nodes.Get(24), nodes.Get(23));
  dict_interf[22] = 22; dict_interf_2[22] = 1;
  dict_interf[23] = 23; dict_interf_2[23] = 1;

  nodeLinks[24] = NodeContainer(nodes.Get(25), nodes.Get(24));
  nodeLinks[25] = NodeContainer(nodes.Get(26), nodes.Get(25));
  dict_interf[24] = 24; dict_interf_2[24] = 1;
  dict_interf[25] = 25; dict_interf_2[25] = 1;
  dict_interf[26] = 25; dict_interf_2[26] = 0;

  nodeLinks[26] = NodeContainer(nodes.Get(19), nodes.Get(27));
  nodeLinks[27] = NodeContainer(nodes.Get(27), nodes.Get(28));
  dict_interf[27] = 26; dict_interf_2[27] = 1;
  dict_interf[28] = 27; dict_interf_2[28] = 1;


  // install 源节点
  nodeLinks[28] = NodeContainer(nodes.Get(0), nodes.Get(29));
  nodeLinks[29] = NodeContainer(nodes.Get(4), nodes.Get(30));
  nodeLinks[30] = NodeContainer(nodes.Get(5), nodes.Get(31));
  dict_interf[29] = 28; dict_interf_2[29] = 1;
  dict_interf[30] = 29; dict_interf_2[30] = 1;
  dict_interf[31] = 30; dict_interf_2[31] = 1;

  nodeLinks[31] = NodeContainer(nodes.Get(7), nodes.Get(32));
  nodeLinks[32] = NodeContainer(nodes.Get(8), nodes.Get(33));
  nodeLinks[33] = NodeContainer(nodes.Get(10), nodes.Get(34));
  nodeLinks[34] = NodeContainer(nodes.Get(11), nodes.Get(35));
  nodeLinks[35] = NodeContainer(nodes.Get(12), nodes.Get(36));
  dict_interf[32] = 31; dict_interf_2[32] = 1;
  dict_interf[33] = 32; dict_interf_2[33] = 1;
  dict_interf[34] = 33; dict_interf_2[34] = 1;
  dict_interf[35] = 34; dict_interf_2[35] = 1;
  dict_interf[36] = 35; dict_interf_2[36] = 1;
  
  nodeLinks[36] = NodeContainer(nodes.Get(14), nodes.Get(37));
  nodeLinks[37] = NodeContainer(nodes.Get(18), nodes.Get(38));
  nodeLinks[38] = NodeContainer(nodes.Get(28), nodes.Get(39));
  dict_interf[37] = 36; dict_interf_2[37] = 1;
  dict_interf[38] = 37; dict_interf_2[38] = 1;
  dict_interf[39] = 38; dict_interf_2[39] = 1;

  nodeLinks[39] = NodeContainer(nodes.Get(20), nodes.Get(40));
  nodeLinks[40] = NodeContainer(nodes.Get(21), nodes.Get(41));
  nodeLinks[41] = NodeContainer(nodes.Get(22), nodes.Get(42));
  nodeLinks[42] = NodeContainer(nodes.Get(26), nodes.Get(43));
  dict_interf[40] = 39; dict_interf_2[40] = 1;
  dict_interf[41] = 40; dict_interf_2[41] = 1;
  dict_interf[42] = 41; dict_interf_2[42] = 1;
  dict_interf[43] = 42; dict_interf_2[43] = 1;


  // 异构 
  std::cout << "anomoly_link: " << anomoly_link << std::endl;
  setBandwidths(bandwidths, anomoly_link); // 设置2个元素为"10Mbps"
  setDelays(delays, anomoly_link); // 设置2个元素为"45ms"


  // 创建连接
  std::vector<PointToPointHelper> p2p(number_links);
  for (int i = 0; i < number_links; i++)
  {
    p2p[i].SetDeviceAttribute("DataRate", StringValue(bandwidths[i%5]));
    p2p[i].SetChannelAttribute("Delay", StringValue(delays[i%5]));
  }

  std::vector<NetDeviceContainer> devices(number_links);
  for (int i = 0; i < number_links; i++ ){
    devices[i] = p2p[i].Install(nodeLinks[i]);
  }

  // 安装协议栈
  InternetStackHelper stack;
  stack.Install (nodes);

  // 分配 IP 地址
  Ipv4AddressHelper address;
  std::vector<Ipv4InterfaceContainer> interfaces(number_links);
  for (int i = 0; i < number_links; i++ ){
    std::ostringstream oss;
    oss << "10.1." << i+1 << ".0";
    address.SetBase (oss.str().c_str() , "255.255.255.0");
    interfaces[i] = address.Assign (devices[i]);
    //printf("IP address for interface %d: %s\n", i, interfaces[i].GetAddress(0));
  }
  printf("here!\n");
  //建立全局路由表
  Ipv4GlobalRoutingHelper::PopulateRoutingTables ();

  // 背景流
  bck_(nodes, interfaces, 0, 1, 500 + generateRandomNumbers() * generateRandomNumbers(), num0);
  bck_(nodes, interfaces, 1, 2, 500 + generateRandomNumbers() * generateRandomNumbers(), num1);
  bck_(nodes, interfaces, 2, 5, 500 + generateRandomNumbers() * generateRandomNumbers(), num2);
  bck_(nodes, interfaces, 2, 3, 500 + generateRandomNumbers() * generateRandomNumbers(), num3);
  bck_(nodes, interfaces, 3, 4, 500 + generateRandomNumbers() * generateRandomNumbers(), num5);

  bck_(nodes, interfaces, 3, 6, 500 + generateRandomNumbers() * generateRandomNumbers(), num4);
  bck_(nodes, interfaces, 6, 7, 500 + generateRandomNumbers() * generateRandomNumbers(), num6);
  bck_(nodes, interfaces, 6, 8, 500 + generateRandomNumbers() * generateRandomNumbers(), num7);
  bck_(nodes, interfaces, 6, 9, 500 + generateRandomNumbers() * generateRandomNumbers(), num8);

  bck_(nodes, interfaces, 9, 10, 500 + generateRandomNumbers() * generateRandomNumbers(), num9);
  bck_(nodes, interfaces, 9, 11, 500 + generateRandomNumbers() * generateRandomNumbers(), num10);
  bck_(nodes, interfaces, 9, 12, 500 + generateRandomNumbers() * generateRandomNumbers(), num11);

  bck_(nodes, interfaces, 6, 13, 500 + generateRandomNumbers() * generateRandomNumbers(), num12);
  bck_(nodes, interfaces, 13, 14, 500 + generateRandomNumbers() * generateRandomNumbers(), num13);
  // bck_(nodes, interfaces, 13, 15, 500 + generateRandomNumbers() * generateRandomNumbers(), num14);

  bck_(nodes, interfaces, 15, 16, 500 + generateRandomNumbers() * generateRandomNumbers(), num15);
  bck_(nodes, interfaces, 16, 17, 500 + generateRandomNumbers() * generateRandomNumbers(), num16);
  bck_(nodes, interfaces, 17, 18, 500 + generateRandomNumbers() * generateRandomNumbers(), num17);
  bck_(nodes, interfaces, 17, 19, 500 + generateRandomNumbers() * generateRandomNumbers(), num18);

  bck_(nodes, interfaces, 17, 20, 500 + generateRandomNumbers() * generateRandomNumbers(), num19);
  bck_(nodes, interfaces, 15, 21, 500 + generateRandomNumbers() * generateRandomNumbers(), num20);
  bck_(nodes, interfaces, 23, 15, 500 + generateRandomNumbers() * generateRandomNumbers(), num21);

  bck_(nodes, interfaces, 23, 22, 500 + generateRandomNumbers() * generateRandomNumbers(), num22);
  bck_(nodes, interfaces, 24, 23, 500 + generateRandomNumbers() * generateRandomNumbers(), num23);

  bck_(nodes, interfaces, 25, 24, 500 + generateRandomNumbers() * generateRandomNumbers(), num24);
  bck_(nodes, interfaces, 26, 25, 500 + generateRandomNumbers() * generateRandomNumbers(), num25);

  bck_(nodes, interfaces, 19, 27, 500 + generateRandomNumbers() * generateRandomNumbers(), num26);
  bck_(nodes, interfaces, 27, 28, 500 + generateRandomNumbers() * generateRandomNumbers(), num27);

  // 探测流 
  probe_udp_client(nodes, interfaces, 30, 29, probe_rate);
  probe_udp_client(nodes, interfaces, 31, 29, probe_rate);
  probe_udp_client(nodes, interfaces, 32, 29, probe_rate);
  probe_udp_client(nodes, interfaces, 33, 29, probe_rate);
  probe_udp_client(nodes, interfaces, 34, 29, probe_rate);
  probe_udp_client(nodes, interfaces, 35, 29, probe_rate);
  probe_udp_client(nodes, interfaces, 36, 29, probe_rate);
  probe_udp_client(nodes, interfaces, 37, 29, probe_rate);
  probe_udp_client(nodes, interfaces, 38, 43, probe_rate);
  probe_udp_client(nodes, interfaces, 39, 43, probe_rate);
  probe_udp_client(nodes, interfaces, 40, 43, probe_rate);
  probe_udp_client(nodes, interfaces, 41, 43, probe_rate);
  probe_udp_client(nodes, interfaces, 42, 43, probe_rate);


  // 创建流量监控器
  Ptr<FlowMonitor> flowMonitor;
  FlowMonitorHelper flowHelper;
  Simulator::Stop(Seconds(13.0));
  flowHelper.SetMonitorAttribute("PacketSizeBinWidth", DoubleValue(time_int)); // PacketSizeBinWidth -> 文件名称
  flowMonitor = flowHelper.InstallAll();
  flowMonitor -> Start(Seconds(0.0));
  flowMonitor -> Stop(Seconds(10.0));

  Simulator::Run ();
  std::string xml_name = "./txt/test_" + std::to_string(time_int) + ".xml";
  flowMonitor -> SerializeToXmlFile(xml_name, true, true);
  Simulator::Destroy ();

  std::ostringstream oss;
  oss << ternaryNumber;

  std::string file_name = "./txt/flow_monitor" + std::to_string(time_int) + ".txt";
  appendToFile(file_name, oss.str());

  // 使用 stringstream 来构建命令字符串
  std::ostringstream cmd;
  cmd << "python3 analyze.py " << std::to_string(time_int) << " " << 13 << " " << 10 << " " << probe_rate << " " << "Chinanet_6" << " " << anomoly_link;

  // 调用 system 函数执行 Python 脚本
  system(cmd.str().c_str());
  return 0;
}
