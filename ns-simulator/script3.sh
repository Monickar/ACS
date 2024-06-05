#!/bin/bash

# 检查是否传递了足够的参数
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <start value> <end value> <anomoly_link>"
    exit 1
fi

# 从命令行读取参数
start_value=$1
end_value=$2
anomoly_link=$3
np_name=$4

# 对于probe_rate值从start_value到end_value（包括小数值）
for probe_rate in $(seq $start_value 0.5 $end_value); do
    echo "Current probe_rate: $probe_rate"

    for ((i=1; i<=1000; i++)); do
        echo "Running command iteration $i for probe_rate $probe_rate "

        # 启动每个命令作为后台进程
        ./waf --run "$np_name --myParam=0.9 --probe_v=$probe_rate --anomoly_link=$anomoly_link" &
        sleep 2
        ./waf --run "$np_name --myParam=0.5 --probe_v=$probe_rate --anomoly_link=$anomoly_link" &
        sleep 2
        ./waf --run "$np_name --myParam=0.1 --probe_v=$probe_rate --anomoly_link=$anomoly_link" &

        # 等待所有后台进程完成
        wait
    done

done

echo "All iterations completed."
