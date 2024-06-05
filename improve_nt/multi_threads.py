from multiprocessing import Process, current_process
import time
from evaluate import test



def process_function(name, tp_name, avg_prob, noise_):
    print(f"Process {name}: {tp_name}, {avg_prob}, {noise_}")
    test(tp_name, avg_prob, name, noise_, ternary_flag=False)
    print(f"Process {name}: finishing")

if __name__ == "__main__":
    # 进程列表
    processes = []
    # 最大进程数
    max_processes = 20
    # 总共需要执行的次数
    tp_names = ['Chinanet', 'Agis', 'Japan', 'Canada']
    
    # 外部传入的概率参数
    probs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # probs = [0.02, 0.05, 0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]
    N = 20

    active_processes = []
    for tn in range(len(tp_names)):
        for prob in range(len(probs)):
            for v in range(N):
                index = tn * len(probs) * N + prob * N + v
                p = Process(target=process_function, args=(index, tp_names[tn], probs[prob], 0))
                processes.append(p)
                p.start()
                active_processes.append(p)
                
                # 当活跃的进程数达到最大值时，等待任一进程完成
                if len(active_processes) >= max_processes:
                    p = active_processes.pop(0)
                    p.join()

    # 确保所有进程都已完成
    for p in active_processes:
        p.join()

    print("All processes have finished.")