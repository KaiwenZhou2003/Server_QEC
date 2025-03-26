# import qldpcdecoder
# from qldpcdecoder.codes import gen_BB_code
# from qldpcdecoder.bpdecoders import BPOSD_decoder, BP_decoder
# from qldpcdecoder.simulation.independentsim import measure_noise_simulation
# from qldpcdecoder.sparsegauss_decoder import guass_decoder
# from qldpcdecoder.sparsedecoupleddecoder import DecoupledDecoder
# from multiprocessing import Pool, cpu_count
# import numpy as np

# np.random.seed(12561)

# # p_list = [0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001, 0.002, 0.003, 0.004, 0.005]
# p_list = [0.0009, ]
# data_qubits_num = 144
# css_code = gen_BB_code(data_qubits_num)

# def run_simulation(args):
#     p, num_trials = args
#     bposddecoder = BPOSD_decoder()
#     bpdecoder = BP_decoder()
#     gaussdecoder = guass_decoder()
#     reshapeddecoder = DecoupledDecoder(css_code, p)
    
#     ler, ler_per_round = measure_noise_simulation(
#         css_code, p, [bposddecoder, bpdecoder, gaussdecoder, reshapeddecoder],
#         num_trials=num_trials, num_repeat=6
#     )
    
#     filename = f"LER/BBcode_{data_qubits_num}_d6/dataq_{data_qubits_num}_p{p}.txt"
#     with open(filename, "w") as f:
#         f.write('ler ' + str(ler) + '\n')
#         f.write('ler_per_round ' + str(ler_per_round))

# if __name__ == "__main__":
#     num_cores = cpu_count()
#     print(num_cores)
#     pool = Pool(processes=num_cores)
    
#     tasks = []
#     for p in p_list:
#         if p in [0.0005, 0.0006, 0.0007, 0.0008, 0.0009]:
#             total_trials = 100000
#         else:
#             total_trials = 10000
        
#         num_threads = num_cores  # 限制最大并行线程数，避免资源争夺
#         trials_per_thread = total_trials // num_threads
#         tasks.extend([(p, trials_per_thread) for _ in range(num_threads)])
    
#     pool.map(run_simulation, tasks)
#     pool.close()
#     pool.join()
import qldpcdecoder
from qldpcdecoder.codes import gen_BB_code
from qldpcdecoder.bpdecoders import BPOSD_decoder, BP_decoder
from qldpcdecoder.simulation.independentsim import measure_noise_simulation
from qldpcdecoder.sparsegauss_decoder import guass_decoder
from qldpcdecoder.sparsedecoupleddecoder import DecoupledDecoder
from multiprocessing import Pool, cpu_count
import numpy as np

# 固定全局随机种子
np.random.seed(12561)

# 这里设置噪声参数列表，演示时仅取一个噪声参数
# p_list = [0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001, 0.002, 0.003, 0.004, 0.005]
p_list = [0.0005, ]
data_qubits_num = 144
css_code = gen_BB_code(data_qubits_num)

def run_simulation(args):
    """
    每个进程运行部分仿真，返回结果包括：
      - ler: 字典，存储各个解码器的逻辑错误率
      - ler_per_round: 字典，存储各个解码器各轮逻辑错误率（可能为列表、数组或单个数值）
      - num_trials: 本次部分仿真的试验次数（用于加权聚合）
    """
    p, num_trials = args
    # 为每个进程设置不同的随机种子，避免各进程间随机数重复
    np.random.seed()
    
    # 初始化各个解码器
    bposddecoder = BPOSD_decoder()
    bpdecoder = BP_decoder()
    gaussdecoder = guass_decoder()
    reshapeddecoder = DecoupledDecoder(css_code, p)
    
    # 调用仿真函数
    ler, ler_per_round = measure_noise_simulation(
        css_code, p, [bposddecoder, bpdecoder, gaussdecoder, reshapeddecoder],
        num_trials=num_trials, num_repeat=6
    )
    return (ler, ler_per_round, num_trials)

if __name__ == "__main__":
    num_cores = cpu_count()
    print(f"使用的CPU核心数：{num_cores}")
    pool = Pool(processes=num_cores)
    
    tasks = []
    # 对于每个噪声参数 p，根据 p 的取值确定总仿真次数，然后均分给各进程
    for p in p_list:
        if p in [0.0005, 0.0006, 0.0007, 0.0008, 0.0009]:
            total_trials = 1000000
        else:
            total_trials = 10000
        
        num_threads = num_cores  # 使用所有可用核心
        trials_per_thread = total_trials // num_threads
        tasks.extend([(p, trials_per_thread) for _ in range(num_threads)])
    
    # 并行执行任务，并收集各个进程返回的部分结果
    results = pool.map(run_simulation, tasks)
    pool.close()
    pool.join()
    
    # 对于同一噪声参数 p 的所有部分结果进行聚合（加权求和）
    final_results = {}
    for p in p_list:
        total_trials_sum = 0
        weighted_ler = {}          # 存储各解码器的加权累加值
        weighted_ler_per_round = {}  # 存储各解码器各轮逻辑错误率的加权累加值
        
        for res in results:
            ler, ler_per_round, trials = res
            total_trials_sum += trials
            # 对逻辑错误率字典进行累加
            for key, value in ler.items():
                weighted_ler[key] = weighted_ler.get(key, 0) + value * trials
            # 对每轮逻辑错误率进行累加
            for key, value in ler_per_round.items():
                # 判断value是否为可迭代对象（如列表或元组）
                if hasattr(value, '__iter__'):
                    if key not in weighted_ler_per_round:
                        weighted_ler_per_round[key] = [v * trials for v in value]
                    else:
                        weighted_ler_per_round[key] = [
                            prev + curr * trials for prev, curr in zip(weighted_ler_per_round[key], value)
                        ]
                else:
                    # 如果value为单个数字
                    weighted_ler_per_round[key] = weighted_ler_per_round.get(key, 0) + value * trials
        
        final_ler = {key: weighted_ler[key] / total_trials_sum for key in weighted_ler}
        # 对于每轮错误率，根据value是否为列表分别处理
        final_ler_per_round = {}
        for key, value in weighted_ler_per_round.items():
            if hasattr(value, '__iter__'):
                final_ler_per_round[key] = [v / total_trials_sum for v in value]
            else:
                final_ler_per_round[key] = value / total_trials_sum
        
        final_results[p] = (final_ler, final_ler_per_round)
    
    # 将聚合后的结果写入文件
    for p, (ler, ler_per_round) in final_results.items():
        filename = f"LER/BBcode_{data_qubits_num}_d6/dataq_{data_qubits_num}_p{p}.txt"
        with open(filename, "w") as f:
            f.write('Aggregated ler: ' + str(ler) + '\n')
            f.write('Aggregated ler_per_round: ' + str(ler_per_round))
        print(f"结果已写入文件: {filename}")
