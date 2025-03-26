# import qldpcdecoder
# from qldpcdecoder.codes import gen_BB_code, gen_HP_ring_code
# from qldpcdecoder.bpdecoders import BPOSD_decoder, BP_decoder
# from qldpcdecoder.simulation.independentsim import measure_noise_simulation,independentnoise_simulation
# from qldpcdecoder.simulation.bbcodesim import circuit_level_simulation
# from qldpcdecoder.sparsegauss_decoder import guass_decoder
# from qldpcdecoder.sparsedecoupleddecoder import DecoupledDecoder
# from functools import reduce
# import numpy as np
# from rich.pretty import pprint
# np.random.seed(12561)


# p_list = [0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001, 0.002, 0.003, 0.004, 0.005]
# # p_list = [0.001]
# data_qubits_num = 288
# css_code = gen_BB_code(data_qubits_num)
# # css_code = gen_HP_ring_code(7,7)


# for p in p_list:
  
#     bposddecoder = BPOSD_decoder()
#     bpdecoder = BP_decoder()
#     gaussdecoder = guass_decoder() # [H, I] no decouple, hybrid
#     reshapeddecoder = DecoupledDecoder(css_code,p) # [H, I] decouple, hybrid

#     if p in [0.0005, 0.0006, 0.0007, 0.0008, 0.0009]:
#         ler, ler_per_round = measure_noise_simulation(css_code,p,[bposddecoder,bpdecoder,gaussdecoder,reshapeddecoder],num_trials=100000,num_repeat=6)
#     if p in [0.001, 0.002, 0.003, 0.004, 0.005]:
#         ler, ler_per_round = measure_noise_simulation(css_code,p,[bposddecoder,bpdecoder,gaussdecoder,reshapeddecoder],num_trials=10000,num_repeat=6)

#     filename = f"LER/BBcode_{data_qubits_num}/dataq_{data_qubits_num}_p{p}.txt"

#     with open(filename, "w") as f:
#         f.write('ler ' + str(ler) + '\n')
#         f.write('ler_per_round ' + str(ler_per_round))
import qldpcdecoder
from qldpcdecoder.codes import gen_BB_code
from qldpcdecoder.bpdecoders import BPOSD_decoder, BP_decoder
from qldpcdecoder.simulation.independentsim import measure_noise_simulation
from qldpcdecoder.sparsegauss_decoder import guass_decoder
from qldpcdecoder.sparsedecoupleddecoder import DecoupledDecoder
import numpy as np
import ray

np.random.seed(12561)

# 初始化Ray
ray.init()

@ray.remote
def run_simulation_chunk(data_qubits_num, p, decoder_classes, decoder_args_list, num_trials_chunk, num_repeat):
    # 生成BB码
    code = gen_BB_code(data_qubits_num)
    
    # 实例化解码器
    decoders = []
    for cls, args in zip(decoder_classes, decoder_args_list):
        if cls == DecoupledDecoder:
            # DecoupledDecoder需要code和p参数
            decoder = cls(code=code, p=p)
        else:
            decoder = cls(**args) if args else cls()
        decoders.append(decoder)
    
    # 运行噪声模拟
    ler, ler_per_round = measure_noise_simulation(code, p, decoders, num_trials=num_trials_chunk, num_repeat=num_repeat)
    return (ler, ler_per_round)

p_list = [0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001, 0.002, 0.003, 0.004, 0.005]
data_qubits_num = 288

for p in p_list:
    # 配置解码器类和参数
    decoder_classes = [BPOSD_decoder, BP_decoder, guass_decoder, DecoupledDecoder]
    decoder_args_list = [
        {},  # BPOSD_decoder参数
        {},  # BP_decoder参数
        {},  # guass_decoder参数
        {}   # DecoupledDecoder参数会在远程函数中动态设置
    ]
    
    # 设置总试验次数
    if p in [0.0005, 0.0006, 0.0007, 0.0008, 0.0009]:
        num_trials_total = 100000
        num_repeat = 6
    else:
        num_trials_total = 10000
        num_repeat = 6
    
    # 分块参数
    num_chunks = 64  # 根据CPU核心数调整
    chunk_sizes = [num_trials_total // num_chunks] * num_chunks
    remainder = num_trials_total % num_chunks
    for i in range(remainder):
        chunk_sizes[i] += 1
    
    # 提交并行任务
    result_refs = []
    for size in chunk_sizes:
        if size > 0:
            result_refs.append(
                run_simulation_chunk.remote(
                    data_qubits_num, p, decoder_classes, decoder_args_list, size, num_repeat
                )
            )
    
    # 获取结果并合并
    chunk_results = ray.get(result_refs)
    
    # 计算总LER
    total_trials = sum(chunk_sizes)
    total_failures = sum(ler * size for (ler, _), size in zip(chunk_results, chunk_sizes))
    ler = total_failures / total_trials if total_trials > 0 else 0.0
    
    # 计算每轮LER
    ler_per_round = []
    if chunk_results:
        num_rounds = len(chunk_results[0][1])
        total_errors = [0.0] * num_rounds
        for (_, lpr), size in zip(chunk_results, chunk_sizes):
            for r in range(num_rounds):
                total_errors[r] += lpr[r] * size
        ler_per_round = [errors / total_trials for errors in total_errors]
    
    # 保存结果
    filename = f"LER/BBcode_{data_qubits_num}/dataq_{data_qubits_num}_p{p}.txt"
    with open(filename, "w") as f:
        f.write(f'ler {ler}\n')
        f.write(f'ler_per_round {ler_per_round}\n')

# 关闭Ray
ray.shutdown()