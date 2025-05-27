

import numpy as np
from multiprocessing import Pool
import functools
import os

def independentnoise_simulation(code, error_rate, decoders,num_trials,**kwargs):
    import numpy as np
    np.random.seed(0)
    hx = code.hx
    lz = code.lz
    p = error_rate
    N = hx.shape[1]
    for decoder in decoders:
        decoder.set_h(hx,[None],error_rate)
    print("___ Independent Noise Simulation ___")
    error_num = {decoder.name : 0 for decoder in decoders}
    for i in range(num_trials):

        # generate error
        error = np.zeros(N).astype(int)
        for q in range(N):
            if np.random.rand() < p:
                error[q] = 1

        syndrome = hx @ error % 2

        """Decode"""
        for decoder in decoders:
            correction = decoder.decode(syndrome)
            residual_error = (correction + error) % 2
            flag = (lz @ residual_error % 2).any()
            if flag == 1:
                error_num[decoder.name] += 1
                print(f"{decoder.name} {np.nonzero(correction)[0]}, error HW = {np.nonzero(error)[0]}")
    print("logistic qubits", end='    ')
    for lz_arr in lz:
        print(f"{np.nonzero(lz_arr)[0]}", end='    ')
    print()
    print('-'*10+"decoding simulation results"+'-'*10)
    for decoder in decoders:
        logical_error_rate = error_num[decoder.name]*100/num_trials
        print(f"{decoder.name} Decoding error rate: {logical_error_rate:.6f}%")


from time import perf_counter


def measure_noise_simulation(code, error_rate, decoders,num_trials,num_repeat):
    import numpy as np
    np.random.seed(0)
    hx = code.hx
    lz = code.lz
    p = error_rate
    measure_p = error_rate
    N = hx.shape[1]
    print("___ Independent Noise Simulation ___")
    error_num = {decoder.name : 0 for decoder in decoders}
    decodingtime = {decoder.name : [] for decoder in decoders}
    for decoder in decoders:
        mat = np.hstack((hx,np.identity(hx.shape[0])))
        decoder.set_h(mat,[None],error_rate)
    
    for i in range(num_trials):
        true_errors = []
        true_meaurement_errors = []
        
        syndromes = []
        for j in range(num_repeat):
            # generate error
            if j == 0:
                error = np.zeros(N).astype(int)
                for q in range(N):
                    if np.random.rand() < p:
                        error[q] = 1
                
                measurement_error = np.zeros(hx.shape[0]).astype(int)
                for c in range(hx.shape[0]):
                    if np.random.rand() < measure_p:
                        measurement_error[c] = 1
                syndrome = (hx @ error + measurement_error) % 2
            else:
                error = np.zeros(N).astype(int)
                for q in range(N):
                    if np.random.rand() < p:
                        error[q] = 1
                measurement_error = np.zeros(hx.shape[0]).astype(int)
                for c in range(hx.shape[0]):
                    if np.random.rand() < measure_p:
                        measurement_error[c] = 1
                syndrome = (hx @ error + measurement_error) % 2
                syndrome = (syndrome + true_meaurement_errors[-1]) %2
            true_errors.append(error)
            true_meaurement_errors.append(measurement_error)
            syndromes.append(syndrome)
        if np.sum(np.sum(error) for error in true_errors) + np.sum(np.sum(measurement_error) for measurement_error in true_meaurement_errors) == 0:
            continue
        for decoder in decoders:
            pred_e_ms = []
            for j in range(num_repeat):
                """Decode"""
                syndrome = syndromes[j]
                if j == 0:  # not the last round
                    if np.sum(syndrome) == 0:
                        pred_e_ms.append(np.zeros(N+hx.shape[0]))
                        continue
                    correction = decoder.decode(syndrome.astype(int))
                    pred_e_ms.append(correction)
                else:  # the last round
                    new_syndrome = (syndrome + pred_e_ms[-1][N:]) %2
                    if np.sum(new_syndrome) == 0:
                        pred_e_ms.append(np.zeros(N+hx.shape[0]))
                        continue
                    start_time = perf_counter()
                    correction = decoder.decode(new_syndrome.astype(int))
                    end_time = perf_counter()
                    if decoder.name == "BP":
                        decodingtime[decoder.name].append(decoder.decoder.iter*8*1e-9)
                    else:
                        decodingtime[decoder.name].append(end_time - start_time)
                    
                    pred_e_ms.append(correction)
            
            pred_errors = [pred_e_m[:N] for pred_e_m in pred_e_ms]
            # pred_measurement_errors = [pred_e_m[N:] for pred_e_m in pred_e_ms]
            cumulative_error = np.zeros(N).astype(int)
            for pred_error in pred_errors:
                cumulative_error = (cumulative_error + pred_error) % 2
            cumulative_true_error = np.zeros(N).astype(int)
            for true_error in true_errors:
                cumulative_true_error = (cumulative_true_error + true_error) % 2
                    
            residual_error = (cumulative_error + cumulative_true_error) % 2
            # residual_error[int(len(cumulative_error)/2-1)] = 0
            flag = (lz @ residual_error % 2).any()
            if flag:
                error_num[decoder.name] += 1
                print(f"{decoder.name} {np.nonzero(cumulative_error)[0]}, error HW = {np.nonzero(cumulative_true_error)[0]}")
    print("logistic qubits", end='    ')
    for lz_arr in lz:
        print(f"{np.nonzero(lz_arr)[0]}", end='    ')
    print()
    print('-'*10+"decoding simulation results"+'-'*10)
    logical_error_rate = {decoder.name: error_num[decoder.name]/num_trials for decoder in decoders}
    logical_error_rate_per_round = {decoder.name: (1 - (1 - logical_error_rate[decoder.name]) ** (1 / num_repeat)) for decoder in decoders}
    for decoder in decoders:
        print(f"{decoder.name} logical error rate per round: {logical_error_rate_per_round[decoder.name]:.6f}")
        print(f"{decoder.name} logical error rate: {logical_error_rate[decoder.name]:.6f}")
    decoded_time = {decoder.name: np.mean(decodingtime[decoder.name]) for decoder in decoders}
    for decoder in decoders:
        print(f"{decoder.name} decoding time: {decoded_time[decoder.name]:.6f}s")
    return logical_error_rate, logical_error_rate_per_round, decoded_time


def generate_trial_data(num_trials, num_repeat, hx, p, measure_p):
    np.random.seed(0)  # Seed only in main process
    trial_data = []
    N = hx.shape[1]
    
    for i in range(num_trials):
        true_errors = []
        true_measurement_errors = []
        syndromes = []
        
        for j in range(num_repeat):
            # Generate error
            error = np.zeros(N, dtype=int)
            error[np.random.rand(N) < p] = 1
            
            # Generate measurement error
            measurement_error = np.zeros(hx.shape[0], dtype=int)
            measurement_error[np.random.rand(hx.shape[0]) < measure_p] = 1
            
            # Calculate syndrome
            if j == 0:
                syndrome = (hx @ error + measurement_error) % 2
            else:
                syndrome = (hx @ error + measurement_error + true_measurement_errors[-1]) % 2
            
            true_errors.append(error)
            true_measurement_errors.append(measurement_error)
            syndromes.append(syndrome)
            
        if np.sum(np.sum(error) for error in true_errors) + np.sum(np.sum(me) for me in true_measurement_errors) != 0: 
            trial_data.append({
                'true_errors': true_errors,
                'true_measurement_errors': true_measurement_errors,
                'syndromes': syndromes
            })
    print(f"Generated {len(trial_data)} trials data")
    return trial_data

def process_trial(args, decoders, hx, lz):
    trial_idx, trial = args
    N = hx.shape[1]
    true_errors = trial['true_errors']
    syndromes = trial['syndromes']
    num_repeat = len(true_errors)
    
    trial_results = {}
    for decoder in decoders:
        pred_e_ms = []
        for j in range(num_repeat):
            syndrome = syndromes[j]
            if j == 0:
                if np.sum(syndrome) == 0:
                    pred_e_ms.append(np.zeros(N + hx.shape[0], dtype=int))
                    continue
                correction = decoder.decode(syndrome.astype(int))
                pred_e_ms.append(correction)
            else:
                new_syndrome = (syndrome + pred_e_ms[-1][N:]) % 2
                if np.sum(new_syndrome) == 0:
                    pred_e_ms.append(np.zeros(N + hx.shape[0], dtype=int))
                    continue
                correction = decoder.decode(new_syndrome.astype(int))
                pred_e_ms.append(correction)
        
        pred_errors = [pred_e_m[:N] for pred_e_m in pred_e_ms]
        cumulative_error = np.zeros(N, dtype=int)
        for pred_error in pred_errors:
            cumulative_error = (cumulative_error + pred_error) % 2
        
        cumulative_true_error = np.zeros(N, dtype=int)
        for true_error in true_errors:
            cumulative_true_error = (cumulative_true_error + true_error) % 2
        
        residual_error = (cumulative_error + cumulative_true_error) % 2
        residual_error[int(len(cumulative_error)/2-1)] = 0
        flag = (lz @ residual_error % 2).any()
        trial_results[decoder.name] = 1 if flag else 0
        
        if flag:
            print(f"Trial {trial_idx}: {decoder.name}\n {np.nonzero(cumulative_error)[0]}, true error =\n {np.nonzero(cumulative_true_error)[0]}")
    
    return trial_results

def measure_noise_simulation_parallel(code, error_rate, decoders, num_trials, num_repeat, num_processes=None):
    hx = code.hx
    lz = code.lz
    p = error_rate
    measure_p = error_rate
    
    print("___ Independent Noise Simulation (Parallel) ___")
    
    # Initialize decoders
    for decoder in decoders:
        mat = np.hstack((hx, np.identity(hx.shape[0])))
        decoder.set_h(mat, [None], error_rate)
    
    # Generate all trial data in main process first
    trial_data = generate_trial_data(num_trials, num_repeat, hx, p, measure_p)
    
    # Prepare arguments for parallel processing
    args = [(i, trial_data[i]) for i in range(len(trial_data))]
    
    # Create partial function with fixed parameters
    partial_func = functools.partial(process_trial, 
                                   decoders=decoders,
                                   hx=hx,
                                   lz=lz)
    
    # Use multiprocessing Pool
    if num_processes is None:
        num_processes = os.cpu_count()
    with Pool(processes=num_processes) as pool:
        results = pool.map(partial_func, args)
    
    # Aggregate results
    error_num = {decoder.name: 0 for decoder in decoders}
    for trial_result in results:
        for decoder_name, error_count in trial_result.items():
            error_num[decoder_name] += error_count
    
    # Print logical qubits information
    print("logical qubits", end='    ')
    for lz_arr in lz:
        print(f"{np.nonzero(lz_arr)[0]}", end='    ')
    print()
    
    # Calculate and print results
    print('-'*10 + "decoding simulation results" + '-'*10)
    logical_error_rate = {decoder.name: error_num[decoder.name]/num_trials for decoder in decoders}
    logical_error_rate_per_round = {decoder.name: (1 - (1 - logical_error_rate[decoder.name]) ** (1 / num_repeat)) for decoder in decoders}
    
    for decoder in decoders:
        print(f"{decoder.name} logical error rate per round: {logical_error_rate_per_round[decoder.name]:.6f}")
        print(f"{decoder.name} logical error rate: {logical_error_rate[decoder.name]:.6f}")
    
    return [logical_error_rate, logical_error_rate_per_round], trial_data



def measure_noise_simulation_by_trial_data(code, error_rate, decoders,num_trials, num_repeat, trial_data):
    hx = code.hx
    lz = code.lz
    
    print("___ Independent Noise Simulation (Parallel) ___")
    
    # Initialize decoders
    for decoder in decoders:
        mat = np.hstack((hx, np.identity(hx.shape[0])))
        decoder.set_h(mat, [None], error_rate)
    
    # Generate all trial data in main process first
    # trial_data = generate_trial_data(num_trials, num_repeat, hx, p, measure_p)
    
    # Prepare arguments for parallel processing
    args = [(i, trial_data[i]) for i in range(len(trial_data))]
    
    # Create partial function with fixed parameters
    partial_func = functools.partial(process_trial, 
                                   decoders=decoders,
                                   hx=hx,
                                   lz=lz)
    
    results = map(partial_func, args)
    # Aggregate results
    error_num = {decoder.name: 0 for decoder in decoders}
    for trial_result in results:
        for decoder_name, error_count in trial_result.items():
            error_num[decoder_name] += error_count
    
    # Print logical qubits information
    print("logical qubits", end='    ')
    for lz_arr in lz:
        print(f"{np.nonzero(lz_arr)[0]}", end='    ')
    print()
    
    # Calculate and print results
    print('-'*10 + "decoding simulation results" + '-'*10)
    logical_error_rate = {decoder.name: error_num[decoder.name]/num_trials for decoder in decoders}
    logical_error_rate_per_round = {decoder.name: (1 - (1 - logical_error_rate[decoder.name]) ** (1 / num_repeat)) for decoder in decoders}
    
    for decoder in decoders:
        print(f"{decoder.name} logical error rate per round: {logical_error_rate_per_round[decoder.name]:.6f}")
        print(f"{decoder.name} logical error rate: {logical_error_rate[decoder.name]:.6f}")
    
    return [logical_error_rate, logical_error_rate_per_round]