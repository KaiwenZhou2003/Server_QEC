

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
    for decoder in decoders:
        mat = np.hstack((hx,np.identity(hx.shape[0])))
        decoder.set_h(mat,[None],error_rate)
        for i in range(num_trials):
            true_errors = []
            true_meaurement_errors = []
            pred_e_ms = []
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
                    true_errors.append(error)
                    true_meaurement_errors.append(measurement_error)
                # elif j == num_repeat-1:
                #     error = np.zeros(N).astype(int)
                #     for q in range(N):
                #         if np.random.rand() < p:
                #             error[q] = 1
                #     syndrome = (hx @ error) % 2
                #     syndrome = (syndrome + true_meaurement_errors[-1]) %2
                #     true_errors.append(error)
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
                    
                """Decode"""
                
                if j == 0:  # not the last round
                    correction = decoder.decode(syndrome.astype(int))
                    pred_e_ms.append(correction)
                else:  # the last round
                    new_syndrome = (syndrome + pred_e_ms[-1][N:]) %2
                    correction = decoder.decode(new_syndrome.astype(int))
                    pred_e_ms.append(correction)
            
            pred_errors = [pred_e_m[:N] for pred_e_m in pred_e_ms]
            pred_measurement_errors = [pred_e_m[N:] for pred_e_m in pred_e_ms]
            cumulative_error = np.zeros(N).astype(int)
            for pred_error in pred_errors:
                cumulative_error = (cumulative_error + pred_error) % 2
            cumulative_true_error = np.zeros(N).astype(int)
            for true_error in true_errors:
                cumulative_true_error = (cumulative_true_error + true_error) % 2
            # if 71 in np.nonzero(cumulative_error)[0] and decoder.name == "Gauss":
            #     cumulative_error[71] = 0
            residual_error = (cumulative_error + cumulative_true_error) % 2
            flag = (lz @ residual_error % 2).any()
            if flag == 1:
                error_num[decoder.name] += 1
                print(f"{decoder.name} {np.nonzero(cumulative_error)[0]}, error HW = {np.nonzero(cumulative_true_error)[0]}")
    print("logistic qubits", end='    ')
    for lz_arr in lz:
        print(f"{np.nonzero(lz_arr)[0]}", end='    ')
    print()
    print('-'*10+"decoding simulation results"+'-'*10)
    logical_error_rate = {decoder.name: error_num[decoder.name]*100/num_trials for decoder in decoders}
    for decoder in decoders:
        print(f"{decoder.name} Decoding error rate: {logical_error_rate[decoder.name]:.6f}%")
    
    return logical_error_rate