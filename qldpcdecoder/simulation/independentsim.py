

def independentnoise_simulation(code, error_rate, decoders,num_trials):
    import numpy as np
    np.random.seed(0)
    hx = code.hx
    lz = code.lz
    p = error_rate
    N = hx.shape[1]
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

