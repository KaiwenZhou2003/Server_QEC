import stim
import numpy as np
from scipy.sparse import csc_matrix
from typing import List, FrozenSet, Dict
import numpy as np
import math
import matplotlib.pyplot as plt
import time
def build_circuit(code, A_list, B_list, p, num_repeat, z_basis=True, use_both=False, HZH=False):

    n = code.N
    a1, a2, a3 = A_list
    b1, b2, b3 = B_list

    def nnz(m):
        a, b = m.nonzero()
        return b[np.argsort(a)]

    A1, A2, A3 = nnz(a1), nnz(a2), nnz(a3)
    B1, B2, B3 = nnz(b1), nnz(b2), nnz(b3)

    A1_T, A2_T, A3_T = nnz(a1.T), nnz(a2.T), nnz(a3.T)
    B1_T, B2_T, B3_T = nnz(b1.T), nnz(b2.T), nnz(b3.T)

    # |+> ancilla: 0 ~ n/2-1. Control in CNOTs.
    X_check_offset = 0
    # L data qubits: n/2 ~ n-1. 
    L_data_offset = n//2
    # R data qubits: n ~ 3n/2-1.
    R_data_offset = n
    # |0> ancilla: 3n/2 ~ 2n-1. Target in CNOTs.
    Z_check_offset = 3*n//2

    p_after_clifford_depolarization = p
    p_after_reset_flip_probability = p
    p_before_measure_flip_probability = p
    p_before_round_data_depolarization = p

    detector_circuit_str = ""
    for i in range(n//2):
        detector_circuit_str += f"DETECTOR rec[{-n//2+i}]\n"
    detector_circuit = stim.Circuit(detector_circuit_str)

    detector_repeat_circuit_str = ""
    for i in range(n//2):
        detector_repeat_circuit_str += f"DETECTOR rec[{-n//2+i}] rec[{-n-n//2+i}]\n"
    detector_repeat_circuit = stim.Circuit(detector_repeat_circuit_str)

    def append_blocks(circuit, repeat=False):
        # Round 1
        if repeat:        
            for i in range(n//2):
                # measurement preparation errors
                circuit.append("X_ERROR", Z_check_offset + i, p_after_reset_flip_probability)
                if HZH:
                    circuit.append("X_ERROR", X_check_offset + i, p_after_reset_flip_probability)
                    circuit.append("H", [X_check_offset + i])
                    circuit.append("DEPOLARIZE1", X_check_offset + i, p_after_clifford_depolarization)
                else:
                    circuit.append("Z_ERROR", X_check_offset + i, p_after_reset_flip_probability)
                # identity gate on R data
                circuit.append("DEPOLARIZE1", R_data_offset + i, p_before_round_data_depolarization)
        else:
            for i in range(n//2):
                circuit.append("H", [X_check_offset + i])
                if HZH:
                    circuit.append("DEPOLARIZE1", X_check_offset + i, p_after_clifford_depolarization)

        for i in range(n//2):
            # CNOTs from R data to to Z-checks
            circuit.append("CNOT", [R_data_offset + A1_T[i], Z_check_offset + i])
            circuit.append("DEPOLARIZE2", [R_data_offset + A1_T[i], Z_check_offset + i], p_after_clifford_depolarization)
            # identity gate on L data
            circuit.append("DEPOLARIZE1", L_data_offset + i, p_before_round_data_depolarization)

        # tick
        circuit.append("TICK")

        # Round 2
        for i in range(n//2):
            # CNOTs from X-checks to L data
            circuit.append("CNOT", [X_check_offset + i, L_data_offset + A2[i]])
            circuit.append("DEPOLARIZE2", [X_check_offset + i, L_data_offset + A2[i]], p_after_clifford_depolarization)
            # CNOTs from R data to Z-checks
            circuit.append("CNOT", [R_data_offset + A3_T[i], Z_check_offset + i])
            circuit.append("DEPOLARIZE2", [R_data_offset + A3_T[i], Z_check_offset + i], p_after_clifford_depolarization)

        # tick
        circuit.append("TICK")

        # Round 3
        for i in range(n//2):
            # CNOTs from X-checks to R data
            circuit.append("CNOT", [X_check_offset + i, R_data_offset + B2[i]])
            circuit.append("DEPOLARIZE2", [X_check_offset + i, R_data_offset + B2[i]], p_after_clifford_depolarization)
            # CNOTs from L data to Z-checks
            circuit.append("CNOT", [L_data_offset + B1_T[i], Z_check_offset + i])
            circuit.append("DEPOLARIZE2", [L_data_offset + B1_T[i], Z_check_offset + i], p_after_clifford_depolarization)

        # tick
        circuit.append("TICK")

        # Round 4
        for i in range(n//2):
            # CNOTs from X-checks to R data
            circuit.append("CNOT", [X_check_offset + i, R_data_offset + B1[i]])
            circuit.append("DEPOLARIZE2", [X_check_offset + i, R_data_offset + B1[i]], p_after_clifford_depolarization)
            # CNOTs from L data to Z-checks
            circuit.append("CNOT", [L_data_offset + B2_T[i], Z_check_offset + i])
            circuit.append("DEPOLARIZE2", [L_data_offset + B2_T[i], Z_check_offset + i], p_after_clifford_depolarization)

        # tick
        circuit.append("TICK")

        # Round 5
        for i in range(n//2):
            # CNOTs from X-checks to R data
            circuit.append("CNOT", [X_check_offset + i, R_data_offset + B3[i]])
            circuit.append("DEPOLARIZE2", [X_check_offset + i, R_data_offset + B3[i]], p_after_clifford_depolarization)
            # CNOTs from L data to Z-checks
            circuit.append("CNOT", [L_data_offset + B3_T[i], Z_check_offset + i])
            circuit.append("DEPOLARIZE2", [L_data_offset + B3_T[i], Z_check_offset + i], p_after_clifford_depolarization)

        # tick
        circuit.append("TICK")

        # Round 6
        for i in range(n//2):
            # CNOTs from X-checks to L data
            circuit.append("CNOT", [X_check_offset + i, L_data_offset + A1[i]])
            circuit.append("DEPOLARIZE2", [X_check_offset + i, L_data_offset + A1[i]], p_after_clifford_depolarization)
            # CNOTs from R data to Z-checks
            circuit.append("CNOT", [R_data_offset + A2_T[i], Z_check_offset + i])
            circuit.append("DEPOLARIZE2", [R_data_offset + A2_T[i], Z_check_offset + i], p_after_clifford_depolarization)

        # tick
        circuit.append("TICK")

        # Round 7
        for i in range(n//2):
            # CNOTs from X-checks to L data
            circuit.append("CNOT", [X_check_offset + i, L_data_offset + A3[i]])
            circuit.append("DEPOLARIZE2", [X_check_offset + i, L_data_offset + A3[i]], p_after_clifford_depolarization)
            # Measure Z-checks
            circuit.append("X_ERROR", Z_check_offset + i, p_before_measure_flip_probability)
            circuit.append("MR", [Z_check_offset + i])
            # identity gates on R data, moved to beginning of the round
            # circuit.append("DEPOLARIZE1", R_data_offset + i, p_before_round_data_depolarization)
        
        # Z check detectors
        if z_basis:
            if repeat:
                circuit += detector_repeat_circuit
            else:
                circuit += detector_circuit
        elif use_both and repeat:
            circuit += detector_repeat_circuit

        # tick
        circuit.append("TICK")
        
        # Round 8
        for i in range(n//2):
            if HZH:
                circuit.append("H", [X_check_offset + i])
                circuit.append("DEPOLARIZE1", X_check_offset + i, p_after_clifford_depolarization)
                circuit.append("X_ERROR", X_check_offset + i, p_before_measure_flip_probability)
                circuit.append("MR", [X_check_offset + i])
            else:
                circuit.append("Z_ERROR", X_check_offset + i, p_before_measure_flip_probability)
                circuit.append("MRX", [X_check_offset + i])
            # identity gates on L data, moved to beginning of the round
            # circuit.append("DEPOLARIZE1", L_data_offset + i, p_before_round_data_depolarization)
            
        # X basis detector
        if not z_basis:
            if repeat:
                circuit += detector_repeat_circuit
            else:
                circuit += detector_circuit
        elif use_both and repeat:
            circuit += detector_repeat_circuit
        
        # tick
        circuit.append("TICK")

   
    circuit = stim.Circuit()
    for i in range(n//2): # ancilla initialization
        circuit.append("R", X_check_offset + i)
        circuit.append("R", Z_check_offset + i)
        circuit.append("X_ERROR", X_check_offset + i, p_after_reset_flip_probability)
        circuit.append("X_ERROR", Z_check_offset + i, p_after_reset_flip_probability)
    for i in range(n):
        circuit.append("R" if z_basis else "RX", L_data_offset + i)
        circuit.append("X_ERROR" if z_basis else "Z_ERROR", L_data_offset + i, p_after_reset_flip_probability)

    # begin round tick
    circuit.append("TICK") 
    append_blocks(circuit, repeat=False) # encoding round


    rep_circuit = stim.Circuit()
    append_blocks(rep_circuit, repeat=True)
    circuit += (num_repeat-1) * rep_circuit

    for i in range(0, n):
        # flip before collapsing data qubits
        # circuit.append("X_ERROR" if z_basis else "Z_ERROR", L_data_offset + i, p_before_measure_flip_probability)
        circuit.append("M" if z_basis else "MX", L_data_offset + i)
    
    if z_basis:
        for i in range(0, n//2):
            circuit.append("X_ERROR", Z_check_offset + i, p_before_measure_flip_probability)
            circuit.append("MR", Z_check_offset + i)
    else:
        for i in range(0, n//2):
            circuit.append("Z_ERROR", X_check_offset + i, p_before_measure_flip_probability)
            circuit.append("MRX", X_check_offset + i)
    
    pcm = code.hz if z_basis else code.hx
    logical_pcm = code.lz if z_basis else code.lx
    stab_detector_circuit_str = "" # stabilizers
    for i, s in enumerate(pcm):
        nnz = np.nonzero(s)[0]
        det_str = "DETECTOR"
        det_str += f" rec[{-n//2+i}]"
        for ind in nnz:
            det_str += f" rec[{-n-n//2+ind}]"       
        det_str += f" rec[{-n-n-n//2+i}]" if z_basis else f" rec[{-n-n//2-n//2+i}]"
        det_str += "\n"
        stab_detector_circuit_str += det_str
    stab_detector_circuit = stim.Circuit(stab_detector_circuit_str)
    circuit += stab_detector_circuit
        
    log_detector_circuit_str = "" # logical operators
    for i, l in enumerate(logical_pcm):
        nnz = np.nonzero(l)[0]
        det_str = f"OBSERVABLE_INCLUDE({i})"
        for ind in nnz:
            det_str += f" rec[{-n-n//2+ind}]"        
        det_str += "\n"
        log_detector_circuit_str += det_str
    log_detector_circuit = stim.Circuit(log_detector_circuit_str)
    circuit += log_detector_circuit

    return circuit

def dict_to_csc_matrix(elements_dict, shape):
    # Constructs a `scipy.sparse.csc_matrix` check matrix from a dictionary `elements_dict` 
    # giving the indices of nonzero rows in each column.
    nnz = sum(len(v) for k, v in elements_dict.items())
    data = np.ones(nnz, dtype=np.uint8)
    row_ind = np.zeros(nnz, dtype=np.int64)
    col_ind = np.zeros(nnz, dtype=np.int64)
    i = 0
    for col, v in elements_dict.items():
        for row in v:
            row_ind[i] = row
            col_ind[i] = col
            i += 1
    return csc_matrix((data, (row_ind, col_ind)), shape=shape)

def dem_to_check_matrices(dem: stim.DetectorErrorModel, return_col_dict=False):

    DL_ids: Dict[str, int] = {} # detectors + logical operators
    L_map: Dict[int, FrozenSet[int]] = {} # logical operators
    priors_dict: Dict[int, float] = {} # for each fault

    def handle_error(prob: float, detectors: List[int], observables: List[int]) -> None:
        dets = frozenset(detectors)
        obs = frozenset(observables)
        key = " ".join([f"D{s}" for s in sorted(dets)] + [f"L{s}" for s in sorted(obs)])

        if key not in DL_ids:
            DL_ids[key] = len(DL_ids)
            priors_dict[DL_ids[key]] = 0.0

        hid = DL_ids[key]
        L_map[hid] = obs
#         priors_dict[hid] = priors_dict[hid] * (1 - prob) + prob * (1 - priors_dict[hid])
        priors_dict[hid] += prob

    for instruction in dem.flattened():
        if instruction.type == "error":
            dets: List[int] = []
            frames: List[int] = []
            t: stim.DemTarget
            p = instruction.args_copy()[0]
            for t in instruction.targets_copy():
                if t.is_relative_detector_id():
                    dets.append(t.val)
                elif t.is_logical_observable_id():
                    frames.append(t.val)
            handle_error(p, dets, frames)
        elif instruction.type == "detector":
            pass
        elif instruction.type == "logical_observable":
            pass
        else:
            raise NotImplementedError()
    check_matrix = dict_to_csc_matrix({v: [int(s[1:]) for s in k.split(" ") if s.startswith("D")] 
                                       for k, v in DL_ids.items()},
                                      shape=(dem.num_detectors, len(DL_ids)))
    observables_matrix = dict_to_csc_matrix(L_map, shape=(dem.num_observables, len(DL_ids)))
    priors = np.zeros(len(DL_ids))
    for i, p in priors_dict.items():
        priors[i] = p

    if return_col_dict:
        return check_matrix, observables_matrix, priors, DL_ids
    return check_matrix, observables_matrix, priors


    
def circuit_level_simulation(code, error_rate, decoders,
    num_repeat=12,
    num_trials=10000,
    W=1,
    F=1,
    z_basis=False,
    noisy_prior=None,
    method = 2,
    plot = False
):

    circuit = build_circuit(code, code.A_matrixs, code.B_matrixs, error_rate, num_repeat, z_basis=z_basis)
    dem = circuit.detector_error_model()
    chk, obs, priors, col_dict = dem_to_check_matrices(dem, return_col_dict=True)
    n = code.N
    n_half = n // 2
    # chk = chk[:,:-n_half]
    # obs = obs[:,:-n_half]
    # priors = priors[:-n_half]
    num_row, num_col = chk.shape
    code_h = code.hz if z_basis else code.hx
    code_l = code.lz if z_basis else code.lx

    lower_bounds = []
    upper_bounds = []
    i = 0
    while i < num_row:
        lower_bounds.append(i)
        upper_bounds.append(i + n_half)
        if i + n > num_row:
            break
        lower_bounds.append(i)
        upper_bounds.append(i + n)
        i += n_half

    region_dict = {}
    for i, (l, u) in enumerate(zip(lower_bounds, upper_bounds)):
        region_dict[(l, u)] = i

    region_cols = [[] for _ in range(len(region_dict))]

    for i in range(num_col):
        nnz_col = np.nonzero(chk[:, i])[0]
        l = nnz_col.min() // n_half * n_half
        u = (nnz_col.max() // n_half + 1) * n_half
        region_cols[region_dict[(l, u)]].append(i)

    chk = np.concatenate([chk[:, col].toarray() for col in region_cols], axis=1)
    obs = np.concatenate([obs[:, col].toarray() for col in region_cols], axis=1)
    priors = np.concatenate([priors[col] for col in region_cols])

    anchors = []
    j = 0
    for i in range(num_col):
        nnz_col = np.nonzero(chk[:, i])[0]
        if nnz_col.min() >= j:
            anchors.append((j, i))
            j += n_half
    anchors.append((num_row, num_col))

    if noisy_prior is None and method != 0:
        b = anchors[W]
        c = anchors[W - 1]
        if method == 1:
            c = (c[0], c[1] + n_half * 3) if z_basis else (c[0], c[1] + n)
        #             c = (c[0], c[1]+n_half*3) # try also this for x basis, change the later one as well
        noisy_prior = np.sum(
            chk[c[0] : b[0], c[1] : b[1]] * priors[c[1] : b[1]], axis=1
        )
        print("prior for noisy syndrome", noisy_prior[0])

    if method != 0:
        noisy_syndrome_priors = np.ones(n_half) * noisy_prior

    num_win = math.ceil((len(anchors) - W + F - 1) / F)
    chk_submats = []
    prior_subvecs = []
    if plot:
        fig, ax = plt.subplots(num_win, 1)
    top_left = 0
    i = 0
    for i in range(num_win):
        a = anchors[top_left]
        bottom_right = min(top_left + W, len(anchors) - 1)
        b = anchors[bottom_right]

        if i != num_win - 1 and method != 0:  # not the last round
            c = anchors[top_left + W - 1]
            if method == 1:
                c = (c[0], c[1] + n_half * 3) if z_basis else (c[0], c[1] + n)
            #                 c = (c[0], c[1]+n_half*3) # try also this for x basis, change the previous one as well
            noisy_syndrome = np.zeros((n_half * W, n_half))
            noisy_syndrome[-n_half:, :] = np.eye(n_half)  # * noisy_syndrome_prior
            mat = chk[a[0] : b[0], a[1] : c[1]]
            mat = np.hstack((mat, noisy_syndrome))
            prior = priors[a[1] : c[1]]
            prior = np.concatenate((prior, noisy_syndrome_priors))
        else:  # method==0 or last round
            mat = chk[a[0] : b[0], a[1] : b[1]]
            prior = priors[a[1] : b[1]]
        chk_submats.append(mat)
        prior_subvecs.append(prior)
        if plot:
            if num_win == 1:
                ax.imshow(mat, cmap="gist_yarg")
            else:
                ax[i].imshow(mat, cmap="gist_yarg")
        top_left += F

    # save figure
    if plot:
        plt.savefig("circuit_level_decodingmatrix.png")
    start_time = time.perf_counter()
    dem_sampler: stim.CompiledDemSampler = dem.compile_sampler()
    det_data, obs_data, err_data = dem_sampler.sample(
        shots=num_trials, return_errors=False, bit_packed=False
    )
    end_time = time.perf_counter()
    print(
        f"Stim: noise sampling for {num_trials} shots, Elapsed time:",
        end_time - start_time,
    )
    logical_errs = {decoder.name:0 for decoder in decoders}
    logical_errs_per_round = {decoder.name:0 for decoder in decoders}
    decodingtime= {}
    if method == 4: # use the last round decoding
        for decoder in decoders:
            print(f">>> testing {decoder.name} >>>")
            new_det_data = det_data.copy()
            last_syndrome = np.zeros((num_trials,n_half), dtype=int)
            for i in range(num_win):
                last_syndrome ^= det_data[:,n_half*W*i:n_half*W*(i+1)]
            decoder.set_h(code_h, [None], error_rate)
            start_time = time.perf_counter()
            all_e_hat = np.zeros((num_trials, n), dtype=int)
            for j in range(num_trials):
                syndrome = np.array([1 if item else 0 for item in last_syndrome[j]])
                decodingerror = decoder.decode(syndrome)
                all_e_hat[j] = decodingerror
            decoding_end_time = time.perf_counter()
            decodingtime[decoder.name] = (decoding_end_time-start_time)/num_trials
            logical_e_hat = all_e_hat @ code_l.T % 2
            logical_err = ((obs_data + logical_e_hat) % 2).any(axis=1)
            num_err = logical_err.astype(int).sum()
            p_l = num_err / num_trials
            p_l_per_round = 1 - (1 - p_l) ** (1 / num_repeat)
            # may also use ** (1/(num_repeat-1))
            # because the first round is for encoding, the next (num_repeat-1) rounds are syndrome measurements rounds
            print("logical error per round:", p_l_per_round)
            logical_errs[decoder.name] = p_l
            logical_errs_per_round[decoder.name] = p_l_per_round
    elif method == 3: # use max_likelihood decoding without windowing
        for decoder in decoders:
            print(f">>> testing {decoder.name} >>>")
            new_det_data = det_data.copy()
            start_time = time.perf_counter()
            all_e_hat = np.zeros((num_trials, n* (num_repeat +1)))
            all_m_hat = np.zeros((num_trials, n_half*num_repeat))
            decodingtime_list = []
            fullhz = np.zeros((n_half*(num_repeat+1), n* (num_repeat +1)+ n_half*num_repeat),dtype=int)
            for i in range(num_win):
                if i == 0:
                    fullhz[n_half*i:n_half*(i+1),(n+n_half)*i:(n+n_half)*i+n] = code_h
                    fullhz[n_half*i:n_half*(i+1),(n+n_half)*i+n:(n+n_half)*(i+1)] = np.identity(n_half)
                elif i == num_win - 1:
                    fullhz[n_half*i:n_half*(i+1),(n+n_half)*i-n_half:(n+n_half)*i] = np.identity(n_half)
                    fullhz[n_half*i:n_half*(i+1),(n+n_half)*i:(n+n_half)*i+n] = code_h
                else:
                    fullhz[n_half*i:n_half*(i+1),(n+n_half)*i-n_half:(n+n_half)*i] = np.identity(n_half)
                    fullhz[n_half*i:n_half*(i+1),(n+n_half)*i:(n+n_half)*i+n] = code_h
                    fullhz[n_half*i:n_half*(i+1),(n+n_half)*i+n:(n+n_half)*(i+1)] = np.identity(n_half)
            decoder.set_h(fullhz, [None], error_rate)
            all_e_m_hat = np.zeros((num_trials, n* (num_repeat +1)+ n_half*num_repeat))
            for j in range(num_trials):
                syndrome = np.array([1 if item else 0 for item in new_det_data[j]])
                decodingerror = decoder.decode(syndrome)
                all_e_m_hat[j] = decodingerror
            final_e_hat = np.zeros((num_trials, n), dtype=int)
            left = 0
            for i in range(num_win):
                final_e_hat = (final_e_hat + all_e_m_hat[:,left:left+n]) % 2
                left += n+n_half
            logical_e_hat = final_e_hat @ code_l.T %2
            logical_err = ((obs_data + logical_e_hat) % 2).any(axis=1)
            num_err = logical_err.astype(int).sum()
            p_l = num_err / num_trials
            p_l_per_round = 1 - (1 - p_l) ** (1 / num_repeat)
            # may also use ** (1/(num_repeat-1))
            # because the first round is for encoding, the next (num_repeat-1) rounds are syndrome measurements rounds
            print("logical error per round:", p_l_per_round)
            logical_errs[decoder.name] = p_l
            logical_errs_per_round[decoder.name] = p_l_per_round
                    
    elif method == 2: # use max_likelihood decoding
        for decoder in decoders:
            print(f">>> testing {decoder.name} >>>")
            new_det_data = det_data.copy()
            start_time = time.perf_counter()
            all_e_hat = np.zeros((num_trials, n* (num_repeat +1)))
            all_m_hat = np.zeros((num_trials, n_half*num_repeat))
            decodingtime_list = []
            
            # 分窗口进行解码，提升效率
            for i in range(num_win):
                if i != num_win - 1:  # not the last round
                    mat = np.hstack((code_h,np.identity(n_half)))
                    detector_win = new_det_data[:,n_half*W*i:n_half*W*(i+1)]
                    decoder.set_h(mat,[None],error_rate)
                    for j in range(num_trials):
                        syndrome = np.array([1 if item else 0 for item in detector_win[j]])
                        if i != 0:
                            syndrome = (syndrome + all_m_hat[j][n_half*W*(i-1):n_half*W*i]) % 2
                        e_hat = decoder.decode(syndrome.astype(int))  # detector_win[j] is syndrome, len == m
                        all_e_hat[j][n*W*i:n*W*(i+1)] = e_hat[:n*W]
                        all_m_hat[j][n_half*i:n_half*(i+1)] = e_hat[n*W:]
                else:  # last round
                    mat = code_h
                    detector_win = new_det_data[:,n_half*W*i:n_half*W*(i+1)]
                    decoder.set_h(mat,[None],error_rate)
                    for j in range(num_trials):
                        syndrome = np.array([1 if item else 0 for item in detector_win[j]])
                        if i != 0:
                            syndrome = (syndrome + all_m_hat[j][n_half*W*(i-1):n_half*W*i]) % 2
                        e_hat = decoder.decode(syndrome.astype(int))  # detector_win[j] is syndrome, len == m
                        all_e_hat[j][n*W*i:n*W*(i+1)] = e_hat[:n*W]
            final_e_hat = np.zeros((num_trials, n), dtype=int)
            for i in range(num_win):
                final_e_hat = (final_e_hat + all_e_hat[:,n*W*i:n*W*(i+1)]) % 2
            # final_e_hat = all_e_hat[:,-n:]
            logical_e_hat = final_e_hat @ code_l.T %2
            logical_err = ((obs_data + logical_e_hat) % 2).any(axis=1)
            num_err = logical_err.astype(int).sum()
            p_l = num_err / num_trials
            p_l_per_round = 1 - (1 - p_l) ** (1 / num_repeat)
            # may also use ** (1/(num_repeat-1))
            # because the first round is for encoding, the next (num_repeat-1) rounds are syndrome measurements rounds
            print("logical error per round:", p_l_per_round)
            logical_errs[decoder.name] = p_l
            logical_errs_per_round[decoder.name] = p_l_per_round
            
    else:
        for decoder in decoders:
            print(f">>> testing {decoder.name} >>>")
            total_e_hat = np.zeros((num_trials, num_col))
            new_det_data = det_data.copy()
            start_time = time.perf_counter()
            top_left = 0
            decodingtime_list = []
            # 分窗口进行解码，提升效率
            for i in range(num_win):
                mat = chk_submats[i]
                prior = prior_subvecs[i]
                a = anchors[top_left]
                bottom_right = min(top_left + W, len(anchors) - 1)
                b = anchors[bottom_right]
                if method == 0:  # not the last round
                    c = anchors[top_left + F]  # commit region bottom right
                # 判断当前窗口是否解码成功。每个窗口内解码num_trials次
                elif i != num_win - 1:  # not the last round
                    c = anchors[top_left + W - 1]
                    if method == 1:
                        c = (c[0], c[1] + n_half * 3) if z_basis else (c[0], c[1] + n)
                
                num_flag_err = 0
                detector_win = new_det_data[:, a[0] : b[0]]
                # if i == num_win - 1:  # last window
                #     if 'Gauss' in decoder.name:
                #         decoder = decoders[1]
                #         assert decoder.name == 'BP'
                decoder.set_h(mat, prior, error_rate)
                for j in range(num_trials):
                    syndrome = np.array([1 if item else 0 for item in detector_win[j]])
                    decoding_start_time = time.perf_counter()
                    # print(f"detector_win[j] = {detector_win[j]}")
                    # print(f"syndrome = {syndrome}")
                    e_hat = decoder.decode(syndrome)  # detector_win[j] is syndrome, len == m
                    decoding_end_time = time.perf_counter()
                    decodingtime_list.append(decoding_end_time-decoding_start_time)
                    # if shorten: print(f"pm: {bpd.min_pm}")
                    
                    is_flagged = ((mat @ e_hat + detector_win[j]) % 2).any()
                    if decoder.name == 'Sparse_Gauss_both':
                        assert not is_flagged
                    num_flag_err += is_flagged
                    if i == num_win - 1:  # last window
                        total_e_hat[j][a[1] : b[1]] = e_hat
                    else:
                        total_e_hat[j][a[1] : c[1]] = e_hat[: c[1] - a[1]]
                        
                print(f"Window {i}, flagged Errors: {num_flag_err}/{num_trials}")

                new_det_data = (det_data + total_e_hat @ chk.T) % 2
                top_left += F

            end_time = time.perf_counter()
            print("Elapsed time:", end_time - start_time)

            flagged_err = ((det_data + total_e_hat @ chk.T) % 2).any(axis=1)
            num_flagged_err = flagged_err.astype(int).sum()
            print(f"Overall Flagged Errors: {num_flagged_err}/{num_trials}")
            logical_err = ((obs_data + total_e_hat @ obs.T) % 2).any(axis=1)
            num_err = np.logical_or(flagged_err, logical_err).astype(int).sum()
            # num_err = logical_err.astype(int).sum()
            print(f"Logical Errors: {num_err}/{num_trials}")
            p_l = num_err / num_trials
            p_l_per_round = 1 - (1 - p_l) ** (1 / num_repeat)
            # may also use ** (1/(num_repeat-1))
            # because the first round is for encoding, the next (num_repeat-1) rounds are syndrome measurements rounds
            print("logical error per round:", p_l_per_round)
            print(f">>> end testing >>>")
            logical_errs[decoder.name] = p_l
            logical_errs_per_round[decoder.name] = p_l_per_round
            decodingtime[decoder.name] = np.mean(decodingtime_list)
        
    return logical_errs, logical_errs_per_round
