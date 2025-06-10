import os
import time
import numpy as np
import matplotlib.pyplot as plt
import torch

from src.data_generation import sample_parameters, build_common_times
from src.utils import generate_pycbc_waveform
from src.models import PhaseDNN_Full, AmplitudeNet
from src.evaluate import load_models

from src.config import (
    WAVEFORM_NAME, DELTA_T, F_LOWER, DETECTOR_NAME, PSI_FIXED,
    T_BEFORE, T_AFTER,
    MASS_MIN, MASS_MAX, SPIN_MAG_MIN, SPIN_MAG_MAX,
    INCL_MIN, INCL_MAX, ECC_MIN, ECC_MAX,
    RA_MIN, RA_MAX, DEC_MIN, DEC_MAX,
    DIST_MIN, DIST_MAX, COAL_MIN, COAL_MAX,
    NUM_SAMPLES, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE,
    PATIENCE, FINE_TUNE_EPOCHS, FINE_TUNE_LR,
    DEVICE, CHECKPOINT_DIR
)

def benchmark(N_waveforms, checkpoint_dir="checkpoints"):
    """
    For N_waveforms random parameter sets (sampled on‐the‐fly), compares:
      1) Time to generate PyCBC waveforms on a fixed grid.
      2) Time for DNN inference over the same grid (batched).
    Returns:
      time_model (float)   : total DNN inference time (seconds)
      time_pycbc (float)   : total PyCBC waveform generation time (seconds)
      mae_per_waveform (np.ndarray of shape (N_waveforms,))
    """
    # 1) Sample N_waveforms random parameter sets
    param_list, thetas = sample_parameters(N_waveforms)

    # 2) Build common grid (shared by both methods)
    common_times, N_common = build_common_times(delta_t=DELTA_T, t_before=T_BEFORE, t_after=T_AFTER)

    # 3) Compute normalization stats from these thetas
    param_means = thetas.mean(axis=0).astype(np.float32)   # (15,)
    param_stds  = thetas.std(axis=0).astype(np.float32)    # (15,)

    # 4) Precompute time_norm ∈ [–1, +1]
    time_norm = ((2.0 * (common_times + T_BEFORE) / (T_BEFORE + T_AFTER)) - 1.0).astype(np.float32)

    # 5) Load trained models
    phase_model, amp_model = load_models(checkpoint_dir)
    phase_model.to(DEVICE)
    amp_model.to(DEVICE)

    # 6) Time PyCBC generation for all N_waveforms
    pycbc_waveforms = np.zeros((N_waveforms, N_common), dtype=np.float32)
    A_peaks = np.zeros(N_waveforms, dtype=np.float32)

    start_pycbc = time.time()
    for i, params in enumerate(param_list):
        h_true = generate_pycbc_waveform(params, common_times)  # (N_common,)
        pycbc_waveforms[i] = h_true
        A_peaks[i] = np.max(np.abs(h_true)) + 1e-30
    time_pycbc = time.time() - start_pycbc

    # 7) Time DNN inference for all N_waveforms (batching)
    #    Build a single (N_waveforms*N_common, 16) input array
    X = np.zeros((N_waveforms * N_common, 16), dtype=np.float32)
    for i, params in enumerate(param_list):
        (m1, m2,
         S1x, S1y, S1z,
         S2x, S2y, S2z,
         incl, ecc,
         ra, dec,
         d, t0, phi0) = params

        # Normalize each parameter
        m1_n  = (m1  - param_means[0])  / param_stds[0]
        m2_n  = (m2  - param_means[1])  / param_stds[1]
        s1x_n = (S1x - param_means[2])  / param_stds[2]
        s1y_n = (S1y - param_means[3])  / param_stds[3]
        s1z_n = (S1z - param_means[4])  / param_stds[4]
        s2x_n = (S2x - param_means[5])  / param_stds[5]
        s2y_n = (S2y - param_means[6])  / param_stds[6]
        s2z_n = (S2z - param_means[7])  / param_stds[7]
        inc_n = (incl - param_means[8])  / param_stds[8]
        ecc_n = (ecc  - param_means[9])  / param_stds[9]
        ra_n  = (ra   - param_means[10]) / param_stds[10]
        dec_n = (dec  - param_means[11]) / param_stds[11]
        d_n   = (d    - param_means[12]) / param_stds[12]
        t0_n  = (t0   - param_means[13]) / param_stds[13]
        ph0_n = (phi0 - param_means[14]) / param_stds[14]

        base = i * N_common
        X[base:base+N_common,  0] = time_norm
        X[base:base+N_common,  1] = m1_n
        X[base:base+N_common,  2] = m2_n
        X[base:base+N_common,  3] = s1x_n
        X[base:base+N_common,  4] = s1y_n
        X[base:base+N_common,  5] = s1z_n
        X[base:base+N_common,  6] = s2x_n
        X[base:base+N_common,  7] = s2y_n
        X[base:base+N_common,  8] = s2z_n
        X[base:base+N_common,  9] = inc_n
        X[base:base+N_common, 10] = ecc_n
        X[base:base+N_common, 11] = ra_n
        X[base:base+N_common, 12] = dec_n
        X[base:base+N_common, 13] = d_n
        X[base:base+N_common, 14] = t0_n
        X[base:base+N_common, 15] = ph0_n

    X_torch = torch.from_numpy(X).to(DEVICE)

    start_model = time.time()
    with torch.no_grad():
        A_pred_all   = amp_model(X_torch).cpu().numpy().ravel()     # (N_waveforms*N_common,)
        phi_pred_all = phase_model(X_torch[:, 0:1], X_torch[:, 1:]).cpu().numpy().ravel()
    time_model = time.time() - start_model

    # Reshape predictions to (N_waveforms, N_common)
    A_pred_all   = A_pred_all.reshape(N_waveforms, N_common)
    phi_pred_all = phi_pred_all.reshape(N_waveforms, N_common)

    # 8) Compute MAE per waveform
    mae_per_waveform = np.zeros(N_waveforms, dtype=np.float32)
    for i in range(N_waveforms):
        h_pred_i = A_peaks[i] * A_pred_all[i] * np.cos(phi_pred_all[i])
        h_true_i = pycbc_waveforms[i]
        mae_per_waveform[i] = np.mean(np.abs(h_pred_i - h_true_i))

    return time_model, time_pycbc, mae_per_waveform


if __name__ == "__main__":
    sizes = [1, 10, 50, 100]

    results = []
    for N in sizes:
        tm, tp, maes = benchmark(N, checkpoint_dir="checkpoints")
        avg_mae = np.mean(maes)
        results.append((N, tm, tp, avg_mae))
        print(f"N={N:3d} | Model time = {tm:.3f}s | PyCBC time = {tp:.3f}s | MAE = {avg_mae:.3e}")

    # Plot generation time vs. N
    Ns       = [r[0] for r in results]
    times_m  = [r[1] for r in results]
    times_p  = [r[2] for r in results]
    mae_vals = [r[3] for r in results]

    plt.figure(figsize=(8, 4))
    plt.plot(Ns, times_m, 'o-', label="DNN inference")
    plt.plot(Ns, times_p, 's--', label="PyCBC waveform")
    plt.xlabel("Number of waveforms")
    plt.ylabel("Total generation time (s)")
    plt.title("Generation time: DNN vs. PyCBC")
    plt.legend()
    plt.tight_layout()
    plt.savefig("benchmark_time_vs_N.png")
    plt.close()

    # Plot accuracy (MAE) vs. Model time per waveform
    time_per_waveform = np.array(times_m) / np.array(Ns)
    plt.figure(figsize=(8, 4))
    plt.plot(time_per_waveform, mae_vals, 'o-')
    plt.xlabel("Model time per waveform (s)")
    plt.ylabel("Mean Absolute Error")
    plt.title("Accuracy vs. DNN inference speed")
    plt.tight_layout()
    plt.savefig("benchmark_accuracy_vs_time.png")
    plt.close()

    print("Saved benchmark plots:")
    print("  - benchmark_time_vs_N.png")
    print("  - benchmark_accuracy_vs_time.png")
