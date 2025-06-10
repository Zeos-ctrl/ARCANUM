# src/evaluate.py

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from scipy.signal import hilbert
from scipy.stats import gaussian_kde

from src.data_generation import sample_parameters, build_common_times, build_waveform_chunks
from src.models import PhaseDNN_Full, AmplitudeNet
from src.utils import trimming_indices

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

def load_models(checkpoint_dir: str):
    """
    Instantiate PhaseDNN_Full and AmplitudeNet, load saved weights from checkpoint_dir.
    Returns:
      phase_model, amp_model
    """
    phase_model = PhaseDNN_Full(
        param_dim=15,
        time_dim=1,
        emb_hidden=[256, 256, 256],
        emb_dim=256,
        phase_hidden=[192]*8,
        N_banks=3,
        dropout_p=0.2
    ).to(DEVICE)

    amp_model = AmplitudeNet(
        in_dim=16,
        hidden_dims=[192]*8,
        dropout_p=0.2
    ).to(DEVICE)

    phase_path = os.path.join(checkpoint_dir, "phase_model_best.pth")
    amp_path = os.path.join(checkpoint_dir, "amp_model_best.pth")

    if not os.path.exists(phase_path) or not os.path.exists(amp_path):
        raise FileNotFoundError(
            f"Expected model files not found in {checkpoint_dir}:\n"
            f"  {phase_path}\n  {amp_path}"
        )

    phase_model.load_state_dict(torch.load(phase_path, map_location=DEVICE))
    amp_model.load_state_dict(torch.load(amp_path, map_location=DEVICE))

    phase_model.eval()
    amp_model.eval()

    return phase_model, amp_model


def compute_laplace_hessians(train_loader, phase_model, amp_model, lambda_A=1e-6, lambda_phi=1e-6):
    """
    For the amplitude network: let 'feature_extractor_A' be all layers up to but not
    including amp_model.linear_out. Compute C_A = sum(feats.T @ feats) across train_loader,
    then H_A = C_A + lambda_A * I, and Σ_A = inv(H_A).

    For the phase network (with N_banks): for each bank i, let phi_i(x) = [t_norm; θ_embed(x)],
    so collect C_i = sum(phi_i_batch.T @ phi_i_batch), H_i = C_i + lambda_phi * I, Σ_i = inv(H_i).

    Returns:
      Σ_A:          np.ndarray of shape (d_A, d_A)
      Σ_phase_list: list of length N_banks, each a (d_phase, d_phase) np.ndarray
    """
    C_A = None
    Σ_phase_list = []
    # --- Amplitude network ---
    amp_body_layers = list(amp_model.net_body.children())
    amp_last_linear = amp_model.linear_out  # nn.Linear
    feature_extractor_A = nn.Sequential(*amp_body_layers).to(DEVICE)
    d_A = amp_last_linear.weight.shape[1]

    C_A = np.zeros((d_A, d_A), dtype=np.float64)
    with torch.no_grad():
        for x_batch, _ in train_loader:
            feats = feature_extractor_A(x_batch.to(DEVICE)).cpu().numpy()  # (batch, d_A)
            C_A += feats.T.dot(feats)

    H_A = C_A + lambda_A * np.eye(d_A, dtype=np.float64)
    Σ_A = np.linalg.inv(H_A)

    # --- Phase network ---
    emb_dim = phase_model.theta_embed(torch.zeros(1, 15)).shape[-1]
    d_phase = emb_dim + 1

    with torch.no_grad():
        for bank in range(phase_model.N_banks):
            C_i = np.zeros((d_phase, d_phase), dtype=np.float64)
            for x_batch, _ in train_loader:
                batch = x_batch.to(DEVICE)
                t_b = batch[:, 0:1]                            # (batch,1)
                θ_embed = phase_model.theta_embed(batch[:, 1:]).cpu().numpy()  # (batch, emb_dim)
                t_np = t_b.cpu().numpy()                       # (batch,1)
                phi_i = np.concatenate([t_np, θ_embed], axis=1)  # (batch, d_phase)
                C_i += phi_i.T.dot(phi_i)

            H_i = C_i + lambda_phi * np.eye(d_phase, dtype=np.float64)
            Σ_i = np.linalg.inv(H_i)
            Σ_phase_list.append(Σ_i)

    return Σ_A, Σ_phase_list


def generate_pycbc_waveform(params, common_times):
    """
    Given params = (m1,m2,S1x,S1y,S1z,S2x,S2y,S2z,incl,ecc,ra,dec,d,t0,phi0),
    generate the detector-frame waveform on PyCBC's grid, then resample/pad
    onto the global `common_times` grid.
    Returns: h_true_common (length len(common_times))
    """
    from pycbc.waveform import get_td_waveform
    from pycbc.detector import Detector

    (m1, m2,
     S1x, S1y, S1z,
     S2x, S2y, S2z,
     incl, ecc,
     ra, dec,
     d, t0, phi0) = params

    hp, hc = get_td_waveform(
        mass1             = m1,
        mass2             = m2,
        spin1x            = S1x,
        spin1y            = S1y,
        spin1z            = S1z,
        spin2x            = S2x,
        spin2y            = S2y,
        spin2z            = S2z,
        eccentricity      = ecc,
        inclination       = incl,
        distance          = d,
        coalescence_time  = t0,
        coalescence_phase = phi0,
        delta_t           = DELTA_T,
        f_lower           = F_LOWER,
        approximant       = WAVEFORM
    )
    h_plus = hp.numpy().astype(np.float32)
    h_cross = hc.numpy().astype(np.float32)
    t_plus = hp.sample_times.numpy().astype(np.float32)

    det = Detector(DETECTOR_NAME)
    Fp, Fx = det.antenna_pattern(ra, dec, 0.0, 0.0)
    h_det_pycbc = (Fp * h_plus + Fx * h_cross).astype(np.float32)

    h_true_common = np.zeros_like(common_times, dtype=np.float32)
    idxs = np.round((t_plus - common_times[0]) / DELTA_T).astype(int)
    valid = (idxs >= 0) & (idxs < len(common_times))
    h_true_common[idxs[valid]] = h_det_pycbc[valid]

    return h_true_common


def evaluate(checkpoint_dir: str = "checkpoints", output_dir: str = "plots", num_examples: int = 3):
    """
    Full evaluation pipeline:
      1. Sample a fresh set of NUM_SAMPLES parameters.
      2. Build common grid and waveform chunks.
      3. Normalize parameters and precompute time_norm.
      4. Assemble a DataLoader over the entire dataset (for Laplace).
      5. Load trained models and compute Laplace covariances.
      6. For num_examples random waveforms, plot amplitude & phase comparisons & uncertainty.
      7. Compute pointwise errors over all NUM_SAMPLES to:
         - Plot KDE of errors
         - Plot empirical CDF of absolute errors
         - Print summary statistics
         - Plot signed error vs. time for one random sample
         - Plot mean absolute error vs. time
         - Plot heatmap of absolute errors (samples × time)
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1) SAMPLE PARAMETERS
    param_list, thetas = sample_parameters(NUM_SAMPLES)

    # 2) BUILD COMMON TIME GRID
    common_times, N_common = build_common_times(delta_t=DELTA_T, t_before=T_BEFORE, t_after=T_AFTER)

    # 3) GENERATE WAVEFORM CHUNKS (for GWFlatDataset and Laplace)
    waveform_chunks = build_waveform_chunks(
        param_list=param_list,
        common_times=common_times,
        n_common=N_common,
        delta_t=DELTA_T,
        f_lower=F_LOWER,
        waveform_name=WAVEFORM,
        detector_name=DETECTOR_NAME,
        psi_fixed=PSI_FIXED,
    )

    # 4) NORMALIZE PARAMETERS
    param_means = thetas.mean(axis=0).astype(np.float32)  # (15,)
    param_stds  = thetas.std(axis=0).astype(np.float32)  # (15,)
    theta_norm_all = ((thetas - param_means) / param_stds).astype(np.float32)

    # 5) PRECOMPUTE time_norm
    time_norm = ((2.0 * (common_times + T_BEFORE) / (T_BEFORE + T_AFTER)) - 1.0).astype(np.float32)

    # 6) BUILD FULL DATASET & DATALOADER FOR LAPLACE
    from src.dataset import GWFlatDataset
    from torch.utils.data import DataLoader, Subset

    dataset = GWFlatDataset(
        waveform_chunks=waveform_chunks,
        theta_norm_all=theta_norm_all,
        time_norm=time_norm,
        N_common=N_common
    )
    all_indices = list(range(NUM_SAMPLES * N_common))
    train_loader = DataLoader(Subset(dataset, all_indices), batch_size=1024, shuffle=False)

    # 7) LOAD MODELS
    phase_model, amp_model = load_models(checkpoint_dir)

    # 8) COMPUTE LAPLACE COVARIANCES
    Σ_A, Σ_phase_banks = compute_laplace_hessians(train_loader, phase_model, amp_model)

    # 9) EXTRACT FEATURE EXTRACTOR FOR AMP UNCERTAINTY
    amp_body_layers = list(amp_model.net_body.children())
    feature_extractor_A = nn.Sequential(*amp_body_layers).to(DEVICE)

    # 10) INDIVIDUAL WAVEFORM PLOTS & UNCERTAINTY
    K = min(num_examples, NUM_SAMPLES)
    indices = np.random.choice(NUM_SAMPLES, K, replace=False)

    for idx_k, i in enumerate(indices):
        params = param_list[i]
        m1, m2, S1x, S1y, S1z, S2x, S2y, S2z, incl, ecc, ra, dec, d, t0, phi0 = params

        # (a) Regenerate true waveform on full grid
        h_true_full = generate_pycbc_waveform(params, common_times)

        analytic_true = hilbert(h_true_full)
        inst_amp_true = np.abs(analytic_true).astype(np.float32)
        phi_true = np.unwrap(np.angle(analytic_true)).astype(np.float32)
        A_peak = inst_amp_true.max() + 1e-30
        amp_norm_true = inst_amp_true / A_peak

        # (b) Build input grid (N_common × 16)
        raw_theta = thetas[i]
        m1_n  = (raw_theta[0] - param_means[0])  / param_stds[0]
        m2_n  = (raw_theta[1] - param_means[1])  / param_stds[1]
        s1x_n = (raw_theta[2] - param_means[2])  / param_stds[2]
        s1y_n = (raw_theta[3] - param_means[3])  / param_stds[3]
        s1z_n = (raw_theta[4] - param_means[4])  / param_stds[4]
        s2x_n = (raw_theta[5] - param_means[5])  / param_stds[5]
        s2y_n = (raw_theta[6] - param_means[6])  / param_stds[6]
        s2z_n = (raw_theta[7] - param_means[7])  / param_stds[7]
        inc_n = (raw_theta[8] - param_means[8])  / param_stds[8]
        ecc_n = (raw_theta[9] - param_means[9])  / param_stds[9]
        ra_n  = (raw_theta[10] - param_means[10]) / param_stds[10]
        dec_n = (raw_theta[11] - param_means[11]) / param_stds[11]
        d_n   = (raw_theta[12] - param_means[12]) / param_stds[12]
        t0_n  = (raw_theta[13] - param_means[13]) / param_stds[13]
        ph0_n = (raw_theta[14] - param_means[14]) / param_stds[14]

        inp_wave = np.zeros((N_common, 16), dtype=np.float32)
        inp_wave[:,  0] = time_norm
        inp_wave[:,  1] = m1_n
        inp_wave[:,  2] = m2_n
        inp_wave[:,  3] = s1x_n
        inp_wave[:,  4] = s1y_n
        inp_wave[:,  5] = s1z_n
        inp_wave[:,  6] = s2x_n
        inp_wave[:,  7] = s2y_n
        inp_wave[:,  8] = s2z_n
        inp_wave[:,  9] = inc_n
        inp_wave[:, 10] = ecc_n
        inp_wave[:, 11] = ra_n
        inp_wave[:, 12] = dec_n
        inp_wave[:, 13] = d_n
        inp_wave[:, 14] = t0_n
        inp_wave[:, 15] = ph0_n

        inp_wave_t = torch.from_numpy(inp_wave).to(DEVICE)

        # (c) Predict normalized amplitude & dphi; reconstruct phi_pred
        with torch.no_grad():
            A_pred_norm = amp_model(inp_wave_t).cpu().numpy().ravel()   # (N_common,)
            dphi_pred   = phase_model(inp_wave_t[:, 0:1], inp_wave_t[:, 1:]).cpu().numpy().ravel()
        phi_pred = np.cumsum(dphi_pred).astype(np.float32)  # (N_common,)

        # (d) Plot normalized amplitude: true vs. predicted
        t_full = common_times
        plt.figure(figsize=(10, 4))
        plt.plot(t_full, amp_norm_true, label="True $A_{\\mathrm{norm}}(t)$", linewidth=1)
        plt.plot(t_full, A_pred_norm, label="Predicted $\\widehat{A}_{\\mathrm{norm}}(t)$",
                 linestyle="--", linewidth=1)
        plt.xlabel("Time [s]")
        plt.ylabel("Normalized Amplitude")
        plt.title(f"Waveform #{i}: Amplitude Comparison")
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"amp_compare_{i}.png"))
        plt.close()

        # (e) Plot unwrapped phase: true vs. predicted
        plt.figure(figsize=(10, 4))
        plt.plot(t_full, phi_true,   label="True $\\phi(t)$", linewidth=1)
        plt.plot(t_full, phi_pred,   label="Predicted $\\widehat{\\phi}(t)$",
                 linestyle="--", linewidth=1)
        plt.xlabel("Time [s]")
        plt.ylabel("Unwrapped Phase [rad]")
        plt.title(f"Waveform #{i}: Phase Comparison")
        plt.legend(loc="upper left")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"phase_compare_{i}.png"))
        plt.close()

        # (f) Compute amplitude uncertainty var_A
        with torch.no_grad():
            feats_A = feature_extractor_A(inp_wave_t).cpu().numpy()  # (N_common, d_A)
        var_A = np.einsum('bi,ij,bj->b', feats_A, Σ_A, feats_A)     # (N_common,)

        # (g) Compute phase uncertainty var_phi
        with torch.no_grad():
            θ_embed_all = phase_model.theta_embed(inp_wave_t[:, 1:]).cpu().numpy()  # (N_common, emb_dim)
            t_np = inp_wave[:, 0:1]  # (N_common,1)

        var_phi = np.zeros(N_common, dtype=np.float64)
        for b in range(phase_model.N_banks):
            phi_i_batch = np.concatenate([t_np, θ_embed_all], axis=1)  # (N_common, emb_dim+1)
            Σ_i = Σ_phase_banks[b]
            var_i = np.einsum('bi,ij,bj->b', phi_i_batch, Σ_i, phi_i_batch)  # (N_common,)
            var_phi += var_i

        # (h) Propagate to h(t)
        dhdA   = A_peak * np.cos(phi_true)                              # (N_common,)
        dhdphi = -A_peak * A_pred_norm * np.sin(phi_pred)               # (N_common,)

        var_h   = (dhdA**2) * var_A + (dhdphi**2) * var_phi             # (N_common,)
        sigma_h = np.sqrt(var_h)                                        # (N_common,)

        # (i) Trim zeros to plot active region + buffer
        start_idx, end_idx = trimming_indices(h_true_full, buffer=0.1, delta_t=DELTA_T)
        t_plot      = common_times[start_idx:end_idx]
        h_true_plot = h_true_full[start_idx:end_idx]
        h_pred_plot = A_peak * A_pred_norm[start_idx:end_idx] * np.cos(phi_pred[start_idx:end_idx])
        sigma_plot  = sigma_h[start_idx:end_idx]

        # (j) Plot h(t) ± 2σ
        plt.figure(figsize=(10, 4))
        plt.plot(t_plot,      h_true_plot, color="#000000", label="True $h(t)$", linewidth=1)
        plt.plot(t_plot,      h_pred_plot, color="#1f77b4", label=r"Predicted $\hat{h}(t)$", linewidth=1)
        plt.fill_between(
            t_plot,
            h_pred_plot - 2 * sigma_plot,
            h_pred_plot + 2 * sigma_plot,
            color="#1f77b4",
            alpha=0.3,
            label="±2σ uncertainty"
        )
        plt.xlabel("Time [s]")
        plt.ylabel("Strain")
        plt.title(f"Waveform #{i}: $h(t)$ ±2σ")
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"waveform_uncertainty_{i}.png"))
        plt.close()

        # (k) Percentage‐error summary for this waveform
        abs_errors_1d = np.abs(h_pred_plot - h_true_plot)  # (active_length,)
        perc_errors_1d = 100.0 * abs_errors_1d / (np.abs(h_true_plot) + 1e-30)

        pct_50 = np.percentile(perc_errors_1d, 50)
        pct_90 = np.percentile(perc_errors_1d, 90)
        pct_95 = np.percentile(perc_errors_1d, 95)
        pct_99 = np.percentile(perc_errors_1d, 99)

        print(f"Waveform #{i} Percentage‐Error Summary:")
        print(f"  50th  %ile: {pct_50:.3f} %")
        print(f"  90th  %ile: {pct_90:.3f} %")
        print(f"  95th  %ile: {pct_95:.3f} %")
        print(f"  99th  %ile: {pct_99:.3f} %")

    # 11) POINTWISE ERRORS ACROSS ALL WAVEFORMS & GLOBAL PLOTS/STATISTICS
    # Regenerate all true & predicted waveforms, compute errors
    all_errors_list = []
    h_true_array = np.zeros((NUM_SAMPLES, N_common), dtype=np.float32)
    h_pred_array = np.zeros((NUM_SAMPLES, N_common), dtype=np.float32)

    for i in range(NUM_SAMPLES):
        params = param_list[i]
        h_true_common = generate_pycbc_waveform(params, common_times)
        h_true_array[i, :] = h_true_common

        analytic_true = hilbert(h_true_common)
        inst_amp_true = np.abs(analytic_true).astype(np.float32)
        phi_true = np.unwrap(np.angle(analytic_true)).astype(np.float32)
        A_peak = inst_amp_true.max() + 1e-30

        raw_theta = thetas[i]
        m1_n  = (raw_theta[0] - param_means[0])  / param_stds[0]
        m2_n  = (raw_theta[1] - param_means[1])  / param_stds[1]
        s1x_n = (raw_theta[2] - param_means[2])  / param_stds[2]
        s1y_n = (raw_theta[3] - param_means[3])  / param_stds[3]
        s1z_n = (raw_theta[4] - param_means[4])  / param_stds[4]
        s2x_n = (raw_theta[5] - param_means[5])  / param_stds[5]
        s2y_n = (raw_theta[6] - param_means[6])  / param_stds[6]
        s2z_n = (raw_theta[7] - param_means[7])  / param_stds[7]
        inc_n = (raw_theta[8] - param_means[8])  / param_stds[8]
        ecc_n = (raw_theta[9] - param_means[9])  / param_stds[9]
        ra_n  = (raw_theta[10] - param_means[10]) / param_stds[10]
        dec_n = (raw_theta[11] - param_means[11]) / param_stds[11]
        d_n   = (raw_theta[12] - param_means[12]) / param_stds[12]
        t0_n  = (raw_theta[13] - param_means[13]) / param_stds[13]
        ph0_n = (raw_theta[14] - param_means[14]) / param_stds[14]

        inp_wave = np.zeros((N_common, 16), dtype=np.float32)
        inp_wave[:,  0] = time_norm
        inp_wave[:,  1] = m1_n
        inp_wave[:,  2] = m2_n
        inp_wave[:,  3] = s1x_n
        inp_wave[:,  4] = s1y_n
        inp_wave[:,  5] = s1z_n
        inp_wave[:,  6] = s2x_n
        inp_wave[:,  7] = s2y_n
        inp_wave[:,  8] = s2z_n
        inp_wave[:,  9] = inc_n
        inp_wave[:, 10] = ecc_n
        inp_wave[:, 11] = ra_n
        inp_wave[:, 12] = dec_n
        inp_wave[:, 13] = d_n
        inp_wave[:, 14] = t0_n
        inp_wave[:, 15] = ph0_n

        inp_torch = torch.from_numpy(inp_wave).to(DEVICE)

        with torch.no_grad():
            A_pred_norm = amp_model(inp_torch).cpu().numpy().ravel()        # (N_common,)
            dphi_pred   = phase_model(inp_torch[:, 0:1], inp_torch[:, 1:]).cpu().numpy().ravel()
        phi_pred = np.cumsum(dphi_pred).astype(np.float32)  # (N_common,)
        h_pred_full = A_peak * A_pred_norm * np.cos(phi_pred)  # (N_common,)
        h_pred_array[i, :] = h_pred_full

        error_i = h_pred_full - h_true_common
        all_errors_list.append(error_i)

    all_errors = np.concatenate(all_errors_list)  # (NUM_SAMPLES * N_common,)

    # 11.1) KDE of errors
    kde = gaussian_kde(all_errors)
    x_grid = np.linspace(all_errors.min(), all_errors.max(), 1000)
    pdf_vals = kde(x_grid)

    plt.figure(figsize=(6, 4))
    plt.plot(x_grid, pdf_vals, color='#1f77b4')
    plt.axvline(0, color='k', linestyle='--', linewidth=1)
    plt.xlabel("Error: $h_{pred}(t)-h_{true}(t)$")
    plt.ylabel("Density (KDE)")
    plt.title("KDE of Pointwise Prediction Errors")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "kde_errors.png"))
    plt.close()

    # 11.2) Empirical CDF of absolute errors
    abs_errors = np.abs(all_errors)
    sorted_abs = np.sort(abs_errors)
    cdf = np.linspace(0, 1, len(sorted_abs))

    plt.figure(figsize=(6, 4))
    plt.plot(sorted_abs, cdf, color='#ff7f0e')
    plt.xlabel("Absolute Error $|h_{pred}-h_{true}|$")
    plt.ylabel("Empirical CDF")
    plt.title("Empirical CDF of Absolute Errors")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cdf_abs_errors.png"))
    plt.close()

    # 11.3) Summary statistics
    mean_err    = np.mean(all_errors)
    std_err     = np.std(all_errors)
    mae         = np.mean(np.abs(all_errors))
    max_err     = np.max(np.abs(all_errors))
    pct_90_err  = np.percentile(np.abs(all_errors), 90)
    pct_95_err  = np.percentile(np.abs(all_errors), 95)

    print("Error summary:")
    print(f"  Mean error:            {mean_err:.3e}")
    print(f"  Std   error:           {std_err:.3e}")
    print(f"  Mean absolute error:   {mae:.3e}")
    print(f"  Max absolute error:    {max_err:.3e}")
    print(f"  90th percentile error: {pct_90_err:.3e}")
    print(f"  95th percentile error: {pct_95_err:.3e}")

    # 11.4) Plot Signed Error vs. Time for one random sample
    plt.figure(figsize=(8, 5))
    chosen = np.random.choice(NUM_SAMPLES, size=1, replace=False)
    for idx in chosen:
        err_i = h_pred_array[idx] - h_true_array[idx]
        plt.plot(common_times, err_i, label=f"Sample {idx}")
    plt.axhline(0, color='k', linestyle='--', linewidth=0.7)
    plt.xlabel("Time [s]")
    plt.ylabel("Error $h_{pred}(t) - h_{true}(t)$")
    plt.title("Signed Error vs. Time for One Random Sample")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "signed_error_random_sample.png"))
    plt.close()

    # 11.5) Plot Mean Absolute Error vs. Time
    abs_errors_2d = np.abs(h_pred_array - h_true_array)  # (NUM_SAMPLES, N_common)
    mae_time      = abs_errors_2d.mean(axis=0)           # (N_common,)

    plt.figure(figsize=(8, 4))
    plt.plot(common_times, mae_time, color='#1f77b4')
    plt.xlabel("Time [s]")
    plt.ylabel(r"Mean Absolute Error $\mathrm{MAE}(t)$")
    plt.title("Mean Absolute Error as a Function of Time")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mae_vs_time.png"))
    plt.close()

    # 11.6) Heatmap of Absolute Errors (Samples × Time)
    plt.figure(figsize=(10, 6))
    plt.imshow(
        abs_errors_2d,
        aspect='auto',
        extent=[common_times.min(), common_times.max(), NUM_SAMPLES, 0],
        cmap='inferno'
    )
    plt.colorbar(label="|Error|")
    plt.xlabel("Time [s]")
    plt.ylabel("Sample Index")
    plt.title(r"Heatmap of $|h_{pred}-h_{true}|$ for All Samples")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "heatmap_abs_errors.png"))
    plt.close()

    print(f"\nSaved all evaluation plots to: {output_dir}")


if __name__ == "__main__":
    evaluate(checkpoint_dir="checkpoints", output_dir="plots", num_examples=3)
