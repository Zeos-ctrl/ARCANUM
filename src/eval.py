# General utils
import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as mticker
from scipy.stats import gaussian_kde

# PyCBC waveform
from pycbc.waveform import get_td_waveform
from pycbc.psd import aLIGOZeroDetHighPower

# Libraries
from src.data.config import *
from src.data.dataset import generate_data, make_waveform
from src.utils.utils import compute_match, WaveformPredictor, notify_discord
from src.data.dataset import unscale_target

import time
from typing import Sequence, Dict, Any
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Get model
try:
    wf = str(WAVEFORM).lower()
    MODEL = "EOB" if wf == "seobnrv4" else "IMR_NS"
except Exception:
    MODEL = "IMR_NS"

def pi_formatter(x, pos):
    """Format multiples of pi nicely."""
    # how many pi is this?
    m = x / np.pi
    # round to nearest half‐integer
    m_rounded = np.round(m * 2) / 2
    if m_rounded == 0:
        return "0"
    # express as fraction
    num, den = int(np.round(m_rounded * 2)), 2
    # if it's an integer multiple
    if num % den == 0:
        k = num // den
        return rf"${k}\pi$" if k != 1 else r"$\pi$"
    else:
        # we have an odd numerator
        k = num
        return rf"$\dfrac{{{k}}}{{{den}}}\pi$"

def evaluate():
    logger.info("Starting single‐waveform evaluation and visualization…")
    pred = WaveformPredictor("checkpoints", model=MODEL, device=DEVICE)
    data = generate_data(samples=10)

    i = np.random.randint(0, 10)
    m1, m2, chi1z, chi2z, incl, ecc = data.thetas[i]

    # time grid & true targets
    time = data.time_unscaled
    L = len(time)
    amp_true_norm = data.targets_A.reshape(10, L)[i]
    phi_true      = data.targets_phi.reshape(10, L)[i]
    amp_true      = unscale_target(amp_true_norm, pred.amp_scale)
    h_true        = amp_true * np.cos(phi_true)

    # model prediction
    t_norm, amp_pred, phi_pred = pred.predict_debug(m1, m2, chi1z, chi2z, incl, ecc)
    h_pred      = amp_pred * np.cos(phi_pred)

    flen = L // 2 + 1
    df = 1.0 / (L * DELTA_T)
    psd_vals = np.asarray(aLIGOZeroDetHighPower(flen, df, 20.0), dtype=float)

    match = compute_match(h_true, h_pred, DELTA_T)

    # wrapped phase and residuals
    phi_wr_true = np.mod(phi_true, 2*np.pi)
    phi_wr_pred = np.mod(phi_pred, 2*np.pi)
    # wrap residual
    dphi = phi_pred - phi_true
    phi_res_wrapped = (dphi + np.pi) % (2*np.pi) - np.pi

    # set up figure: 2 rows × 4 cols
    fig, axes = plt.subplots(2, 4, figsize=(24, 10), sharex=False)
    title_str = (f"m1={m1:.1f}, m2={m2:.1f}, "
                 f"χ1z={chi1z:.2f}, χ2z={chi2z:.2f}, "
                 f"incl={incl:.2f}, ecc={ecc:.2f}, "
                 f"waveform match = {match}")
    logger.info(f"Evaluating for parameters: {title_str}")
    logger.info(f"Using {WAVEFORM} approximant...")
    fig.suptitle(f"{title_str}", fontsize=16)

    # Top row: true vs pred
    ax = axes[0,0]
    ax.plot(time, h_true,     label="True", linewidth=1)
    ax.plot(time, h_pred, "--",label="Pred", linewidth=1)
    ax.set_title("Strain $h_+(t)$")
    ax.set_ylabel("Strain")
    ax.legend()

    # Amplitude
    ax = axes[0,1]
    ax.plot(time, amp_true,     label="True", linewidth=1)
    ax.plot(time, amp_pred, "--",label="Pred", linewidth=1)
    ax.set_title("Amplitude")
    ax.legend()

    # Unwrapped phase
    ax = axes[0,2]
    ax.plot(time, phi_true,     label="True", linewidth=1)
    ax.plot(time, phi_pred, "--",label="Pred", linewidth=1)
    ax.set_title("Phase (unwrapped)")
    ax.set_ylabel("Phase [rad]")
    ax.legend()

    # Wrapped phase
    ax = axes[0,3]
    ax.plot(time, phi_wr_true,     label="True", linewidth=1)
    ax.plot(time, phi_wr_pred, "--",label="Pred", linewidth=1)
    ax.set_title(r"Phase (wrapped $0 -> 2\pi$)")
    ax.set_ylabel("Phase")
    ax.yaxis.set_major_locator(mticker.MultipleLocator(np.pi/2))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(pi_formatter))
    ax.set_ylim(0, 2*np.pi)
    ax.set_xlim(-0.6, -0.4)
    ax.legend()

    # Bottom row: residuals
    # Strain residual
    ax = axes[1,0]
    ax.plot(time, h_pred - h_true, color="C2", linewidth=1)
    ax.set_title("Strain Residual")
    ax.set_xlabel("Time [s]")

    # Amplitude residual
    ax = axes[1,1]
    ax.plot(time, amp_pred - amp_true, color="C2", linewidth=1)
    ax.set_title("Amplitude Residual")
    ax.set_xlabel("Time [s]")

    # Unwrapped phase residual
    ax = axes[1,2]
    ax.plot(time, phi_pred - phi_true, color="C2", linewidth=1)
    ax.set_title("Phase Residual (unwrapped)")
    ax.set_ylabel("Delta Phase [rad]")
    ax.set_xlabel("Time [s]")

    # Wrapped phase residual
    ax = axes[1,3]
    ax.plot(time, phi_res_wrapped, color="C2", linewidth=1)
    ax.set_title(r"Phase Residual (wrapped $\pm \pi$)")
    ax.set_ylabel(r"Delta Phase mod $2\pi$ [rad]")
    ax.set_ylim(-np.pi, np.pi)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(np.pi/2))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(pi_formatter))
    ax.set_xlabel("Time [s]")

    plt.tight_layout(rect=[0,0,1,0.95])
    output_dir = "plots"
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "waveform_plots.png")
    plt.savefig(out_file)
    logger.info(f"Saved waveform plot to {out_file}")

    m1, m2, c1z, c2z, incl, ecc = data.thetas[i]
    plot_prediction_uncertainty(
        pred,
        m1, m2, c1z, c2z, incl, ecc,
        output_path="plots/prediction_uncertainty.png"
    )

def cross_correlation(samples=1000, checkpoint_dir="checkpoints", device=DEVICE):
    """
    Generate 'samples' waveforms, predict them, compute per-pair cross-correlation (true[i] vs pred[i]),
    and plot:
      - Grid comparison (strain, amplitude, phase) for best and worst matching pairs (differences normalized 0-1)
      - Scatter of match vs mass ratio q
      - Smooth 3D surface (m1, m2 -> match) filling the full x-y area

    Saves plots in 'plots/cross_correlation'. Returns matches array (shape (samples,)).
    """

    # Prepare output directory
    plot_dir = "plots/cross_correlation"
    os.makedirs(plot_dir, exist_ok=True)

    # Load data and model
    data = generate_data(samples=samples)
    pred = WaveformPredictor(checkpoint_dir, model=MODEL, device=device)

    thetas = data.thetas
    L = data.time_unscaled.size

    # Containers
    h_trues, h_preds, t_norms = [], [], []
    qs, m1s, m2s = [], [] ,[]
    matches = np.zeros(samples)

    # Psd stuff
    flen = L // 2 + 1
    df = 1.0 / (L * DELTA_T)
    psd_vals = np.asarray(aLIGOZeroDetHighPower(flen, df, 20.0), dtype=float)

    # Helper: normalize absolute residuals to [0,1] per-array
    def _norm01_abs(x):
        x = np.asarray(x, dtype=float)
        x = np.abs(x)
        amin = np.nanmin(x)
        amax = np.nanmax(x)
        if np.isclose(amax, amin):
            return np.zeros_like(x)
        return (x - amin) / (amax - amin)

    # Generate, predict and match each sample
    for idx in range(samples):
        m1, m2, c1z, c2z, incl_i, ecc_i = thetas[idx]
        # True waveform (unscale amplitude)
        amp_norm = data.targets_A.reshape(-1, L)[idx]
        phi_arr  = data.targets_phi.reshape(-1, L)[idx]
        amp_true = unscale_target(amp_norm, pred.amp_scale)
        h_true   = amp_true * np.cos(phi_arr)
        # Prediction (assume predict_debug returns unscaled amp and phi)
        t_norm, amp_pred, phi_pred = pred.predict_debug(m1, m2, c1z, c2z, incl_i, ecc_i)
        amp_pred = np.asarray(amp_pred)
        phi_pred = np.asarray(phi_pred)
        h_pred = amp_pred * np.cos(phi_pred)

        # Compute match
        match_val = compute_match(h_true, h_pred, DELTA_T)

        # Store
        h_trues.append(h_true)
        h_preds.append(h_pred)
        t_norms.append(np.asarray(t_norm))
        matches[idx] = match_val
        m1s.append(m1)
        m2s.append(m2)
        qs.append(m2/m1)

    best_idx = int(np.argmax(matches))
    worst_idx = int(np.argmin(matches))

    def plot_comparison(idx, title, fname):
        # Compute analytic signals
        h_t = np.asarray(h_trues[idx])
        h_p = np.asarray(h_preds[idx])
        t   = np.asarray(t_norms[idx])
        an_t = hilbert(h_t); A_t = np.abs(an_t); ph_t = np.unwrap(np.angle(an_t))
        an_p = hilbert(h_p); A_p = np.abs(an_p); ph_p = np.unwrap(np.angle(an_p))

        # Raw differences
        dh_raw = h_p - h_t
        dA_raw = A_p - A_t
        dph_raw = ph_p - ph_t

        # Normalized absolute residuals [0,1] per-array
        dh = _norm01_abs(dh_raw)
        dA = _norm01_abs(dA_raw)
        dph = _norm01_abs(dph_raw)

        fig, axs = plt.subplots(2, 2, figsize=(18, 10))
        # Strain
        ax = axs[0, 0]
        ax.plot(t, h_t, label='True', linewidth=1)
        ax.plot(t, h_p, '--', label='Predicted', linewidth=1)
        ax.set_title('Strain Comparison')
        ax.set_ylabel('Strain')
        ax.grid(True)
        ax.legend()
        # Amplitude (unscaled analytic envelopes)
        ax = axs[0, 1]
        ax.plot(t, A_t, label='True Amp (unscaled)', linewidth=1)
        ax.plot(t, A_p, '--', label='Pred Amp (unscaled)', linewidth=1)
        ax.set_title('Amplitude Comparison (unscaled)')
        ax.set_ylabel('Amplitude')
        ax.grid(True)
        ax.legend()
        # Phase
        ax = axs[1, 0]
        ax.plot(t, ph_t, label='True Phase', linewidth=1)
        ax.plot(t, ph_p, '--', label='Pred Phase', linewidth=1)
        ax.set_title('Phase Comparison')
        ax.set_xlabel('Normalized Time')
        ax.set_ylabel('Phase (rad)')
        ax.grid(True)
        ax.legend()
        # Normalized Differences (0-1)
        ax = axs[1, 1]
        ax.plot(t, dh, label='|d Strain| (norm 0-1)', linewidth=1)
        ax.plot(t, dA, label='|d Amp| (norm 0-1)', linestyle='--', linewidth=1)
        ax.plot(t, dph, label='|d Phase| (norm 0-1)', linestyle=':', linewidth=1)
        ax.set_title('Normalized Differences (absolute, min-max → 0..1)')
        ax.set_xlabel('Normalized Time')
        ax.set_ylabel('Normalized residual')
        ax.grid(True)
        ax.legend(loc='upper right', fontsize='small')

        fig.suptitle(f"{title} (idx={idx}, match={matches[idx]:.4f})", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(plot_dir, fname), dpi=200)
        plt.close()

    # Best and worst overlays
    plot_comparison(best_idx, "Best match", "best_match_comparison.png")
    plot_comparison(worst_idx, "Worst match", "worst_match_comparison.png")

    # Scatter match vs q
    plt.figure(figsize=(16, 8), dpi=150)
    plt.scatter(qs, matches, c=matches, cmap='viridis', s=80)
    plt.xlabel('Mass ratio q = m2/m1')
    plt.ylabel('Match')
    plt.title('Match vs Mass Ratio')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "match_vs_q.png"), dpi=150)
    plt.close()

    # --- Smooth 3D surface that fills the whole x-y area ---
    # build grid over full rectangle and interpolate smoothly
    m1s_arr = np.asarray(m1s)
    m2s_arr = np.asarray(m2s)
    pts = np.vstack([m1s_arr, m2s_arr]).T
    vals = np.asarray(matches)

    # higher resolution grid for smoothness
    nx = ny = 400
    m1_lin = np.linspace(m1s_arr.min(), m1s_arr.max(), nx)
    m2_lin = np.linspace(m2s_arr.min(), m2s_arr.max(), ny)
    M1_grid, M2_grid = np.meshgrid(m1_lin, m2_lin)

    # cubic for smooth interior, nearest to fill NaNs at boundaries
    match_grid_cubic = griddata(pts, vals, (M1_grid, M2_grid), method='cubic')
    mask_nan = np.isnan(match_grid_cubic)
    if mask_nan.any():
        match_grid_nearest = griddata(pts, vals, (M1_grid, M2_grid), method='nearest')
        match_grid_cubic[mask_nan] = match_grid_nearest[mask_nan]
    match_grid = match_grid_cubic

    # Plot surface
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(M1_grid, M2_grid, match_grid, cmap='viridis', linewidth=0, antialiased=True)
    ax.set_xlabel('Mass m1')
    ax.set_ylabel('Mass m2')
    ax.set_zlabel('Match')
    ax.set_title('3D Surface: Masses vs Match (smooth)')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "3d_match_surface.png"), dpi=150)
    plt.close()

    return matches

def plot_prediction_uncertainty(
    predictor: WaveformPredictor,
    mass_1: float, mass_2: float,
    spin1_z: float, spin2_z: float,
    inclination: float, eccentricity: float,
    output_path: str = "plots/prediction_uncertainty.png"
):
    """
    Generate and save a plot of h(t) with its uncertainty band.
    """

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    logger.info(f"Plotting prediction with uncertainty to {output_path}")

    set_sigma_level = 3
    # Get waveform + uncertainty
    plus_strain, _ = predictor.predict_with_uncertainty(
        mass_1, mass_2,
        spin1_z, spin2_z,
        inclination, eccentricity,
        sigma_level=set_sigma_level
    )

    # Plot
    fig, ax = plt.subplots(figsize=(10,4))
    time = plus_strain.time
    mean = plus_strain.data
    sigma = plus_strain.uncertainty

    ax.plot(time, mean, label=f"Predicted $h_+(t)$", linewidth=1)
    ax.fill_between(time,
                    mean - sigma,
                    mean + sigma,
                    color=ax.lines[-1].get_color(),
                    alpha=0.3,
                    label=rf"$\pm{set_sigma_level}\sigma $band")
    ax.set_title(
        rf"Prediction $\pm1\sigma$ | $m_1$={mass_1:.1f}, $m_2$={mass_2:.1f}, "
        rf"$χ_1z$={spin1_z:.2f}, $χ_2z$={spin2_z:.2f}, incl={inclination:.2f}, ecc={eccentricity:.2f}"
    )
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Strain h₊")
    ax.legend(loc="best")
    ax.grid(True)

    # Save and close
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)

    logger.info(f"Saved uncertainty plot to {output_path}")

def generate_match_heatmap(MASS_MIN, MASS_MAX,
                           chi1z=0.0, chi2z=0.0,
                           incl=0.0, ecc=0.0,
                           output_path: str = "plots/heatmap.png"):
    """
    Generates and plots a *smooth* 2D heatmap of match values over (m1, m2),
    sampling both masses from MASS_MIN to MASS_MAX in steps of 5, then
    interpolating to a fine grid for display.
    """
    step = 5
    # coarse sampling grid
    m1_vals = np.arange(MASS_MIN, MASS_MAX + step, step)
    m2_vals = np.arange(MASS_MIN, MASS_MAX + step, step)

    pts_m1, pts_m2, pts_match = [], [], []
    pred = WaveformPredictor("checkpoints", model=MODEL, device=DEVICE)

    # Psd stuff
    flen = WAVEFORM_LENGTH // 2 + 1
    df = 1.0 / (WAVEFORM_LENGTH * DELTA_T)
    psd_vals = np.asarray(aLIGOZeroDetHighPower(flen, df, 20.0), dtype=float)

    # compute matches at coarse points
    for m1 in m1_vals:
        for m2 in m2_vals:
            # true waveform padded/truncated
            hp_t = make_waveform((m2,m2,chi1z,chi2z,incl,ecc))
            hp_true = np.asarray(hp_t.data[-int(WAVEFORM_LENGTH/DELTA_T):])

            # predicted waveform
            hp_p, _ = pred.predict(
                m1, m2, chi1z, chi2z, incl, ecc,
                waveform_length=WAVEFORM_LENGTH,
                sampling_dt=DELTA_T
            )
            hp_pred = np.asarray(hp_p.data[-int(WAVEFORM_LENGTH/DELTA_T):])

            match_val = compute_match(hp_true, hp_pred, DELTA_T)
            pts_m1.append(m1)
            pts_m2.append(m2)
            pts_match.append(match_val)

    # now build a fine grid for smooth plotting
    fine_n = 200
    m1_fine = np.linspace(MASS_MIN, MASS_MAX, fine_n)
    m2_fine = np.linspace(MASS_MIN, MASS_MAX, fine_n)
    m1_grid_f, m2_grid_f = np.meshgrid(m1_fine, m2_fine)

    # cubic interpolation onto fine grid
    match_grid_f = griddata(
        (pts_m1, pts_m2),
        pts_match,
        (m1_grid_f, m2_grid_f),
        method='cubic'
    )

    # plot
    plt.figure(figsize=(16, 8))
    im = plt.imshow(
        match_grid_f,
        extent=[MASS_MIN, MASS_MAX, MASS_MIN, MASS_MAX],
        origin='lower',
        cmap='viridis',
        aspect='auto',
    )
    plt.colorbar(im, label='Match Value')
    plt.xlabel('m1 [$M_\\odot$]')
    plt.ylabel('m2 [$M_\\odot$]')
    plt.title(f'Match Heatmap ({MASS_MIN}–{MASS_MAX} $M_\\odot$, coarse step={step})')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Saved smooth heatmap to {output_path}")

def _plot_overlay(h_true, h_pred, t, title, fname):
    """
    Overlay plots: strain, amplitude, phase, and *normalized* differences (0-1).
    The differences are plotted as absolute residuals min-max normalized per-array to [0,1].
    """
    an_t = hilbert(h_true); A_t = np.abs(an_t); ph_t = np.unwrap(np.angle(an_t))
    an_p = hilbert(h_pred); A_p = np.abs(an_p); ph_p = np.unwrap(np.angle(an_p))

    # Raw differences
    dh_raw = h_pred - h_true
    dA_raw = A_p - A_t
    dph_raw = ph_p - ph_t

    # Convert to absolute residuals and normalize each to [0,1] (per-array)
    def _norm01(x):
        x = np.asarray(x, dtype=float)
        x = np.abs(x)
        amin = x.min()
        amax = x.max()
        if amax <= amin or np.isclose(amax, amin):
            return np.zeros_like(x)
        return (x - amin) / (amax - amin)

    dh = _norm01(dh_raw)
    dA = _norm01(dA_raw)
    dph = _norm01(dph_raw)

    fig, axs = plt.subplots(2, 2, figsize=(18, 10))
    # Strain
    ax = axs[0, 0]
    ax.plot(t, h_true, label='True', linewidth=1)
    ax.plot(t, h_pred, '--', label='Predicted', linewidth=1)
    ax.set_title('Strain Comparison')
    ax.set_ylabel('Strain')
    ax.grid(True)
    ax.legend()
    # Amplitude
    ax = axs[0, 1]
    ax.plot(t, A_t, label='True Amp', linewidth=1)
    ax.plot(t, A_p, '--', label='Pred Amp', linewidth=1)
    ax.set_title('Amplitude Comparison')
    ax.set_ylabel('Amplitude')
    ax.grid(True)
    ax.legend()
    # Phase
    ax = axs[1, 0]
    ax.plot(t, ph_t, label='True Phase', linewidth=1)
    ax.plot(t, ph_p, '--', label='Pred Phase', linewidth=1)
    ax.set_title('Phase Comparison')
    ax.set_xlabel('Normalized Time')
    ax.set_ylabel('Phase (rad)')
    ax.grid(True)
    ax.legend()
    # Normalized Differences (0-1)
    ax = axs[1, 1]
    ax.plot(t, dh, label='|d Strain| (norm 0-1)', linewidth=1)
    ax.plot(t, dA, label='|d Amp| (norm 0-1)', linestyle='--', linewidth=1)
    ax.plot(t, dph, label='|d Phase| (norm 0-1)', linestyle=':', linewidth=1)
    ax.set_title('Normalized Differences (absolute, min-max -> 0..1)')
    ax.set_xlabel('Normalized Time')
    ax.set_ylabel('Normalized residual')
    ax.grid(True)
    ax.legend(loc='upper right', fontsize='small')

    fig.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(fname, dpi=200)
    plt.close()

def compare_surrogates_against_approximants(
    approximants: Sequence[str],
    surrogates: Dict[str, Any],
    samples: int = 200,
    batch_size: int = 128,
    out_dir: str = "plots/approximant_benchmark",
    use_tqdm: bool = True,
):
    """
    Run match benchmarks for multiple approximants and surrogate predictors.

    Plots:
      - Scatter plots of matches
      - Density plots (histogram + KDE) with 1σ, 2σ, 3σ vertical lines
      - Best & worst overlays (absolute residuals normalized 0-1)
    Stored in: plots/approximant_benchmark/<approximant>/<predictor>/
    """
    results = {}

    loop = tqdm(approximants, desc="approximants") if use_tqdm else approximants

    # PSD for match computation
    flen = WAVEFORM_LENGTH // 2 + 1
    df = 1.0 / (WAVEFORM_LENGTH * DELTA_T)
    psd_vals = np.asarray(aLIGOZeroDetHighPower(flen, df, 20.0), dtype=float)

    for approx in loop:
        dataset = generate_data(waveform=approx, samples=samples, clean=True)
        L = dataset.time_unscaled.size
        t_norm = getattr(dataset, "t_norm_array", dataset.time_unscaled)

        amps_norm = dataset.targets_A.reshape(samples, L)
        phis = dataset.targets_phi.reshape(samples, L)
        thetas = dataset.thetas[:samples]

        results[approx] = {}

        for name, predictor in surrogates.items():
            # Create per-comparison output dir
            comp_dir = os.path.join(out_dir, approx, name)
            os.makedirs(comp_dir, exist_ok=True)

            # Determine amplitude scale
            amp_scale = getattr(predictor, "amp_scale", getattr(dataset, "amp_scale", 1.0))

            # Unscale amplitude
            try:
                amp_true_mat = np.asarray(unscale_target(amps_norm, amp_scale)).reshape(samples, L)
            except Exception:
                amp_true_mat = np.stack([
                    np.asarray(unscale_target(amps_norm[i], amp_scale)).reshape(-1)
                    for i in range(samples)
                ], axis=0)

            # Construct true strain
            h_true = amp_true_mat * np.cos(phis)

            # Predict strain
            h_pred = None
            if hasattr(predictor, "batch_predict"):
                try:
                    out = predictor.batch_predict(thetas, batch_size=batch_size)
                    h_list = out[0] if isinstance(out, tuple) else out
                    if isinstance(h_list, np.ndarray):
                        h_pred = h_list
                    else:
                        h_pred = np.stack([
                            np.asarray(getattr(h, "data", h)).reshape(-1) for h in h_list
                        ], axis=0)
                except Exception:
                    pass

            if h_pred is None:
                h_pred = []
                rng = tqdm(range(samples), desc=f"{approx}:{name}", leave=False) if use_tqdm else range(samples)
                for i in rng:
                    theta = thetas[i]
                    if hasattr(predictor, "predict_debug"):
                        _, amp_pred, phi_pred = predictor.predict_debug(*theta)
                        h_i = np.asarray(amp_pred) * np.cos(np.asarray(phi_pred))
                    else:
                        out = predictor.predict(*theta)
                        hs = out[0] if isinstance(out, tuple) else out
                        h_i = np.asarray(getattr(hs, "data", hs)).reshape(-1)
                    h_pred.append(h_i)
                h_pred = np.stack(h_pred, axis=0)

            # Align lengths
            if h_pred.shape[1] != L:
                if h_pred.shape[1] > L:
                    h_pred = h_pred[:, :L]
                else:
                    pad = np.zeros((samples, L - h_pred.shape[1]))
                    h_pred = np.concatenate([h_pred, pad], axis=1)

            # Compute matches
            matches = np.array([compute_match(h_true[i], h_pred[i], DELTA_T) for i in range(samples)])
            results[approx][name] = matches

            # Scatter plot
            plt.figure(figsize=(10, 4))
            plt.scatter(np.arange(samples), matches, s=12)
            plt.xlabel("Sample idx")
            plt.ylabel("Match")
            plt.title(f"{approx} / {name} — Match Scatter")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(comp_dir, "match_scatter.png"), dpi=150)
            plt.close()

            # Density plot with 1σ, 2σ, 3σ lines
            mean_val = np.mean(matches)
            std_val = np.std(matches)
            plt.figure(figsize=(8, 4))
            plt.hist(matches, bins=30, density=True, alpha=0.6, label="Histogram")
            if len(matches) > 1:
                kde = gaussian_kde(matches)
                xs = np.linspace(np.min(matches), np.max(matches), 500)
                plt.plot(xs, kde(xs), linewidth=2, label="KDE")
            for n in [1, 2, 3]:
                plt.axvline(mean_val + n * std_val, color="r", linestyle=":", linewidth=1,
                            label=f"+{n}σ" if n == 1 else None)
                plt.axvline(mean_val - n * std_val, color="r", linestyle=":", linewidth=1)
            plt.axvline(mean_val, color="k", linestyle="--", linewidth=1, label="Mean")
            plt.title(f"{approx} / {name} — Match Distribution")
            plt.xlabel("Match")
            plt.ylabel("Density")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(comp_dir, "match_density.png"), dpi=150)
            plt.close()

            # Best & worst overlays
            best_idx = int(np.argmax(matches))
            worst_idx = int(np.argmin(matches))
            for idx, label in [(best_idx, "best"), (worst_idx, "worst")]:
                m1, m2, s1z, s2z, inc, ecc = map(float, thetas[idx])
                params_str = f"m1={m1:.6g}, m2={m2:.6g}, s1z={s1z:.3f}, s2z={s2z:.3f}, inc={inc:.3f}, ecc={ecc:.3f}"
                title = f"{approx} / {name} — {label} match={matches[idx]:.4f} | {params_str}"
                fname = os.path.join(comp_dir, f"{label}_overlay_idx{idx}.png")
                _plot_overlay(h_true[idx], h_pred[idx], t_norm, title, fname)

    # Summary DataFrame
    rows = []
    for approx, d in results.items():
        for name, arr in d.items():
            rows.append({
                "approximant": approx,
                "predictor": name,
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "25%": float(np.percentile(arr, 25)),
                "50%": float(np.median(arr)),
                "75%": float(np.percentile(arr, 75)),
                "max": float(np.max(arr)),
                "n": len(arr),
            })
    summary_df = pd.DataFrame(rows).set_index(["approximant", "predictor"]).sort_index()

    return results, summary_df

if __name__ == "__main__":
    # Logging
    os.makedirs("logs", exist_ok=True)

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("logs/evaluation.log", mode='a'),
        ]
    )

    # Stopping the voices
    logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)
    logging.getLogger("matplotlib.ticker").setLevel(logging.WARNING)

    evaluate()

    matches = cross_correlation()

    generate_match_heatmap(25,125)

    approximants = ["SEOBNRv4", "IMRPhenomD", "SEOBNRv4HM", "IMRPhenomXHM"]
    surrogates = {
        "IMR Model": WaveformPredictor("checkpoints", model="IMR", device=DEVICE),
        "EOB model": WaveformPredictor("checkpoints", model="EOB", device=DEVICE),
    }

    results, summary_df = compare_surrogates_against_approximants(
        approximants, surrogates, samples=1000, batch_size=128, out_dir="plots/approximants"
    )

    print(summary_df)
