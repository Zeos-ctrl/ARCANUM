# General utils
import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert

# PyCBC waveform
from pycbc.waveform import get_td_waveform

# Libraries
from src.config import *
from src.dataset import generate_data
from src.utils import compute_match, WaveformPredictor

logger = logging.getLogger(__name__)

def evaluate():
    logger.info("Starting waveform evaluation and visualization...")
    pred = WaveformPredictor("checkpoints", device=DEVICE)
    data = generate_data()

    logger.debug("Dataset generated. Selecting random samples for plotting.")
    K = 3
    indices = np.random.choice(NUM_SAMPLES, K, replace=False)
    logger.debug(f"Selected indices: {indices.tolist()}")

    _, axs = plt.subplots(K, 3, figsize=(18, 4*K), sharex=True)
    time = data.time_unscaled

    for row, i in enumerate(indices):
        m1, m2, c1z, c2z, incl, ecc = data.thetas[i]
        logger.debug(f"Sample {i}: m1={m1}, m2={m2}, spin1z={c1z}, spin2z={c2z}, incl={incl}, ecc={ecc}")

        A_peak = data.A_peaks[i]
        amp_true = data.amp_norm[i]
        phi_true = data.phi_unwrap[i]

        t_norm, amp_pred_norm, phi_pred = pred.predict(m1, m2, c1z, c2z, incl, ecc)

        h_true = A_peak * amp_true * np.cos(phi_true)
        h_pred = A_peak * amp_pred_norm * np.cos(phi_pred)

        A_true = A_peak * amp_true
        A_pred = A_peak * amp_pred_norm

        # Strain
        ax = axs[row, 0]
        ax.plot(time, h_true, label="True $h_+(t)$", linewidth=1)
        ax.plot(time, h_pred, "--", label="Predicted $h_+(t)$", linewidth=1)
        if row == 0: ax.set_title("Strain")
        ax.set_ylabel("Strain")
        ax.legend(loc="upper right")

        # Amplitude
        ax = axs[row, 1]
        ax.plot(time, A_true, label="True Amplitude", linewidth=1)
        ax.plot(time, A_pred, "--", label="Predicted Amplitude", linewidth=1)
        if row == 0: ax.set_title("Amplitude")
        ax.set_ylabel("Amplitude")
        ax.legend(loc="upper right")

        # Phase
        ax = axs[row, 2]
        ax.plot(time, phi_true, label="True Phase", linewidth=1)
        ax.plot(time, phi_pred, "--", label="Predicted Phase", linewidth=1)
        if row == 0: ax.set_title("Phase")
        ax.set_ylabel("Phase [rad]")
        ax.legend(loc="upper right")

    for ax in axs[-1, :]:
        ax.set_xlabel("Time [s]")

    plt.tight_layout()
    output_dir = "plots"
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "waveform_plots.png")
    plt.savefig(out_file)
    logger.info(f"Saved waveform plots to {out_file}")


def cross_correlation_fixed_q(
    q_list=(1.0, 1.5, 2.0, 2.5),
    chi1z=0.0, chi2z=0.0,
    incl=0.0, ecc=0.0
):
    logger.info("Running waveform cross-correlation vs mass ratio q.")
    plot_dir = "plots/cross_correlation"
    os.makedirs(plot_dir, exist_ok=True)

    pred = WaveformPredictor("checkpoints", device=DEVICE)
    qs, matches = [], []

    # Prepare storage for waveforms
    h_trues, h_preds, t_norms = [], [], []

    for q in q_list:
        m1 = MASS_MIN
        m2 = min(q * m1, MASS_MAX)
        logger.debug(f"Evaluating q={q:.2f}: m1={m1}, m2={m2}")

        # Generate true waveform
        hp, _ = get_td_waveform(
            mass1=m1, mass2=m2,
            spin1z=chi1z, spin2z=chi2z,
            inclination=incl, eccentricity=ecc,
            delta_t=DELTA_T, f_lower=F_LOWER,
            approximant=WAVEFORM
        )
        h_plus = hp.numpy()
        if len(h_plus) >= WAVEFORM_LENGTH:
            h_true = h_plus[-WAVEFORM_LENGTH:]
        else:
            pad = WAVEFORM_LENGTH - len(h_plus)
            h_true = np.pad(h_plus, (pad, 0), mode='constant')

        # Prediction
        t_norm, amp_pred_n, phi_pred = pred.predict(m1, m2, chi1z, chi2z, incl, ecc)
        A_peak = np.max(np.abs(h_true)) + 1e-30
        h_pred = A_peak * amp_pred_n * np.cos(phi_pred)

        # Compute match
        match = compute_match(h_true, h_pred)
        logger.debug(f"Match for q={q:.2f}: {match:.4f}")
        qs.append(q)
        matches.append(match)

        # Store for plotting grid
        h_trues.append(h_true)
        h_preds.append(h_pred)
        t_norms.append(t_norm)

    # Plot grid of Strain / Amplitude / Phase for each q
    K = len(q_list)
    _, axs = plt.subplots(K, 3, figsize=(18, 4*K), sharex=True)
    for row, (q, h_true, h_pred, t_norm) in enumerate(zip(qs, h_trues, h_preds, t_norms)):
        # Analytic signals for amplitude & phase
        true_analytic = hilbert(h_true)
        pred_analytic = hilbert(h_pred)
        A_true = np.abs(true_analytic)
        phi_true = np.unwrap(np.angle(true_analytic))
        A_pred = np.abs(pred_analytic)
        phi_pred = np.unwrap(np.angle(pred_analytic))

        # Strain
        ax = axs[row, 0]
        ax.plot(t_norm, h_true, label="True", linewidth=1)
        ax.plot(t_norm, h_pred, '--', label="Predicted", linewidth=1)
        if row == 0: ax.set_title("Strain")
        ax.set_ylabel(f"q={q:.1f}")
        ax.legend(loc="upper right")

        # Amplitude
        ax = axs[row, 1]
        ax.plot(t_norm, A_true, label="True Amp", linewidth=1)
        ax.plot(t_norm, A_pred, '--', label="Predicted Amp", linewidth=1)
        if row == 0: ax.set_title("Amplitude")
        ax.legend(loc="upper right")

        # Phase
        ax = axs[row, 2]
        ax.plot(t_norm, phi_true, label="True Phase", linewidth=1)
        ax.plot(t_norm, phi_pred, '--', label="Predicted Phase", linewidth=1)
        if row == 0: ax.set_title("Phase")
        ax.legend(loc="upper right")

    for ax in axs[-1, :]:
        ax.set_xlabel("Normalized time")

    plt.tight_layout()
    grid_path = os.path.join(plot_dir, "waveform_grid_fixed_q.png")
    plt.savefig(grid_path)
    plt.close()
    logger.info(f"Saved waveform grid plot: {grid_path}")

    # Match vs q
    qs = np.array(qs)
    matches = np.array(matches)
    plt.figure(figsize=(10, 6))
    plt.scatter(qs, matches)
    plt.xlabel(r'Mass ratio $q = m_2/m_1$')
    plt.ylabel('Match')
    plt.ylim(0, 1)
    plt.title('Waveform Match vs Mass Ratio')
    plt.grid(True)
    plt.tight_layout()
    match_plot_path = os.path.join(plot_dir, "cross_correlation_fixed_q.png")
    plt.savefig(match_plot_path)
    plt.close()
    logger.info(f"Saved match vs mass ratio plot: {match_plot_path}")
    
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

    evaluate()
    cross_correlation_fixed_q()
