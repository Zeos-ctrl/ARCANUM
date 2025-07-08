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
from src.utils import compute_match, WaveformPredictor, notify_discord

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
    L = len(time)

    for row, i in enumerate(indices):
        # Unpack parameters and get true log-norm amplitude & phase
        m1, m2, c1z, c2z, incl, ecc = data.thetas[i]
        amp_true_norm = data.targets_A.reshape(NUM_SAMPLES, L)[i]
        phi_true      = data.targets_phi.reshape(NUM_SAMPLES, L)[i]

        # Predict normalized log-amplitude and phase
        t_norm, amp_pred_norm, phi_pred = pred.predict(m1, m2, c1z, c2z, incl, ecc)

        # Inverse-log-normalize amplitudes into physical units
        amp_pred = pred.inverse_log_norm(amp_pred_norm)
        amp_true = pred.inverse_log_norm(amp_true_norm)

        # Reconstruct strains
        h_pred = amp_pred * np.cos(phi_pred)
        h_true = amp_true * np.cos(phi_true)

        # Plot Strain
        ax = axs[row, 0]
        ax.plot(time, h_true, label="True $h_+(t)$", linewidth=1)
        ax.plot(time, h_pred, "--", label="Predicted $h_+(t)$", linewidth=1)
        if row == 0:
            ax.set_title("Strain")
        ax.set_ylabel("Strain")
        ax.legend(loc="upper right")

        # Plot Amplitude
        ax = axs[row, 1]
        ax.plot(time, amp_true, label="True Amplitude", linewidth=1)
        ax.plot(time, amp_pred, "--", label="Predicted Amplitude", linewidth=1)
        if row == 0:
            ax.set_title("Amplitude")
        ax.set_ylabel("Amplitude")
        ax.legend(loc="upper right")

        # Plot Phase
        ax = axs[row, 2]
        ax.plot(time, phi_true, label="True Phase", linewidth=1)
        ax.plot(time, phi_pred, "--", label="Predicted Phase", linewidth=1)
        if row == 0:
            ax.set_title("Phase")
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

    data = generate_data(clean=True)
    pred = WaveformPredictor("checkpoints", device=DEVICE)

    qs, matches = [], []
    h_trues, h_preds, t_norms = [], [], []
    L = data.time_unscaled.size

    # Precompute ratios in the dataset
    mass_ratios = data.thetas[:,1] / data.thetas[:,0]  # m2/m1 for each sample

    for q in q_list:
        # Find the dataset index whose m2/m1 is closest to q
        idx = np.argmin(np.abs(mass_ratios - q))
        m1, m2, c1z, c2z, incl_i, ecc_i = data.thetas[idx]

        # Recover the true envelope + phase from stored targets:
        amp_true_norm = data.targets_A.reshape(-1, L)[idx]
        phi_true      = data.targets_phi.reshape(-1, L)[idx]

        # Invert log‐min‐max → physical amplitude
        log_rec    = amp_true_norm * (data.log_amp_max - data.log_amp_min) \
                     + data.log_amp_min
        amp_true   = 10**log_rec
        h_true     = amp_true * np.cos(phi_true)

        # Model prediction & inversion
        t_norm, amp_pred_norm, phi_pred = pred.predict(
            m1,m2,c1z,c2z,incl_i,ecc_i
        )
        amp_pred = pred.inverse_log_norm(amp_pred_norm)
        h_pred   = amp_pred * np.cos(phi_pred)

        # Compute match
        match = compute_match(h_true, h_pred)
        qs.append(q)
        matches.append(match)

        # store for plotting
        h_trues.append(h_true)
        h_preds.append(h_pred)
        t_norms.append(t_norm)

    # plotting grid of strain / amplitude / phase
    K = len(q_list)
    _, axs = plt.subplots(K, 3, figsize=(18, 4*K), sharex=True)
    for row, (q, h_true, h_pred, t_norm) in enumerate(zip(qs, h_trues, h_preds, t_norms)):
        # true envelope + phase
        analytic_true = hilbert(h_true)
        A_true = np.abs(analytic_true)
        phi_true = np.unwrap(np.angle(analytic_true))

        analytic_pred = hilbert(h_pred)
        A_pred = np.abs(analytic_pred)
        phi_pred = np.unwrap(np.angle(analytic_pred))

        # Strain
        ax = axs[row,0]
        ax.plot(t_norm, h_true, label="True", linewidth=1)
        ax.plot(t_norm, h_pred, '--', label="Predicted", linewidth=1)
        if row==0: ax.set_title("Strain")
        ax.set_ylabel(f"q={q:.1f}")
        ax.legend(loc="upper right")

        # Amplitude
        ax = axs[row,1]
        ax.plot(t_norm, A_true, label="True Amp", linewidth=1)
        ax.plot(t_norm, A_pred, '--', label="Predicted Amp", linewidth=1)
        if row==0: ax.set_title("Amplitude")
        ax.legend(loc="upper right")

        # Phase
        ax = axs[row,2]
        ax.plot(t_norm, phi_true, label="True Phase", linewidth=1)
        ax.plot(t_norm, phi_pred, '--', label="Predicted Phase", linewidth=1)
        if row==0: ax.set_title("Phase")
        ax.legend(loc="upper right")

    for ax in axs[-1,:]:
        ax.set_xlabel("Normalized time")

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "waveform_grid_fixed_q.png"))
    plt.close()
    logger.info("Saved waveform grid plot.")

    # match vs q scatter
    plt.figure(figsize=(8,5))
    plt.scatter(qs, matches)
    plt.xlabel(r"$q=m_2/m_1$")
    plt.ylabel("Match")
    plt.ylim(min(matches),1)
    plt.title("Waveform Match vs Mass Ratio")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "cross_correlation_fixed_q.png"))
    plt.close()
    logger.info("Saved match vs q plot.")

    return matches
  
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

    evaluate()
    matches = cross_correlation_fixed_q()

    notify_discord(
            f"Evaluation complete! cross correlation matches: {matches}\n"
    )
