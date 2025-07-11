# General utils
import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
import matplotlib.ticker as mticker

# PyCBC waveform
from pycbc.waveform import get_td_waveform

# Libraries
from src.config import *
from src.dataset import generate_data
from src.utils import compute_match, WaveformPredictor, notify_discord

logger = logging.getLogger(__name__)

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
    pred = WaveformPredictor("checkpoints", device=DEVICE)
    data = generate_data(samples=10)

    i = np.random.randint(0, 10)
    m1, m2, chi1z, chi2z, incl, ecc = data.thetas[i]
    title_str = (f"m1={m1:.1f}, m2={m2:.1f}, "
                 f"χ1z={chi1z:.2f}, χ2z={chi2z:.2f}, "
                 f"incl={incl:.2f}, ecc={ecc:.2f}")
    logger.info(f"Evaluating for parameters: {title_str}")

    # time grid & true targets
    time = data.time_unscaled
    L = len(time)
    amp_true_norm = data.targets_A.reshape(10, L)[i]
    phi_true      = data.targets_phi.reshape(10, L)[i]
    amp_true      = pred.inverse_log_norm(amp_true_norm)
    h_true        = amp_true * np.cos(phi_true)

    # model prediction
    t_norm, amp_pred, phi_pred = pred.predict_debug(m1, m2, chi1z, chi2z, incl, ecc)
    h_pred      = amp_pred * np.cos(phi_pred)

    # wrapped phase and residuals
    phi_wr_true = np.mod(phi_true, 2*np.pi)
    phi_wr_pred = np.mod(phi_pred, 2*np.pi)
    # wrap residual
    dphi = phi_pred - phi_true
    phi_res_wrapped = (dphi + np.pi) % (2*np.pi) - np.pi

    # set up figure: 2 rows × 4 cols
    fig, axes = plt.subplots(2, 4, figsize=(24, 10), sharex=True)
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

def cross_correlation_fixed_q(
    q_list=(1.0, 1.5, 2.0, 2.5),
    chi1z=0.0, chi2z=0.0,
    incl=0.0, ecc=0.0
):
    logger.info("Running waveform cross-correlation vs mass ratio q.")
    plot_dir = "plots/cross_correlation"
    os.makedirs(plot_dir, exist_ok=True)

    data = generate_data(samples=10)
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
        t_norm, amp_pred, phi_pred = pred.predict_debug(
            m1,m2,c1z,c2z,incl_i,ecc_i
        )
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
        ax.legend(loc="upper left")

        # Amplitude
        ax = axs[row,1]
        ax.plot(t_norm, A_true, label="True Amp", linewidth=1)
        ax.plot(t_norm, A_pred, '--', label="Predicted Amp", linewidth=1)
        if row==0: ax.set_title("Amplitude")
        ax.legend(loc="upper left")

        # Phase
        ax = axs[row,2]
        ax.plot(t_norm, phi_true, label="True Phase", linewidth=1)
        ax.plot(t_norm, phi_pred, '--', label="Predicted Phase", linewidth=1)
        ax.set_yscale("log")
        if row==0: ax.set_title("Phase")
        ax.legend(loc="upper left")

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

def polar():
    logger.info("Plotting same parameters over 0.25s, 0.5s, and 1.0s durations")

    pred = WaveformPredictor("checkpoints", device=DEVICE)

    # pick one parameter set, here first entry
    data = generate_data(samples=1)
    m1, m2, chi1z, chi2z, incl, ecc = data.thetas[0]
    logger.info(f"Using params m1={m1:.1f}, m2={m2:.1f}, chi1z={chi1z:.2f}, "
                f"chi2z={chi2z:.2f}, incl={incl:.2f}, ecc={ecc:.2f}")

    # base sampling interval from checkpoint
    dt = pred.delta_t

    # desired durations (seconds)
    durations = [0.25, 0.5, 1.0]

    # prepare figure: 3 rows, 2 cols
    fig, axs = plt.subplots(3, 2, figsize=(12, 10), sharex=False)
    fig.suptitle("Waveforms at 0.25 s, 0.5 s, 1.0 s Durations", fontsize=16)

    for row, duration in enumerate(durations):
        # compute number of samples
        L = int(np.round(duration / dt))

        # get waveform
        h_plus, h_cross = pred.predict(
            m1, m2, chi1z, chi2z, incl, ecc,
            waveform_length=L,
            sampling_dt=dt
        )

        # plus
        ax_p = axs[row, 0]
        ax_p.plot(h_plus.time, h_plus.data, linewidth=1)
        ax_p.set_ylabel("h₊")
        ax_p.set_title(f"Duration = {duration:.2f} s ({L} samples)")
        ax_p.grid(True)

        # cross
        ax_c = axs[row, 1]
        ax_c.plot(h_cross.time, h_cross.data, linewidth=1, color="C1")
        ax_c.set_ylabel("hₓ")
        ax_c.set_title(f"Duration = {duration:.2f} s ({L} samples)")
        ax_c.grid(True)

        # only bottom row gets x-label
        if row == len(durations)-1:
            ax_p.set_xlabel("Time [s]")
            ax_c.set_xlabel("Time [s]")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs("plots", exist_ok=True)
    outfile = os.path.join("plots", "waveforms_varying_duration.png")
    plt.savefig(outfile)
    logger.info(f"Saved varying‐duration waveforms to {outfile}")

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
    matches = cross_correlation_fixed_q()
    polar()

#    notify_discord(
#            f"Evaluation complete! cross correlation matches: {matches}\n"
#    )
