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
from src.data.config import *
from src.data.dataset import generate_data
from src.utils.utils import compute_match, WaveformPredictor, notify_discord

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
    match, _ = compute_match(h_true, h_pred)

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


def cross_correlation_fixed_q(
    q_list=(1.0, 1.5, 2.0, 2.5),
    chi1z=0.0, chi2z=0.0,
    incl=0.0, ecc=0.0
):
    """
    For each mass ratio q in q_list:
      - Select waveform from dataset closest to q
      - Compute true and predicted strains
      - Compute match (cross-correlation statistic)
      - Plot waveform grid (strain, amplitude, phase)
      - Plot symmetric mass ratio vs match
    Returns:
      matches: list of match values per q
    """
    logger.info("Running waveform cross-correlation vs mass ratio q.")
    plot_dir = "plots/cross_correlation"
    os.makedirs(plot_dir, exist_ok=True)

    # Load data and predictor
    data = generate_data()
    pred = WaveformPredictor("checkpoints", device=DEVICE)

    qs, sym_masses, matches = [], [], []
    h_trues, h_preds, t_norms = [], [], []
    L = data.time_unscaled.size

    # Precompute mass ratios and total masses
    thetas = data.thetas
    mass_ratios = thetas[:,1] / thetas[:,0]  # m2/m1
    total_masses = thetas[:,0] + thetas[:,1]  # m1 + m2
    sym_ratio = (thetas[:,0] * thetas[:,1]) / (total_masses**2)  # symmetric mass ratio η

    for q in q_list:
        # Find index closest to desired q
        idx = np.argmin(np.abs(mass_ratios - q))
        m1, m2, c1z, c2z, incl_i, ecc_i = thetas[idx]

        # True waveform reconstruction
        amp_true_norm = data.targets_A.reshape(-1, L)[idx]
        phi_true_arr   = data.targets_phi.reshape(-1, L)[idx]
        log_rec        = amp_true_norm * (data.log_amp_max - data.log_amp_min) + data.log_amp_min
        amp_true       = 10**log_rec
        h_true         = amp_true * np.cos(phi_true_arr)

        # Model prediction
        t_norm, amp_pred, phi_pred = pred.predict_debug(m1, m2, c1z, c2z, incl_i, ecc_i)
        h_pred = amp_pred * np.cos(phi_pred)

        # Compute match (normalized cross-correlation)
        match, _ = compute_match(h_true, h_pred)

        # Record
        qs.append(q)
        matches.append(match)
        sym_masses.append((m1 * m2) / ((m1 + m2)**2))
        h_trues.append(h_true)
        h_preds.append(h_pred)
        t_norms.append(t_norm)

    # Plot waveform grid: strain, amplitude, phase
    K = len(q_list)
    fig, axs = plt.subplots(K, 3, figsize=(18, 4*K), sharex=True)
    for row, (q, h_true, h_pred, t_norm) in enumerate(zip(qs, h_trues, h_preds, t_norms)):
        # Analytic signals
        analytic_true = hilbert(h_true)
        A_true = np.abs(analytic_true)
        phi_true = np.unwrap(np.angle(analytic_true))

        analytic_pred = hilbert(h_pred)
        A_pred = np.abs(analytic_pred)
        phi_pred = np.unwrap(np.angle(analytic_pred))

        # Strain
        ax = axs[row, 0]
        ax.plot(t_norm, h_true, label="True", linewidth=1)
        ax.plot(t_norm, h_pred, '--', label="Predicted", linewidth=1)
        if row == 0:
            ax.set_title("Strain")
        ax.set_ylabel(f"q={q:.1f}")
        ax.legend(loc="upper left")

        # Amplitude
        ax = axs[row, 1]
        ax.plot(t_norm, A_true, label="True Amp", linewidth=1)
        ax.plot(t_norm, A_pred, '--', label="Pred Amp", linewidth=1)
        if row == 0:
            ax.set_title("Amplitude")
        ax.legend(loc="upper left")

        # Phase
        ax = axs[row, 2]
        ax.plot(t_norm, phi_true, label="True Phase", linewidth=1)
        ax.plot(t_norm, phi_pred, '--', label="Pred Phase", linewidth=1)
        ax.set_yscale("log")
        if row == 0:
            ax.set_title("Phase")
        ax.legend(loc="upper left")

    for ax in axs[-1, :]:
        ax.set_xlabel("Normalized time")

    plt.tight_layout()
    grid_path = os.path.join(plot_dir, "waveform_grid_fixed_q.png")
    plt.savefig(grid_path)
    plt.close()
    logger.info("Saved waveform grid to %s", grid_path)

    # Plot symmetric mass ratio vs match
    plt.figure(figsize=(8, 5))
    plt.scatter(sym_masses, matches)
    plt.xlabel(r"Symmetric mass ratio $\eta = m_1 m_2/(m_1 + m_2)^2$")
    plt.ylabel("Match")
    plt.title("Match vs Symmetric Mass Ratio")
    plt.grid(True)
    plt.tight_layout()
    scatter_path = os.path.join(plot_dir, "match_vs_symmetric_mass.png")
    plt.savefig(scatter_path)
    plt.close()
    logger.info("Saved symmetric mass vs match plot to %s", scatter_path)

    return matches

def polar():
    logger.info("Plotting same parameters over 0.25s, 0.5s, and 1.0s durations")

    pred = WaveformPredictor("checkpoints", device=DEVICE)

    # pick one parameter set, here first entry
    data = generate_data(samples=5)
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
        ax_p.set_ylabel("h+")
        ax_p.set_title(f"Duration = {duration:.2f} s ({L} samples)")
        ax_p.grid(True)

        # cross
        ax_c = axs[row, 1]
        ax_c.plot(h_cross.time, h_cross.data, linewidth=1, color="C1")
        ax_c.set_ylabel("hx")
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
#    matches = cross_correlation_fixed_q()
    polar()

#    notify_discord(
#            f"Evaluation complete! cross correlation matches: {matches}\n"
#    )
