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
from src.data.dataset import unscale_target

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
    amp_true      = unscale_target(amp_true_norm, pred.amp_scale)
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

def cross_correlation(samples=10, checkpoint_dir="checkpoints", device=DEVICE):
    """
    Generate 'samples' waveforms, predict them, compute per-pair cross-correlation (true[i] vs pred[i]),
    and plot:
      - Grid comparison (strain, amplitude, phase) for best and worst matching pairs
      - Scatter of match vs mass ratio q
      - 3D scatter of (m1, m2, match)

    Saves plots in 'plots/cross_correlation_pairs'.
    Returns:
      matches: np.ndarray of shape (samples,) with match values for each pair.
    """
    # Prepare output directory
    plot_dir = "plots/cross_correlation"
    os.makedirs(plot_dir, exist_ok=True)

    # Load data and model
    data = generate_data(samples=samples)
    pred = WaveformPredictor(checkpoint_dir, device=device)

    thetas = data.thetas
    L = data.time_unscaled.size

    # Containers
    h_trues, h_preds, t_norms = [], [], []
    qs, m1s, m2s = [], [] ,[]
    matches = np.zeros(samples)

    # Generate, predict and match each sample
    for idx in range(samples):
        m1, m2, c1z, c2z, incl_i, ecc_i = thetas[idx]
        # True waveform
        amp_norm = data.targets_A.reshape(-1, L)[idx]
        phi_arr  = data.targets_phi.reshape(-1, L)[idx]
        amp_true = unscale_target(amp_norm, pred.amp_scale)
        h_true   = amp_true * np.cos(phi_arr)
        # Prediction
        t_norm, amp_pred, phi_pred = pred.predict_debug(m1, m2, c1z, c2z, incl_i, ecc_i)
        h_pred = amp_pred * np.cos(phi_pred)

        # Compute match
        match_val, _ = compute_match(h_true, h_pred)

        # Store
        h_trues.append(h_true)
        h_preds.append(h_pred)
        t_norms.append(t_norm)
        matches[idx] = match_val
        m1s.append(m1)
        m2s.append(m2)
        qs.append(m2/m1)

    best_idx = np.argmax(matches)
    worst_idx = np.argmin(matches)

    def plot_comparison(idx, title, fname):
        # Compute analytic signals
        h_t, h_p, t = h_trues[idx], h_preds[idx], t_norms[idx]
        an_t = hilbert(h_t); A_t = np.abs(an_t); ph_t = np.unwrap(np.angle(an_t))
        an_p = hilbert(h_p); A_p = np.abs(an_p); ph_p = np.unwrap(np.angle(an_p))
        # Differences
        dh = h_p - h_t
        dA = A_p - A_t
        dph = ph_p - ph_t

        fig, axs = plt.subplots(2, 2, figsize=(18, 10))
        # Strain
        ax = axs[0, 0]
        ax.plot(t, h_t, label='True', linewidth=1)
        ax.plot(t, h_p, '--', label='Predicted', linewidth=1)
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
        # Difference
        ax = axs[1, 1]
        ax.plot(t, dh, label='d Strain', linewidth=1)
        ax.plot(t, dA, label='d Amplitude', linestyle='--', linewidth=1)
        ax.plot(t, dph, label='d Phase', linestyle=':', linewidth=1)
        ax.set_title('Differences')
        ax.set_xlabel('Normalized Time')
        ax.set_ylabel('Difference')
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

    from scipy.interpolate import griddata

    m1_grid = np.linspace(min(m1s), max(m1s), num=100)
    m2_grid = np.linspace(min(m2s), max(m2s), num=100)
    m1_grid, m2_grid = np.meshgrid(m1_grid, m2_grid)

    match_grid = griddata(
        (m1s, m2s), matches, (m1_grid, m2_grid), method='linear'
    )

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(
        m1_grid, m2_grid, match_grid, cmap='viridis', edgecolor='none'
    )

    ax.set_xlabel('Mass m1')
    ax.set_ylabel('Mass m2')
    ax.set_zlabel('Match')
    ax.set_title('3D Surface: Masses vs Match')

    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "3d_match_surface.png"), dpi=150)
    plt.close()

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
    matches = cross_correlation()
    polar()

#    notify_discord(
#            f"Evaluation complete! cross correlation matches: {matches}\n"
#    )
