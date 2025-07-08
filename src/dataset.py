# General Utils
import logging
import numpy as np
from scipy.stats import qmc
from scipy.signal import hilbert
from dataclasses import dataclass
from scipy.signal.windows import tukey

# PyCBC imports
from pycbc import noise
from pycbc.waveform import get_td_waveform
from pycbc.psd import aLIGOZeroDetHighPower

# Libraries
from src.config import *
from src.model import PhaseDNN_Full, AmplitudeNet

logger = logging.getLogger(__name__)

@dataclass
class GeneratedDataset:
    inputs: np.ndarray         # (N_total, 7)
    targets_A: np.ndarray      # (N_total, 1) normalized log10 amplitude
    targets_phi: np.ndarray    # (N_total, 1)
    time_unscaled: np.ndarray  # (WAVEFORM_LENGTH,)
    thetas: np.ndarray         # (NUM_SAMPLES, 6)
    log_amp_min: float         # global minimum of log10 amplitude
    log_amp_max: float         # global maximum of log10 amplitude
    phi_unwrap: np.ndarray     # (NUM_SAMPLES, WAVEFORM_LENGTH)
    param_means: np.ndarray    # (6,)
    param_stds: np.ndarray     # (6,)
    theta_norm: np.ndarray     # (NUM_SAMPLES, 6)
    t_norm_array: np.ndarray   # (WAVEFORM_LENGTH,)

def sample_parameters(n, seed=None, method="lhs"):
    """
    Sample n parameter sets from either Latin Hypercube Sampling (default) or uniform random sampling.

    Args:
        n (int): Number of samples.
        seed (int or None): Random seed for reproducibility.
        method (str): Sampling method, either "lhs" (default) or "uniform".

    Returns:
        np.ndarray: Sampled parameters, shape (n, 6)
    """
    logger.debug(f"Sampling {n} parameter sets using {method} method...")

    lows  = np.array([MASS_MIN, MASS_MIN, SPIN_MIN, SPIN_MIN, INCLINATION_MIN, ECC_MIN])
    highs = np.array([MASS_MAX, MASS_MAX, SPIN_MAX, SPIN_MAX, INCLINATION_MAX, ECC_MAX])
    dim   = 6

    rng = np.random.default_rng(seed)

    if method == "lhs":
        sampler = qmc.LatinHypercube(d=dim, seed=seed)
        sample = sampler.random(n)
        samples = qmc.scale(sample, lows, highs)
    elif method == "uniform":
        samples = rng.uniform(lows, highs, size=(n, dim))
    else:
        raise ValueError(f"Unknown sampling method: {method}")

    logger.debug(f"Sampled parameters shape: {samples.shape}")
    return samples

def make_waveform(theta):
    """Generate a clean (noise-free) waveform for parameters theta."""
    logger.debug(f"Generating clean waveform with theta={theta}")
    m1, m2, chi1z, chi2z, incl, ecc = theta
    hp, _ = get_td_waveform(
        mass1=m1, mass2=m2,
        spin1z=chi1z, spin2z=chi2z,
        inclination=incl, eccentricity=ecc,
        delta_t=DELTA_T, f_lower=F_LOWER,
        approximant=WAVEFORM
    )
    h_plus = hp.numpy()
    L = len(h_plus)
    if L >= WAVEFORM_LENGTH:
        return h_plus[-WAVEFORM_LENGTH:]
    else:
        pad_amt = WAVEFORM_LENGTH - L
        return np.pad(h_plus, (pad_amt, 0), mode="constant")

def make_noisy_waveform(theta, psd_arr, seed=None):
    """Generate a noisy waveform by adding PSD-based noise to the clean waveform."""
    logger.debug(f"Generating noisy waveform with theta={theta}")
    h_clean = make_waveform(theta)
    noise_td = noise.noise_from_psd(WAVEFORM_LENGTH, DELTA_T, psd_arr, seed=seed).numpy()
    return h_clean + noise_td


def generate_data(clean: bool = True, samples: int = NUM_SAMPLES, alpha: float = 0.1) -> GeneratedDataset:
    """
    Generate dataset of (log-scaled) amplitude & phase.
    If clean=True, use noise-free waveforms; otherwise include PSD noise.
    """
    logger.info(f"Generating {'clean' if clean else 'noisy'} dataset with log-scaling...")
    thetas = sample_parameters(samples)
    
    # Precompute window
    window = tukey(WAVEFORM_LENGTH, alpha=alpha)

    # Compute PSD for noisy option
    delta_f = 1.0 / (WAVEFORM_LENGTH * DELTA_T)
    psd_arr = aLIGOZeroDetHighPower(WAVEFORM_LENGTH, delta_f, F_LOWER)

    # Allocate arrays
    all_log_amp = np.zeros((samples, WAVEFORM_LENGTH))
    all_phi_unwrap = np.zeros((samples, WAVEFORM_LENGTH))
    eps = 1e-30

    logger.info(f"Loading {WAVEFORM} waveform profile...")

    for i in range(samples):
        # Choose waveform generator
        if clean:
            h = make_waveform(thetas[i])
        else:
            h = make_noisy_waveform(thetas[i], psd_arr, seed=i)

        # Find the nonzero region
        nz = np.where(np.abs(h) > 0)[0]
        if nz.size > 0:
            start, end = nz[0], nz[-1] + 1
            seg_len = end - start

            # Build a segment‑length Tukey and embed it into a full‑length mask
            window = np.zeros(WAVEFORM_LENGTH)
            window[start:end] = tukey(seg_len, alpha=alpha)

            # Apply it (zeros stay zeros, nonzero region fades in/out)
            h = h * window

        analytic = hilbert(h)
        inst_amp = np.abs(analytic) + eps  # avoid log(0)
        all_log_amp[i] = np.log10(inst_amp)
        all_phi_unwrap[i] = np.unwrap(np.angle(analytic))

    # Determine global min/max in log-space
    log_amp_min = all_log_amp.min()
    log_amp_max = all_log_amp.max()
    logger.debug(f"Log10 amplitude range: [{log_amp_min:.3f}, {log_amp_max:.3f}]")

    # Normalize log amplitudes to [0,1]
    all_log_amp_norm = (all_log_amp - log_amp_min) / (log_amp_max - log_amp_min)

    # Time normalization [-1,1]
    time_unscaled = np.linspace(-WAVEFORM_LENGTH * DELTA_T, 0.0, WAVEFORM_LENGTH)
    t_norm_array = 2.0 * (time_unscaled - time_unscaled.min()) / \
                   (time_unscaled.max() - time_unscaled.min()) - 1.0

    # Parameter normalization
    param_means = thetas.mean(axis=0)
    param_stds = thetas.std(axis=0)
    theta_norm = (thetas - param_means) / param_stds
    logger.debug(f"Parameter normalization: means={param_means}, stds={param_stds}")

    # Build inputs
    S, L = samples, WAVEFORM_LENGTH
    t_grid = np.broadcast_to(t_norm_array, (S, L))
    theta_grid = np.broadcast_to(theta_norm[:, None, :], (S, L, 6))
    inputs = np.concatenate([t_grid[..., None], theta_grid], axis=-1).reshape(-1, 7)

    # Targets
    targets_A = all_log_amp_norm.reshape(-1, 1)
    targets_phi = all_phi_unwrap.reshape(-1, 1)

    logger.info("Dataset generation complete.")

    return GeneratedDataset(
        inputs=inputs.astype(np.float32),
        targets_A=targets_A.astype(np.float32),
        targets_phi=targets_phi.astype(np.float32),
        time_unscaled=time_unscaled,
        thetas=thetas,
        log_amp_min=log_amp_min,
        log_amp_max=log_amp_max,
        phi_unwrap=all_phi_unwrap,
        param_means=param_means,
        param_stds=param_stds,
        theta_norm=theta_norm,
        t_norm_array=t_norm_array,
    )
