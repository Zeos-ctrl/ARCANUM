# General Utils
import logging
import numpy as np
from scipy.stats import qmc
from scipy.signal import hilbert
from dataclasses import dataclass
from scipy.signal.windows import tukey
from scipy.fft import fft, ifft

# PyCBC imports
from pycbc import noise
from pycbc.types import TimeSeries, FrequencySeries
from pycbc.waveform import get_td_waveform
from pycbc.psd import aLIGOZeroDetHighPower

# Libraries
from src.data.config import *

logger = logging.getLogger(__name__)

@dataclass
class GeneratedDataset:
    inputs: np.ndarray        # (N_total, 6)  ← was 7
    targets_A: np.ndarray     # (N_total, 1)
    targets_phi: np.ndarray   # (N_total, 1)
    time_unscaled: np.ndarray # (L,)
    thetas: np.ndarray        # raw thetas (N,6), for reference
    log_amp_min: float
    log_amp_max: float
    phi_unwrap: np.ndarray    # (N,L)
    param_means: np.ndarray   # (5,)
    param_stds: np.ndarray    # (5,)
    theta_norm: np.ndarray    # (N,5)
    t_norm_array: np.ndarray  # (L,)

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

def make_noisy_waveform(theta, psd_arr, snr_target, seed=None):
    """
    Generate a noisy waveform at (approximate) SNR=snr_target,
    but ensure we always use WAVEFORM_LENGTH samples.
    """
    # get exactly WAVEFORM_LENGTH data via your helper
    h_clean = make_waveform(theta)  # now shape == (WAVEFORM_LENGTH,)

    # manual whiten in freq domain
    Hf = fft(h_clean)
    sqrt_psd = np.sqrt(psd_arr) + 1e-30
    Hf_white = Hf / sqrt_psd
    h_white  = np.real(ifft(Hf_white))

    # compute native whitened -> SNR
    rho_clean = np.sqrt(np.sum(h_white**2) * DELTA_T)
    scale     = (snr_target/rho_clean) if rho_clean>0 else 1.0

    # scale the **clean** fixed‑length waveform
    h_scaled = h_clean * scale

    # add PSD‐matched noise
    noise_td = noise.noise_from_psd(
        WAVEFORM_LENGTH, DELTA_T, psd_arr, seed=seed
    ).numpy()

    return h_scaled + noise_td

def generate_data(
    clean: bool = CLEAN,
    samples: int = NUM_SAMPLES,
    alpha: float = 0.1,
    snr_min: float = SNR_MIN,
    snr_max: float = SNR_MAX
) -> GeneratedDataset:
    logger.info(f"Generating {'clean' if clean else 'noisy'} dataset...")
    thetas = sample_parameters(samples)  # (N,6)

    # enforce zero‐spin if that feature’s off
    if "effective_spin" not in TRAIN_FEATURES:
        thetas[:,2] = 0.0
        thetas[:,3] = 0.0
    if "inclination" not in TRAIN_FEATURES:
        thetas[:,4] = 0.01
    if "eccentricity" not in TRAIN_FEATURES:
        thetas[:,5] = 0.0

    # compute derived features map
    m1, m2, chi1z, chi2z, incl, ecc = thetas.T
    derived_map = {
        "chirp_mass":           (m1*m2)**(3/5) / (m1+m2)**(1/5),
        "symmetric_mass_ratio": (m1*m2) / (m1+m2)**2,
        "effective_spin":       (m1*chi1z + m2*chi2z) / (m1+m2),
        "inclination":          incl,
        "eccentricity":         ecc
    }
    D = len(TRAIN_FEATURES)
    thetas_D = np.stack([derived_map[f] for f in TRAIN_FEATURES], axis=1)

    # Prepare PSD both as numpy array and PyCBC TimeSeries
    delta_f = 1.0 / (WAVEFORM_LENGTH * DELTA_T)
    psd_arr = aLIGOZeroDetHighPower(WAVEFORM_LENGTH, delta_f, F_LOWER)

    all_log_amp    = np.zeros((samples, WAVEFORM_LENGTH))
    all_phi_unwrap = np.zeros((samples, WAVEFORM_LENGTH))
    eps = 1e-30

    for i in range(samples):
        theta_raw = thetas[i]
        if clean:
            # no noise
            hp_seg = make_waveform(theta_raw)
        else:
            # pick a random SNR in [snr_min, snr_max]
            snr_target = float(np.random.uniform(snr_min, snr_max))
            hp_seg = make_noisy_waveform(theta_raw, psd_arr, snr_target, seed=i)

        # taper with Tukey
        nz = np.where(np.abs(hp_seg)>0)[0]
        if nz.size:
            start, end = nz[0], nz[-1]+1
            window = np.zeros(WAVEFORM_LENGTH)
            window[start:end] = tukey(end-start, alpha=alpha)
            hp_seg = hp_seg * window

        # analytic → inst amp & phase
        analytic     = hilbert(hp_seg)
        inst_amp     = np.abs(analytic) + eps
        all_log_amp[i]    = np.log10(inst_amp)
        all_phi_unwrap[i] = np.unwrap(np.angle(analytic))

    # normalize amp -> [0,1]
    log_amp_min = all_log_amp.min()
    log_amp_max = all_log_amp.max()
    all_log_amp_norm = (all_log_amp - log_amp_min)/(log_amp_max-log_amp_min)

    # time grids
    time_unscaled = np.linspace(-WAVEFORM_LENGTH*DELTA_T, 0.0, WAVEFORM_LENGTH)
    t_norm = 2*(time_unscaled - time_unscaled.min())/(time_unscaled.max()-time_unscaled.min()) - 1

    # normalize derived features
    param_means = thetas_D.mean(axis=0)
    param_stds  = thetas_D.std(axis=0)
    theta_norm  = (thetas_D - param_means)/param_stds

    # build inputs (flatten)
    t_grid     = np.broadcast_to(t_norm, (samples, WAVEFORM_LENGTH))
    theta_grid = np.broadcast_to(theta_norm[:,None,:], (samples, WAVEFORM_LENGTH, D))
    inputs     = np.concatenate([t_grid[...,None], theta_grid], axis=-1).reshape(-1,1+D)

    targets_A   = all_log_amp_norm.reshape(-1,1)
    targets_phi = all_phi_unwrap.reshape(-1,1)

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
        t_norm_array=t_norm,
    )
