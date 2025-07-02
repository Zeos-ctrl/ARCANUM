# General Utils
import logging
import numpy as np
from scipy.signal import hilbert
from dataclasses import dataclass

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
    targets_A: np.ndarray      # (N_total, 1)
    targets_phi: np.ndarray    # (N_total, 1)
    time_unscaled: np.ndarray  # (WAVEFORM_LENGTH,)
    thetas: np.ndarray         # (NUM_SAMPLES, 6)
    A_peaks: np.ndarray        # (NUM_SAMPLES,)
    amp_norm: np.ndarray       # (NUM_SAMPLES, WAVEFORM_LENGTH)
    phi_unwrap: np.ndarray     # (NUM_SAMPLES, WAVEFORM_LENGTH)
    param_means: np.ndarray    # (6,)
    param_stds: np.ndarray     # (6,)
    theta_norm: np.ndarray     # (NUM_SAMPLES, 6)
    t_norm_array: np.ndarray   # (WAVEFORM_LENGTH,)

def sample_parameters(n):
    logger.debug(f"Sampling {n} parameter sets...")
    rng = np.random.default_rng(0)
    lows  = [MASS_MIN, MASS_MIN, SPIN_MIN, SPIN_MIN, INCLINATION_MIN, ECC_MIN]
    highs = [MASS_MAX, MASS_MAX, SPIN_MAX, SPIN_MAX, INCLINATION_MAX, ECC_MAX]
    samples = rng.uniform(lows, highs, size=(n, 6))
    logger.debug(f"Sampled parameters shape: {samples.shape}")
    return samples

def make_noisy_waveform(theta, psd_arr, seed=None):
    logger.debug(f"Generating waveform with theta={theta}, seed={seed}")
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
    logger.debug(f"Raw waveform length: {L}")

    if L >= WAVEFORM_LENGTH:
        h_cut = h_plus[-WAVEFORM_LENGTH:]
        logger.debug("Waveform truncated.")
    else:
        pad_amt = WAVEFORM_LENGTH - L
        h_cut = np.pad(h_plus, (pad_amt, 0), mode="constant")
        logger.debug(f"Waveform padded by {pad_amt} samples.")

    noise_td = noise.noise_from_psd(WAVEFORM_LENGTH, DELTA_T, psd_arr, seed=seed).numpy()
    return h_cut + noise_td

def generate_data() -> GeneratedDataset:
    logger.info("Generating dataset...")
    
    thetas = sample_parameters(NUM_SAMPLES)
    logger.debug("Computing PSD...")
    
    delta_f = 1.0 / (WAVEFORM_LENGTH * DELTA_T)
    psd_arr = aLIGOZeroDetHighPower(WAVEFORM_LENGTH, delta_f, F_LOWER)

    all_A_peak = np.zeros(NUM_SAMPLES)
    all_amp_norm = np.zeros((NUM_SAMPLES, WAVEFORM_LENGTH))
    all_phi_unwrap = np.zeros((NUM_SAMPLES, WAVEFORM_LENGTH))

    for i in range(NUM_SAMPLES):
        logger.debug(f"Generating waveform {i + 1}/{NUM_SAMPLES}")
        h_noisy = make_noisy_waveform(thetas[i], psd_arr, seed=i)
        A_peak = np.max(np.abs(h_noisy)) + 1e-30
        analytic = hilbert(h_noisy)
        inst_amp = np.abs(analytic)
        inst_phi = np.unwrap(np.angle(analytic))

        all_A_peak[i] = A_peak
        all_amp_norm[i] = inst_amp / A_peak
        all_phi_unwrap[i] = inst_phi

    logger.info("Waveform generation complete. Normalizing inputs...")

    time_unscaled = np.linspace(-WAVEFORM_LENGTH * DELTA_T, 0.0, WAVEFORM_LENGTH)
    t_norm_array = 2.0 * (time_unscaled - time_unscaled.min()) / \
                   (time_unscaled.max() - time_unscaled.min()) - 1.0

    param_means = thetas.mean(axis=0)
    param_stds = thetas.std(axis=0)
    theta_norm = (thetas - param_means) / param_stds
    logger.debug(f"Parameter normalization: means={param_means}, stds={param_stds}")

    S, L = NUM_SAMPLES, WAVEFORM_LENGTH
    t_grid = np.broadcast_to(t_norm_array, (S, L))
    theta_grid = np.broadcast_to(theta_norm[:, None, :], (S, L, 6))
    inputs = np.concatenate([t_grid[..., None], theta_grid], axis=-1).reshape(-1, 7)

    targets_A = all_amp_norm.reshape(-1, 1)
    targets_phi = all_phi_unwrap.reshape(-1, 1)

    logger.info("Dataset generation complete.")
    logger.debug(f"Inputs shape: {inputs.shape}, Targets shape: {targets_A.shape}, {targets_phi.shape}")

    return GeneratedDataset(
        inputs=inputs.astype(np.float32),
        targets_A=targets_A.astype(np.float32),
        targets_phi=targets_phi.astype(np.float32),
        time_unscaled=time_unscaled,
        thetas=thetas,
        A_peaks=all_A_peak,
        amp_norm=all_amp_norm,
        phi_unwrap=all_phi_unwrap,
        param_means=param_means,
        param_stds=param_stds,
        theta_norm=theta_norm,
        t_norm_array=t_norm_array,
    )
