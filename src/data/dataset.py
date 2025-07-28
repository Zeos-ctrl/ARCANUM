# General Utils
import sys
import psutil
import logging
import numpy as np
from scipy.stats import qmc
from scipy.signal import hilbert
from dataclasses import dataclass
from scipy.signal.windows import tukey
from scipy.fft import fft, ifft
from scipy.interpolate import interp1d
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

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

def sizeof_numpy_array(arr):
    return arr.nbytes

def sizeof_tensor(t):
    return t.element_size() * t.nelement()

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

def _resample_and_fill(h: np.ndarray, target_len: int) -> np.ndarray:
    """
    Trim leading/trailing zeros from h, then interpolate the 
    non-zero segment uniformly back to length `target_len`.
    """
    # find non-zero region
    nz = np.where(h != 0)[0]
    if nz.size == 0:
        # all zeros? just return zeros
        return np.zeros(target_len, dtype=h.dtype)

    start, end = nz[0], nz[-1]
    segment = h[start:end+1]

    # build old & new sample grids
    old_x = np.linspace(0, 1, len(segment))
    new_x = np.linspace(0, 1, target_len)

    # linear interpolation (extrapolate just in case)
    f = interp1d(old_x, segment, kind='linear', fill_value='extrapolate')
    return f(new_x)


def make_waveform(theta):
    """Generate a *clean* waveform of exactly WAVEFORM_LENGTH samples."""
    m1, m2, chi1z, chi2z, incl, ecc = theta
    hp, _ = get_td_waveform(
        mass1=m1, mass2=m2,
        spin1z=chi1z, spin2z=chi2z,
        inclination=incl, eccentricity=ecc,
        delta_t=DELTA_T, f_lower=F_LOWER,
        approximant=WAVEFORM
    )
    h_plus = hp.numpy()

    # now trim & resample onto the full window
    return _resample_and_fill(h_plus, WAVEFORM_LENGTH)


def make_noisy_waveform(theta, psd_arr, snr_target, seed=None):
    """
    Generate a noisy waveform of exactly WAVEFORM_LENGTH samples,
    trimmed & resampled to eliminate zero padding artifacts.
    """
    # 1) raw clean waveform (un‐padded)
    m1, m2, chi1z, chi2z, incl, ecc = theta
    hp, _ = get_td_waveform(
        mass1=m1, mass2=m2,
        spin1z=chi1z, spin2z=chi2z,
        inclination=incl, eccentricity=ecc,
        delta_t=DELTA_T, f_lower=F_LOWER,
        approximant=WAVEFORM
    )
    h_clean = hp.numpy()

    # 2) whiten & scale to target SNR  
    Hf      = fft(h_clean)
    sqrt_psd = np.sqrt(psd_arr) + 1e-30
    Hf_white = Hf / sqrt_psd
    h_white  = np.real(ifft(Hf_white))

    rho_clean = np.sqrt(np.sum(h_white**2) * DELTA_T)
    scale     = (snr_target / rho_clean) if rho_clean > 0 else 1.0
    h_scaled  = h_clean * scale

    # 3) add noise
    noise_td = noise.noise_from_psd(
        len(h_scaled), DELTA_T, psd_arr, seed=seed
    ).numpy()
    h_noisy = h_scaled + noise_td

    # 4) trim & resample
    return _resample_and_fill(h_noisy, WAVEFORM_LENGTH)

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

    dataset = GeneratedDataset(
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

    # logging of RAM footprint
    arrays = [
        dataset.inputs,
        dataset.targets_A,
        dataset.targets_phi,
        dataset.thetas,
        dataset.phi_unwrap
    ]
    total_bytes = sum(sizeof_numpy_array(a) for a in arrays)
    logger.info(f" Dataset in‐memory size: {total_bytes/1024**3:.3f} GB "
                f"({total_bytes/1024**2:.1f} MB)")

    proc = psutil.Process(os.getpid())
    rss = proc.memory_info().rss
    logger.info(f" → Process RSS after data gen: {rss/1024**3:.3f} GB")

    return dataset

def pick_batch_size(X, A, phi, safety=0.1, max_cap=None):
    """Return the largest batch size that fits in (safety * free GPU mem)."""
    # per‐sample byte footprint
    N = X.size(0)
    bytes_per = (
        X.element_size()*X.nelement() +
        A.element_size()*A.nelement() +
        phi.element_size()*phi.nelement()
    ) / N

    props    = torch.cuda.get_device_properties(0)
    total    = props.total_memory
    reserved = torch.cuda.memory_reserved(0)
    free_mem = total - reserved

    batch = int((free_mem * safety) // bytes_per)
    if max_cap is not None:
        batch = min(batch, max_cap)
    return max(batch, 1)

def save_dataset(data, path='dataset.pt'):
    """
    Save the entire `data` object to disk.
    """
    torch.save(data, path)

def load_dataset(path='dataset.pt', device='cuda'):
    """
    Load back the full `data` object exactly as it was.
    If you need the tensors on GPU, we map_location accordingly.
    """
    data = torch.load(path, map_location='cpu', weights_only=False)
    # move arrays onto GPU
    if DEVICE == 'cuda':
        data.inputs      = torch.from_numpy(data.inputs).to(device)
        data.targets_A   = torch.from_numpy(data.targets_A).to(device)
        data.targets_phi = torch.from_numpy(data.targets_phi).to(device)
    return data

def make_loaders(data):
    """Generate train/val loaders for amplitude & phase."""
    X = torch.from_numpy(data.inputs).to(DEVICE)      # (N_total,7)
    A = torch.from_numpy(data.targets_A).to(DEVICE)   # (N_total,1)
    phi = torch.from_numpy(data.targets_phi).to(DEVICE)  # (N_total,1)

    bytes_X   = sizeof_tensor(X)
    bytes_A   = sizeof_tensor(A)
    bytes_phi = sizeof_tensor(phi)
    logger.info(f" -> GPU tensors allocated:"
                f"  X={bytes_X/1024**2:.1f} MB,"
                f"  A={bytes_A/1024**2:.1f} MB,"
                f"  phi={bytes_phi/1024**2:.1f} MB")

    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated(DEVICE)
        reserved = torch.cuda.memory_reserved(DEVICE)
        BS = pick_batch_size(X, A, phi, safety=0.01, max_cap=2046)
        logger.info(f" -> torch.cuda memory: allocated={used/1024**3:.3f} GB,"
                    f" reserved={reserved/1024**3:.3f} GB,"
                    f" using BS={BS}")
    else:
        BS = 64
        logger.info(f"Using default batch size of {BS} for cpu...")

    idx = list(range(X.size(0)))
    train_idx, val_idx = train_test_split(
        idx, test_size=VAL_SPLIT,
        random_state=RANDOM_SEED, shuffle=True
    )

    train_ds_amp = TensorDataset(X[train_idx], A[train_idx])
    val_ds_amp   = TensorDataset(X[val_idx],   A[val_idx])
    train_ds_phi = TensorDataset(X[train_idx], phi[train_idx])
    val_ds_phi   = TensorDataset(X[val_idx],   phi[val_idx])
    train_ds_joint = TensorDataset(X[train_idx], A[train_idx], phi[train_idx])
    val_ds_joint   = TensorDataset(X[val_idx],   A[val_idx], phi[val_idx])

    loaders = {
        'amp': {
            'train': DataLoader(train_ds_amp,   batch_size=BS, shuffle=True),
            'val':   DataLoader(val_ds_amp,     batch_size=BS, shuffle=False)
        },
        'phase': {
            'train': DataLoader(train_ds_phi,   batch_size=BS, shuffle=True),
            'val':   DataLoader(val_ds_phi,     batch_size=BS, shuffle=False)
        },
        'joint': {
            'train': DataLoader(train_ds_joint, batch_size=BS, shuffle=True),
            'val':   DataLoader(val_ds_joint,   batch_size=BS, shuffle=False)
        }
    }
    return loaders

