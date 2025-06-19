import numpy as np
import torch

from scipy.signal import hilbert
from scipy.interpolate import interp1d

from pycbc.waveform import get_td_waveform
from pycbc.detector import Detector
from pycbc.filter import overlap
from pycbc.psd import aLIGOZeroDetHighPower
from pycbc.types import TimeSeries

from src.config import DELTA_T

def trimming_indices(h_arr: np.ndarray, buffer: float, delta_t: float) -> (int, int):
    """
    Given a 1D strain array `h_arr` sampled on a uniform time grid with spacing `delta_t`,
    find the first and last nonzero‐like sample indices, then extend by `buffer` seconds
    on each side (clamped to array bounds).
    Returns (start_idx, end_idx), such that h_arr[start_idx:end_idx] covers the active region.
    """
    nonzero = np.where(np.abs(h_arr) > 1e-25)[0]
    if len(nonzero) == 0:
        return 0, len(h_arr)
    buffer_idx = int(buffer / delta_t)
    start_idx = max(nonzero[0] - buffer_idx, 0)
    end_idx = min(nonzero[-1] + buffer_idx + 1, len(h_arr))  # +1 so slice is inclusive
    return start_idx, end_idx

def compute_param_stats(thetas: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Given `thetas` of shape (num_samples, 15), compute per‐column mean and stddev.
    Returns (means, stds), each of shape (15,).
    """
    means = thetas.mean(axis=0)
    stds = thetas.std(axis=0)
    return means.astype(np.float32), stds.astype(np.float32)

def normalize_theta(theta: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    """
    Given a single raw parameter vector `theta` (shape (15,)) or an array of them
    (shape (N, 15)), returns normalized values (same shape) via (theta - means) / stds.
    """
    return ((theta - means) / stds).astype(np.float32)

def generate_pycbc_waveform(params: tuple,
                            common_times: np.ndarray,
                            delta_t: float,
                            waveform_name: str,
                            detector_name: str,
                            psi_fixed: float) -> np.ndarray:
    """
    Given a 15‐tuple `params = (m1,m2,S1x,S1y,S1z,S2x,S2y,S2z,incl,ecc,ra,dec,dist,t0,phi0)`,
    generates the plus/cross polarizations via PyCBC, projects onto the detector,
    and resamples/pads onto the fixed `common_times` grid.

    Returns:
      h_true_common: np.ndarray of length len(common_times)
    """
    (m1, m2,
     S1x, S1y, S1z,
     S2x, S2y, S2z,
     incl, ecc,
     ra, dec,
     d, t0, phi0) = params

    # Generate time‐domain waveform
    hp, hc = get_td_waveform(
        mass1             = m1,
        mass2             = m2,
        spin1x            = S1x,
        spin1y            = S1y,
        spin1z            = S1z,
        spin2x            = S2x,
        spin2y            = S2y,
        spin2z            = S2z,
        eccentricity      = ecc,
        inclination       = incl,
        distance          = d,
        coalescence_time  = t0,
        coalescence_phase = phi0,
        delta_t           = delta_t,
        f_lower           = 20.0,
        approximant       = waveform_name
    )
    h_plus = hp.numpy().astype(np.float32)
    h_cross = hc.numpy().astype(np.float32)
    t_plus = hp.sample_times.numpy().astype(np.float32)

    # Detector antenna patterns at merger
    det = Detector(detector_name)
    Fp, Fx = det.antenna_pattern(ra, dec, psi_fixed, 0.0)

    # Detector‐frame strain on PyCBC grid
    h_det_pycbc = (Fp * h_plus + Fx * h_cross).astype(np.float32)

    # Resample/pad onto `common_times`
    h_true_common = np.zeros_like(common_times, dtype=np.float32)
    idxs = np.round((t_plus - common_times[0]) / delta_t).astype(int)
    valid = (idxs >= 0) & (idxs < len(common_times))
    h_true_common[idxs[valid]] = h_det_pycbc[valid]

    return h_true_common

def reconstruct_waveform(
    model: torch.nn.Module,
    params_norm: torch.Tensor,  # (N,15)
    t_phys: torch.Tensor,       # (N,1)
    A_peak: float
):
    """
    Runs the model and returns:
      h_pred    : (N,) reconstructed strain
      A_pred    : (N,) predicted amp
      phi_pred  : (N,) integrated phase
      omega_pred: (N,) instantaneous freq
    """
    model.eval()
    with torch.no_grad():
        A_t, phi_rate_t, omega_t = model(params_norm, t_phys)

    A_pred     = A_t.cpu().numpy().ravel()
    phi_rate   = phi_rate_t.cpu().numpy().ravel()
    omega_pred = omega_t.cpu().numpy().ravel()

    # integrate phase
    phi_pred = np.cumsum(phi_rate) * DELTA_T

    # reconstruct strain
    h_pred = A_peak * A_pred * np.cos(phi_pred)

    print(f"Strain: {h_pred}, Amp: {A_pred}, Phase: {phi_pred}, Freq:{omega_pred}")

    return h_pred, A_pred, phi_pred, omega_pred

def compute_match(h1, h2, delta_t, f_lower):
    """
    Compute the overlap (match) between two real strains h1,h2:
      O = max_{t0,phi0} <h1|h2> / sqrt(<h1|h1><h2|h2>)
    using pycbc.overlap.
    """
    # Force double precision
    h1_d = np.array(h1, dtype=np.float64)
    h2_d = np.array(h2, dtype=np.float64)

    ts1 = TimeSeries(h1_d, delta_t=delta_t)
    ts2 = TimeSeries(h2_d, delta_t=delta_t)

    npts    = len(ts1)
    delta_f = ts1.delta_f

    # Build PSD in double
    psd = aLIGOZeroDetHighPower(npts, delta_f, low_freq_cutoff=f_lower)

    # Compute overlap (max over time & phase)
    m = overlap(
        ts1, ts2,
        psd=psd,
        low_frequency_cutoff=f_lower,
        high_frequency_cutoff=None
    )
    return abs(m)
