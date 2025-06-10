import numpy as np
from scipy.signal import hilbert
from pycbc.waveform import get_td_waveform
from pycbc.detector import Detector

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

    # 1) Generate time‐domain waveform
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

    # 2) Detector antenna patterns at merger
    det = Detector(detector_name)
    Fp, Fx = det.antenna_pattern(ra, dec, psi_fixed, 0.0)

    # 3) Detector‐frame strain on PyCBC grid
    h_det_pycbc = (Fp * h_plus + Fx * h_cross).astype(np.float32)

    # 4) Resample/pad onto `common_times`
    h_true_common = np.zeros_like(common_times, dtype=np.float32)
    idxs = np.round((t_plus - common_times[0]) / delta_t).astype(int)
    valid = (idxs >= 0) & (idxs < len(common_times))
    h_true_common[idxs[valid]] = h_det_pycbc[valid]

    return h_true_common

def compute_laplace_hessians(train_loader, phase_model, amp_model, lambda_A=1e-6, lambda_phi=1e-6):
    """
    For the amplitude network: let 'feature_extractor_A' be all layers up to but not
    including amp_model.linear_out. Compute C_A = sum(feats.T @ feats) across train_loader,
    then H_A = C_A + lambda_A * I, and Σ_A = inv(H_A).

    For the phase network (with N_banks): for each bank i, let phi_i(x) = [t_norm; θ_embed(x)],
    so collect C_i = sum(phi_i_batch.T @ phi_i_batch), H_i = C_i + lambda_phi * I, Σ_i = inv(H_i).

    Returns:
      Σ_A:          np.ndarray of shape (d_A, d_A)
      Σ_phase_list: list of length N_banks, each a (d_phase, d_phase) np.ndarray
    """
    import torch

    # --- Amplitude network ---
    amp_body_layers = list(amp_model.net_body.children())
    amp_last_linear = amp_model.linear_out  # nn.Linear
    feature_extractor_A = torch.nn.Sequential(*amp_body_layers).to(amp_model.linear_out.weight.device)
    d_A = amp_last_linear.weight.shape[1]

    C_A = np.zeros((d_A, d_A), dtype=np.float64)
    with torch.no_grad():
        for x_batch, _ in train_loader:
            feats = feature_extractor_A(x_batch).cpu().numpy()  # (batch, d_A)
            C_A += feats.T.dot(feats)

    H_A = C_A + lambda_A * np.eye(d_A, dtype=np.float64)
    Σ_A = np.linalg.inv(H_A)

    # --- Phase network ---
    emb_dim = phase_model.theta_embed(torch.zeros(1, 15)).shape[-1]
    d_phase = emb_dim + 1

    Σ_phase_list = []
    with torch.no_grad():
        for bank in range(phase_model.N_banks):
            C_i = np.zeros((d_phase, d_phase), dtype=np.float64)
            for x_batch, _ in train_loader:
                # x_batch[:,0:1] is t_norm; x_batch[:,1:] is θ_norm
                t_b = x_batch[:, 0:1].cpu().numpy()                          # (batch, 1)
                θ_embed = phase_model.theta_embed(x_batch[:, 1:]).cpu().numpy()  # (batch, emb_dim)
                phi_i = np.concatenate([t_b, θ_embed], axis=1)               # (batch, d_phase)
                C_i += phi_i.T.dot(phi_i)
            H_i = C_i + lambda_phi * np.eye(d_phase, dtype=np.float64)
            Σ_i = np.linalg.inv(H_i)
            Σ_phase_list.append(Σ_i)

    return Σ_A, Σ_phase_list
