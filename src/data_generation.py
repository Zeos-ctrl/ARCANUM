import numpy as np
from scipy.signal import hilbert
from pycbc.waveform import get_td_waveform
from pycbc.detector import Detector

from src.config import (
    WAVEFORM_NAME, DELTA_T, F_LOWER, DETECTOR_NAME, PSI_FIXED,
    T_BEFORE, T_AFTER,
    MASS_MIN, MASS_MAX, SPIN_MAG_MIN, SPIN_MAG_MAX,
    INCL_MIN, INCL_MAX, ECC_MIN, ECC_MAX,
    RA_MIN, RA_MAX, DEC_MIN, DEC_MAX,
    DIST_MIN, DIST_MAX, COAL_MIN, COAL_MAX,
    NUM_SAMPLES, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE,
    PATIENCE, FINE_TUNE_EPOCHS, FINE_TUNE_LR,
    DEVICE, CHECKPOINT_DIR
)

def sample_parameters(num_samples):
    """
    Sample `num_samples` random (m1, m2, spins, extrinsic) parameter sets.
    Returns:
      param_list: list of length num_samples, each a 15‐tuple
                  (m1, m2, S1x, S1y, S1z, S2x, S2y, S2z, incl, ecc, ra, dec, dist, t0, phi0)
      thetas:     np.ndarray of shape (num_samples, 15) with raw floats
    """
    param_list = []
    thetas = np.zeros((num_samples, 15), dtype=np.float32)

    for i in range(num_samples):
        # Intrinsic parameters
        m1 = np.random.uniform(MASS_MIN, MASS_MAX)
        m2 = np.random.uniform(MASS_MIN, MASS_MAX)

        chi1 = np.random.uniform(SPIN_MAG_MIN, SPIN_MAG_MAX)
        chi2 = np.random.uniform(SPIN_MAG_MIN, SPIN_MAG_MAX)

        cos_tilt1 = np.random.uniform(-1, 1)
        theta1    = np.arccos(cos_tilt1)
        phi1      = np.random.uniform(0, 2*np.pi)

        cos_tilt2 = np.random.uniform(-1, 1)
        theta2    = np.arccos(cos_tilt2)
        phi2      = np.random.uniform(0, 2*np.pi)

        S1x = chi1 * np.sin(theta1) * np.cos(phi1)
        S1y = chi1 * np.sin(theta1) * np.sin(phi1)
        S1z = chi1 * np.cos(theta1)

        S2x = chi2 * np.sin(theta2) * np.cos(phi2)
        S2y = chi2 * np.sin(theta2) * np.sin(phi2)
        S2z = chi2 * np.cos(theta2)

        incl = np.random.uniform(INCL_MIN, INCL_MAX)
        ecc  = np.random.uniform(ECC_MIN, ECC_MAX)

        # Extrinsic parameters
        ra  = np.random.uniform(RA_MIN, RA_MAX)
        dec = np.random.uniform(DEC_MIN, DEC_MAX)
        psi = PSI_FIXED

        # Distance (log‐uniform)
        log_d_min = np.log10(DIST_MIN)
        log_d_max = np.log10(DIST_MAX)
        d = 10 ** np.random.uniform(log_d_min, log_d_max)

        # Coalescence time and phase
        t0   = np.random.uniform(COAL_MIN, COAL_MAX)
        phi0 = np.random.uniform(0.0, 2*np.pi)

        thetas[i, :] = np.array([
            m1, m2,
            S1x, S1y, S1z,
            S2x, S2y, S2z,
            incl, ecc,
            ra, dec,
            d, t0, phi0
        ], dtype=np.float32)

        param_list.append((
            m1, m2,
            S1x, S1y, S1z,
            S2x, S2y, S2z,
            incl, ecc,
            ra, dec,
            d, t0, phi0
        ))

    return param_list, thetas

def sample_parameters_non_spinning(num_samples):
    """
    Sample `num_samples` random (m1, m2, spins, extrinsic) parameter sets.
    Returns:
      param_list: list of length num_samples, each a 15‐tuple
                  (m1, m2, S1x, S1y, S1z, S2x, S2y, S2z, incl, ecc, ra, dec, dist, t0, phi0)
      thetas:     np.ndarray of shape (num_samples, 15) with raw floats
    """
    param_list = []
    thetas = np.zeros((num_samples, 15), dtype=np.float32)

    for i in range(num_samples):
        # Intrinsic parameters
        m1 = np.random.uniform(MASS_MIN, MASS_MAX)
        m2 = np.random.uniform(MASS_MIN, MASS_MAX)

        # 0 Spin vectors and eccentricity
        S1x = 0
        S1y = 0
        S1z = 0

        S2x = 0
        S2y = 0
        S2z = 0

        incl = np.random.uniform(INCL_MIN, INCL_MAX)
        #ecc  = np.random.uniform(ECC_MIN, ECC_MAX)
        ecc = 0

        # Extrinsic parameters
        ra  = np.random.uniform(RA_MIN, RA_MAX)
        dec = np.random.uniform(DEC_MIN, DEC_MAX)
        psi = PSI_FIXED

        # Distance (log‐uniform)
        log_d_min = np.log10(DIST_MIN)
        log_d_max = np.log10(DIST_MAX)
        d = 10 ** np.random.uniform(log_d_min, log_d_max)

        # Coalescence time and phase
        t0   = np.random.uniform(COAL_MIN, COAL_MAX)
        phi0 = np.random.uniform(0.0, 2*np.pi)

        thetas[i, :] = np.array([
            m1, m2,
            S1x, S1y, S1z,
            S2x, S2y, S2z,
            incl, ecc,
            ra, dec,
            d, t0, phi0
        ], dtype=np.float32)

        param_list.append((
            m1, m2,
            S1x, S1y, S1z,
            S2x, S2y, S2z,
            incl, ecc,
            ra, dec,
            d, t0, phi0
        ))

    return param_list, thetas



def build_common_times(delta_t=DELTA_T, t_before=T_BEFORE, t_after=T_AFTER):
    """
    Build a fixed “common_times” grid running from –t_before to +t_after in steps of delta_t.
    Returns:
      common_times: np.ndarray of shape (N_common,)
      N_common:     integer number of samples along the grid
    """
    common_times = np.arange(-t_before, t_after + delta_t/2, delta_t, dtype=np.float32)
    N_common = len(common_times)
    return common_times, N_common


def build_waveform_chunks(param_list, common_times, n_common,
                          delta_t=DELTA_T, f_lower=F_LOWER,
                          waveform_name=WAVEFORM_NAME, detector_name=DETECTOR_NAME,
                          psi_fixed=PSI_FIXED):
    """
    For each parameter tuple in `param_list`, generate a PyCBC time‐domain waveform,
    project onto the detector, truncate to [–T_BEFORE, +T_AFTER], compute instantaneous
    amplitude & phase, and store contiguous “chunks” of (start_idx, amp_chunk, dphi_chunk).

    Returns:
      waveform_chunks: list of length len(param_list), where each element is a dict:
        {
          "start_idx": int,
          "amp_chunk":  np.ndarray of shape (chunk_len,),
          "dphi_chunk": np.ndarray of shape (chunk_len,)
        }
    """
    waveform_chunks = []

    for params in param_list:
        (
            m1, m2,
            S1x, S1y, S1z,
            S2x, S2y, S2z,
            incl, ecc,
            ra, dec,
            d, t0, phi0
        ) = params

        # 1) Generate plus/cross polarizations
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
            f_lower           = f_lower,
            approximant       = waveform_name
        )
        h_plus = hp.numpy().astype(np.float32)
        h_cross = hc.numpy().astype(np.float32)
        t_plus = hp.sample_times.numpy().astype(np.float32)

        # 2) Detector projection
        det = Detector(detector_name)
        Fp, Fx = det.antenna_pattern(ra, dec, psi_fixed, 0.0)
        h_det = (Fp * h_plus + Fx * h_cross).astype(np.float32)

        # 3) Convert to time relative to merger, then index on common_times
        t_rel = t_plus - t0
        idxs = np.round((t_rel + T_BEFORE) / delta_t).astype(int)
        valid = (idxs >= 0) & (idxs < n_common)

        if not np.any(valid):
            # If no overlap, store an empty chunk
            waveform_chunks.append({
                "start_idx": 0,
                "amp_chunk": np.zeros(0, dtype=np.float32),
                "dphi_chunk": np.zeros(0, dtype=np.float32)
            })
            continue

        kept_idxs = idxs[valid]
        h_chunk_raw = h_det[valid]

        # 4) Instantaneous amplitude & phase on the chunk
        analytic = hilbert(h_chunk_raw)
        inst_amp = np.abs(analytic).astype(np.float32)
        inst_phi = np.unwrap(np.angle(analytic)).astype(np.float32)

        # 5) Normalize amplitude by its peak
        A_peak = inst_amp.max() + 1e-30
        amp_norm_chunk = (inst_amp / A_peak).astype(np.float32)

        # 6) Compute dphi (first element is phi[0], then differences)
        dphi_chunk = np.empty_like(inst_phi, dtype=np.float32)
        dphi_chunk[0] = inst_phi[0]
        dphi_chunk[1:] = inst_phi[1:] - inst_phi[:-1]

        # 7) Determine contiguous chunk indices
        start_idx = int(kept_idxs.min())
        end_idx = int(kept_idxs.max())
        chunk_len = end_idx - start_idx + 1

        # 8) Build full-length arrays and insert normalized values
        full_amp = np.zeros(chunk_len, dtype=np.float32)
        full_dphi = np.zeros(chunk_len, dtype=np.float32)
        offsets = kept_idxs - start_idx
        full_amp[offsets] = amp_norm_chunk
        full_dphi[offsets] = dphi_chunk

        waveform_chunks.append({
            "start_idx": start_idx,
            "amp_chunk": full_amp,
            "dphi_chunk": full_dphi
        })

    return waveform_chunks

def compute_engineered_features(thetas_raw: np.ndarray) -> np.ndarray:
    """
    Input: thetas_raw of shape (N,15) with columns
      [m1, m2, S1x,S1y,S1z, S2x,S2y,S2z, incl, ecc, ra, dec, dist, t0, phi0]
    Output: thetas_feat of shape (N, F) with columns
      [M, eta, chi_eff, chi_p,  incl, ecc, ra, dec, dist, t0, phi0]
    """
    N = thetas_raw.shape[0]
    feats = np.zeros((N, 11), dtype=np.float32)
    m1 = thetas_raw[:,0]; m2 = thetas_raw[:,1]
    # 1) chirp mass
    M  = (m1*m2)**(3/5)/(m1+m2)**(1/5)
    # 2) symmetric mass ratio
    eta = (m1*m2)/(m1+m2)**2
    # 3) effective aligned spin
    S1z = thetas_raw[:,4];  S2z = thetas_raw[:,7]
    chi_eff = (m1*S1z + m2*S2z)/(m1+m2)
    # 4) effective precession spin
    S1perp = np.hypot(thetas_raw[:,2], thetas_raw[:,3])
    S2perp = np.hypot(thetas_raw[:,5], thetas_raw[:,6])
    q = np.minimum(m1,m2)/np.maximum(m1,m2)
    chi_p = np.maximum( S1perp,
                       ((3+4*q)/(4+3*q))*S2perp )
    # 5) copy the remaining extrinsic parameters  incl,ecc,ra,dec,dist,t0,phi0
    extras = thetas_raw[:, 8:15]  # incl,ecc,ra,dec,dist,t0,phi0
    feats[:, :4]  = np.vstack([M,eta,chi_eff,chi_p]).T
    feats[:, 4:]  = extras
    return feats
