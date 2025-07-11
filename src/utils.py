import os
import json
import torch
import logging
import requests
import numpy as np
from dataclasses import dataclass

from src.model import *
from src.config import *

logger = logging.getLogger(__name__)

def save_checkpoint(checkpoint_dir, amp_model, phase_model, data):
    logger.info(f"Saving checkpoint to '{checkpoint_dir}'")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Model weights
    torch.save(amp_model.state_dict(),
               os.path.join(checkpoint_dir, "amp_model.pt"))
    torch.save(phase_model.state_dict(),
               os.path.join(checkpoint_dir, "phase_model.pt"))
    logger.debug("Saved model weights.")

    # Normalization stats & constants
    np.save(os.path.join(checkpoint_dir, "param_means.npy"), data.param_means)
    np.save(os.path.join(checkpoint_dir, "param_stds.npy"),  data.param_stds)
    np.save(os.path.join(checkpoint_dir, "t_norm_array.npy"), data.t_norm_array)
    logger.debug("Saved normalization stats and constants.")

    # Save metadata JSON
    meta = {
        "waveform_length": WAVEFORM_LENGTH,
        "delta_t": DELTA_T,
        "log_amp_min": float(data.log_amp_min),
        "log_amp_max": float(data.log_amp_max),
        "train_samples": NUM_SAMPLES
    }
    with open(os.path.join(checkpoint_dir, "meta.json"), "w") as f:
        json.dump(meta, f)
    logger.info("Saved metadata JSON.")

def notify_discord(message: str, url: str = None):
    """
    Send a notification message to Discord via webhook.
    """
    hook = url or DISCORD_WEBHOOK_URL
    if not hook:
        logger.warning("No DISCORD_WEBHOOK_URL configured—skipping Discord notification.")
        return

    payload = {"content": message}
    headers = {"Content-Type": "application/json"}
    try:
        resp = requests.post(hook, json=payload, headers=headers, timeout=5)
        resp.raise_for_status()
        logger.info("Sent Discord notification.")
    except Exception as e:
        logger.error("Failed to send Discord notification: %s", e)

def compute_match(h_true, h_pred):
    """
    Compute the normalized cross‑correlation match between two 1D arrays.
    Returns a scalar in [0,1].
    """
    logger.debug("Computing waveform match...")
    # zero‑mean
    ht = h_true - np.mean(h_true)
    hp = h_pred - np.mean(h_pred)
    # full cross‑correlation
    corr    = np.correlate(ht, hp, mode="full")
    max_corr= np.max(corr)
    norm    = np.sqrt(np.dot(ht, ht) * np.dot(hp, hp))
    logger.debug(f"Spoiler alert match is: {max_corr / norm}")
    return max_corr / norm

@dataclass
class TimeSeriesStrainData:
    data: np.ndarray # Data array of the waveform
    epoch: float # Start time for the waveform
    sample_rate: float # delta_t
    time: np.ndarray # Normalized time array
    approximant: str # approximant used in training

class WaveformPredictor:
    def __init__(self, checkpoint_dir, device="cpu"):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Initializing WaveformPredictor from checkpoint '{checkpoint_dir}'")

        self.device = torch.device(device)

        # Load normalization for derived params
        self.param_means  = np.load(os.path.join(checkpoint_dir, "param_means.npy"))
        self.param_stds   = np.load(os.path.join(checkpoint_dir, "param_stds.npy"))
        self.t_norm_array = np.load(os.path.join(checkpoint_dir, "t_norm_array.npy"))

        # Load meta
        meta_path = os.path.join(checkpoint_dir, "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            self.log_amp_min   = meta["log_amp_min"]
            self.log_amp_max   = meta["log_amp_max"]
            self.waveform_length = meta["waveform_length"]
            self.delta_t         = meta["delta_t"]
            self.train_samples   = int(meta.get("train_samples", 0))
        else:
            raise FileNotFoundError(f"Meta file not found at {meta_path}")

        self.features = len(TRAIN_FEATURES)

        # Build models
        self.amp_model = AmplitudeDNN_Full(
            in_param_dim=self.features,
            time_dim=1,
            emb_hidden=AMP_EMB_HIDDEN,
            amp_hidden=AMP_HIDDEN,
            N_banks=AMP_BANKS,
            dropout=0.2
        ).to(self.device)

        self.phase_model = PhaseDNN_Full(
            param_dim=self.features,
            time_dim=1,
            emb_hidden=PHASE_EMB_HIDDEN,
            phase_hidden=PHASE_HIDDEN,
            N_banks=PHASE_BANKS,
            dropout=0.1
        ).to(self.device)

        # Load weights
        self.amp_model.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, "amp_model.pt"), map_location=self.device)
        )
        self.phase_model.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, "phase_model.pt"), map_location=self.device)
        )
        self.amp_model.eval()
        self.phase_model.eval()
        self.logger.info("Models loaded and set to eval mode.")

    def _compute_derived(self, m1, m2, chi1z, chi2z, incl, ecc):
        """Return [Mc, η, χ_eff, incl, ecc]."""
        Mc   = (m1*m2)**(3/5) / (m1+m2)**(1/5)
        eta  = (m1*m2) / (m1+m2)**2
        chi_eff = (m1*chi1z + m2*chi2z) / (m1+m2)
        return np.array([Mc, eta, chi_eff, incl, ecc], dtype=np.float32)

    def _normalize_derived(self, derived):
        return (derived - self.param_means) / self.param_stds

    def inverse_log_norm(self, y_norm: np.ndarray) -> np.ndarray:
        """Denormalize log10 amplitude and return linear amplitude."""
        y = y_norm * (self.log_amp_max - self.log_amp_min) + self.log_amp_min
        return 10 ** y

    def predict(
        self,
        m1, m2,
        spin1_z, spin2_z,
        inclination, eccentricity,
        waveform_length=None,
        sampling_dt=None
    ):
        # real & normalized time grids
        L  = waveform_length or self.default_length
        dt = sampling_dt    or self.default_dt
        t_real = np.linspace(-L*dt, 0.0, L)                                           # (L,)
        t_norm = 2*(t_real + L*dt)/(L*dt) - 1                                        # (L,)

        # derived features → normalized
        derived_map = {
            "chirp_mass":           (m1*m2)**(3/5)/(m1+m2)**(1/5),
            "symmetric_mass_ratio": (m1*m2)/(m1+m2)**2,
            "effective_spin":       (m1*spin1_z + m2*spin2_z)/(m1+m2),
            "inclination":          inclination,
            "eccentricity":         eccentricity
        }
        D = len(TRAIN_FEATURES)
        derived = np.array([derived_map[f] for f in TRAIN_FEATURES], dtype=np.float32)  # (D,)
        theta_n = (derived - self.param_means) / self.param_stds                        # (D,)

        # build model input (L, 1+D)
        param_grid = np.tile(theta_n, (L,1))                                           # (L,D)
        model_input = np.concatenate([t_norm[:,None], param_grid], axis=1)             # (L,1+D)
        inp_t = torch.from_numpy(model_input.astype(np.float32)).to(self.device)

        # forward
        with torch.no_grad():
            A_norm = self.amp_model(inp_t[:,:1], inp_t[:,1:]).cpu().numpy().ravel()     # (L,)
            phi    = self.phase_model(inp_t[:,:1], inp_t[:,1:]).cpu().numpy().ravel()  # (L,)

        # Invert amp‐norm and build polarizations
        amp    = self.inverse_log_norm(A_norm)                                         # (L,)
        cosi   = np.cos(inclination)
        h_plus  = amp * ((1 + cosi**2)/2) * np.cos(phi)
        h_cross = amp * ( cosi       ) * np.sin(phi)

        # wrap into TimeSeriesStrainData
        plus = TimeSeriesStrainData(
            data        = h_plus,
            epoch       = t_real[0],
            sample_rate = dt,
            time  = t_norm,
            approximant = WAVEFORM
        )
        cross = TimeSeriesStrainData(
            data        = h_cross,
            epoch       = t_real[0],
            sample_rate = dt,
            time  = t_norm,
            approximant = WAVEFORM
        )
        return plus, cross


    def batch_predict(self, thetas_raw, batch_size=None):
        N = thetas_raw.shape[0]
        L = self.default_length
        dt = self.default_dt
        if batch_size is None:
            batch_size = self.train_samples

        # precompute time grids
        t_real = np.linspace(-L*dt, 0.0, L)
        t_norm = 2*(t_real + L*dt)/(L*dt) - 1

        D = len(TRAIN_FEATURES)
        all_amp, all_phi = [], []

        for start in range(0, N, batch_size):
            end   = min(start+batch_size, N)
            block = thetas_raw[start:end]
            B     = end - start

            # derived & normalize (B,D)
            derived = []
            for (m1,m2,sp1,sp2,inc,ecc) in block:
                dm = {
                    "chirp_mass":           (m1*m2)**(3/5)/(m1+m2)**(1/5),
                    "symmetric_mass_ratio": (m1*m2)/(m1+m2)**2,
                    "effective_spin":       (m1*sp1 + m2*sp2)/(m1+m2),
                    "inclination":          inc,
                    "eccentricity":         ecc
                }
                derived.append([dm[f] for f in TRAIN_FEATURES])
            derived = np.array(derived, dtype=np.float32)
            theta_n = (derived - self.param_means) / self.param_stds

            # build and flatten model input (B*L,1+D)
            t_blk = np.broadcast_to(t_norm, (B, L))
            p_blk = np.broadcast_to(theta_n[:,None,:], (B, L, D))
            flat  = np.concatenate([t_blk[...,None], p_blk], axis=-1).reshape(-1,1+D).astype(np.float32)

            inp_t = torch.from_numpy(flat).to(self.device)
            with torch.no_grad():
                A_n = self.amp_model(inp_t[:,:1], inp_t[:,1:])\
                             .cpu().numpy().reshape(B, L)
                ph  = self.phase_model(inp_t[:,:1], inp_t[:,1:])\
                             .cpu().numpy().reshape(B, L)

            all_amp.append(A_n)
            all_phi.append(ph)

        A_mat   = np.vstack(all_amp)   # (N,L)
        phi_mat = np.vstack(all_phi)   # (N,L)

        # reconstruct plus/cross
        inc_idx = TRAIN_FEATURES.index("inclination")
        incls   = derived[:, inc_idx]
        cosi    = np.cos(incls)[:,None]

        amp_mat = self.inverse_log_norm(A_mat)
        h_plus  = amp_mat * ((1+cosi**2)/2) * np.cos(phi_mat)
        h_cross = amp_mat * ( cosi       ) * np.sin(phi_mat)

        # wrap per sample
        h_plus_list, h_cross_list = [], []
        for i in range(N):
            h_plus_list.append(TimeSeriesStrainData(
                data        = h_plus[i],
                epoch       = t_real[0],
                sample_rate = dt,
                time  = t_norm,
                approximant = WAVEFORM
            ))
            h_cross_list.append(TimeSeriesStrainData(
                data        = h_cross[i],
                epoch       = t_real[0],
                sample_rate = dt,
                time  = t_norm,
                approximant = WAVEFORM
            ))

        return h_plus_list, h_cross_list

    def predict_debug(
        self,
        m1, m2,
        spin1_z, spin2_z,
        inclination, eccentricity,
        waveform_length=None,
        sampling_dt=None
    ):
        """
        Like predict(), but returns the normalized log-amplitude and phase.
        Returns:
          time_seconds (L,),
          amp_pred     (L,),  # in linear units after inverse_log_norm
          phi_pred     (L,)   # raw unwrapped radians
        """
        # 1) time grid
        L  = waveform_length or self.waveform_length
        dt = sampling_dt    or self.delta_t
        t_real = np.linspace(-L*dt, 0.0, L)
        t_norm = 2*(t_real + L*dt)/(L*dt) - 1

        # 2) prepare derived_map
        derived_map = {
          "chirp_mass":           (m1*m2)**(3/5)/(m1+m2)**(1/5),
          "symmetric_mass_ratio": (m1*m2)/(m1+m2)**2,
          "effective_spin":       (m1*spin1_z + m2*spin2_z)/(m1+m2),
          "inclination":          inclination,
          "eccentricity":         eccentricity
        }

        # 3) stack only TRAIN_FEATURES
        D = len(TRAIN_FEATURES)
        derived = np.array([derived_map[f] for f in TRAIN_FEATURES],
                           dtype=np.float32)             # (D,)

        # 4) normalize
        theta_n = (derived - self.param_means) / self.param_stds  # (D,)

        # 5) build model input (L,1+D)
        param_grid = np.tile(theta_n, (L,1))
        model_input = np.concatenate([t_norm[:,None], param_grid], axis=1).astype(np.float32)

        inp_t = torch.from_numpy(model_input).to(self.device)
        with torch.no_grad():
            log_amp_norm = self.amp_model(inp_t[:,:1], inp_t[:,1:])\
                                 .cpu().numpy().ravel()
            phi_pred     = self.phase_model(inp_t[:,:1], inp_t[:,1:])\
                                 .cpu().numpy().ravel()

        # inverse‐log‐norm to get linear amplitude
        amp_pred = self.inverse_log_norm(log_amp_norm)

        return t_real, amp_pred, phi_pred


    def batch_predict_debug(self, thetas_raw, batch_size=None):
        """
        Returns:
          time_seconds (L,),
          amp_matrix   (N,L),  # linear amplitude
          phi_matrix   (N,L)   # raw unwrapped phase
        """
        N = thetas_raw.shape[0]
        L = self.waveform_length
        dt = self.delta_t
        if batch_size is None:
            batch_size = self.train_samples

        # prepare time grid once
        t_real = np.linspace(-L*dt, 0.0, L)
        t_norm = 2*(t_real + L*dt)/(L*dt) - 1

        D = len(TRAIN_FEATURES)
        all_amp, all_phi = [], []

        for start in range(0, N, batch_size):
            end   = min(start + batch_size, N)
            block = thetas_raw[start:end]  # (B,6)
            B     = end - start

            # compute derived & select features
            derived = []
            for m1,m2,sp1,sp2,inc,ecc in block:
                dm = {
                  "chirp_mass":           (m1*m2)**(3/5)/(m1+m2)**(1/5),
                  "symmetric_mass_ratio": (m1*m2)/(m1+m2)**2,
                  "effective_spin":       (m1*sp1 + m2*sp2)/(m1+m2),
                  "inclination":          inc,
                  "eccentricity":         ecc
                }
                derived.append([ dm[f] for f in TRAIN_FEATURES ])
            derived = np.array(derived, dtype=np.float32)  # (B,D)

            # normalize
            theta_n = (derived - self.param_means) / self.param_stds  # (B,D)

            # build input (B,L,1+D) -> flatten to (B*L,1+D)
            t_blk  = np.broadcast_to(t_norm, (B, L))                # (B,L)
            p_blk  = np.broadcast_to(theta_n[:,None,:], (B, L, D)) # (B,L,D)
            inp    = np.concatenate([t_blk[...,None], p_blk], axis=-1)
            flat   = inp.reshape(-1, 1+D).astype(np.float32)

            inp_t = torch.from_numpy(flat).to(self.device)
            with torch.no_grad():
                A_n = self.amp_model(inp_t[:,:1], inp_t[:,1:])\
                             .cpu().numpy().reshape(B, L)
                ph  = self.phase_model(inp_t[:,:1], inp_t[:,1:])\
                             .cpu().numpy().reshape(B, L)

            all_amp.append(self.inverse_log_norm(A_n))
            all_phi.append(ph)

        amp_matrix = np.vstack(all_amp)  # (N,L)
        phi_matrix = np.vstack(all_phi)  # (N,L)

        return t_real, amp_matrix, phi_matrix
