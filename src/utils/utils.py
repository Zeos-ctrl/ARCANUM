import os
import json
import torch
import logging
import requests
import numpy as np
import torch.nn as nn
from dataclasses import dataclass

from src.models.model_factory import make_amp_model, make_phase_model
from src.data.config import *

logger = logging.getLogger(__name__)

def save_checkpoint(checkpoint_dir, amp_model, phase_model, data,
                    amp_weight_var, amp_bias_var,
                    phase_weight_var, phase_bias_var,
                    noise_variance):
    """
    Save important model information
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Model weights
    torch.save(amp_model.state_dict(),
               os.path.join(checkpoint_dir, "amp_model.pt"))
    torch.save(phase_model.state_dict(),
               os.path.join(checkpoint_dir, "phase_model.pt"))

    # Normalization stats & constants
    np.save(os.path.join(checkpoint_dir, "param_means.npy"), data.param_means)
    np.save(os.path.join(checkpoint_dir, "param_stds.npy"),  data.param_stds)
    np.save(os.path.join(checkpoint_dir, "t_norm_array.npy"), data.t_norm_array)

    # Hessian‐diag variances
    np.save(
        os.path.join(checkpoint_dir, "amp_last_weight_variances.npy"),
        amp_weight_var.cpu().numpy()
    )
    np.save(
        os.path.join(checkpoint_dir, "amp_last_bias_variance.npy"),
        amp_bias_var.cpu().numpy()
    )
    np.save(
        os.path.join(checkpoint_dir, "phase_last_weight_variances.npy"),
        phase_weight_var.cpu().numpy()
    )
    np.save(
        os.path.join(checkpoint_dir, "phase_last_bias_variance.npy"),
        phase_bias_var.cpu().numpy()
    )

    # Metadata JSON
    meta = {
        "waveform_length":   WAVEFORM_LENGTH,
        "delta_t":           DELTA_T,
        "log_amp_min":       float(data.log_amp_min),
        "log_amp_max":       float(data.log_amp_max),
        "train_samples":     NUM_SAMPLES,
        "hessian_noise_var": noise_variance
    }
    with open(os.path.join(checkpoint_dir, "meta.json"), "w") as f:
        json.dump(meta, f)

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
    data: np.ndarray        # Data array of the waveform
    uncertainty: np.ndarray # Data array of uncertainty for every datapoint
    epoch: float            # Start time for the waveform
    sample_rate: float      # delta_t
    time: np.ndarray        # Normalized time array
    approximant: str        # approximant used in training

class WaveformPredictor:
    def __init__(self, checkpoint_dir: str, device: str = DEVICE):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Initializing WaveformPredictor from '{checkpoint_dir}'")

        self.device = torch.device(device)

        # Load normalization stats
        self.param_means     = np.load(os.path.join(checkpoint_dir, "param_means.npy"))
        self.param_stds      = np.load(os.path.join(checkpoint_dir, "param_stds.npy"))
        self.time_norm_array = np.load(os.path.join(checkpoint_dir, "t_norm_array.npy"))

        # Load metadata
        meta_path = os.path.join(checkpoint_dir, "meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"No meta.json in {checkpoint_dir}")
        with open(meta_path) as fp:
            meta = json.load(fp)
        self.log_amp_min     = meta["log_amp_min"]
        self.log_amp_max     = meta["log_amp_max"]
        self.waveform_length = meta["waveform_length"]
        self.delta_t         = meta["delta_t"]
        self.train_samples   = int(meta.get("train_samples", 0))

        # Build and load models
        features = len(TRAIN_FEATURES)

        self.amp_model = make_amp_model(
            in_param_dim=features,
        ).to(DEVICE)

        self.phase_model = make_phase_model(
            param_dim=features,
        ).to(DEVICE)

        self.amp_model.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, "amp_model.pt"), map_location=self.device)
        )
        self.phase_model.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, "phase_model.pt"), map_location=self.device)
        )

        # Load Laplace variances
        self.amp_last_weight_variances  = torch.from_numpy(
            np.load(os.path.join(checkpoint_dir, "amp_last_weight_variances.npy"))
        ).to(self.device)
        self.amp_last_bias_variance     = torch.from_numpy(
            np.load(os.path.join(checkpoint_dir, "amp_last_bias_variance.npy"))
        ).to(self.device)
        self.phase_last_weight_variances = torch.from_numpy(
            np.load(os.path.join(checkpoint_dir, "phase_last_weight_variances.npy"))
        ).to(self.device)
        self.phase_last_bias_variance    = torch.from_numpy(
            np.load(os.path.join(checkpoint_dir, "phase_last_bias_variance.npy"))
        ).to(self.device)

        self.amp_model.eval()
        self.phase_model.eval()
        self.logger.info("Models and variances loaded; in eval mode.")

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
        m1: float,
        m2: float,
        spin1_z: float,
        spin2_z: float,
        inclination: float,
        eccentricity: float,
        waveform_length=None,
        sampling_dt=None
    ):
        """
        Returns the predicted waveforms based off specific input parameters
        """
        # real & normalized time grids
        length  = waveform_length or self.waveform_length
        delta_t = sampling_dt    or self.delta_t
        real_time = np.linspace(-length*delta_t, 0.0, length)
        time_norm = 2*(real_time + length*delta_t)/(length*delta_t) - 1

        # derived features -> normalized
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

        # build model input
        param_grid = np.tile(theta_n, (length,1)) # (L,D)
        model_input = np.concatenate([time_norm[:,None], param_grid], axis=1) # (L,1+D)
        inp_t = torch.from_numpy(model_input.astype(np.float32)).to(self.device)

        # forward
        with torch.no_grad():
            A_norm = self.amp_model(inp_t[:,:1], inp_t[:,1:]).cpu().numpy().ravel()
            phi    = self.phase_model(inp_t[:,:1], inp_t[:,1:]).cpu().numpy().ravel()

        # Invert amp‐norm and build polarizations
        amp    = self.inverse_log_norm(A_norm)
        cosi   = np.cos(inclination)
        h_plus  = amp * ((1 + cosi**2)/2) * np.cos(phi)
        h_cross = amp * ( cosi ) * np.sin(phi)

        # wrap into TimeSeriesStrainData, as we dont calculate uncertainty just
        # duplicate strain, in future iterations get rid of it
        plus = TimeSeriesStrainData(
            data        = h_plus,
            uncertainty = None,
            epoch       = real_time[0],
            sample_rate = delta_t,
            time        = time_norm,
            approximant = WAVEFORM
        )
        cross = TimeSeriesStrainData(
            data        = h_cross,
            uncertainty = None,
            epoch       = real_time[0],
            sample_rate = delta_t,
            time        = time_norm,
            approximant = WAVEFORM
        )
        return plus, cross


    def batch_predict(self, thetas_raw, batch_size=None):
        """
        Using GPU batching take an array of parameters and return waveforms
        """
        # get sizes
        N = thetas_raw.shape[0]
        length = self.waveform_length
        delta_t = self.delta_t
        if batch_size is None:
            batch_size = self.train_samples

        # precompute time grids
        real_time = np.linspace(-length*delta_t, 0.0, length)
        time_norm = 2*(real_time + length*delta_t)/(length*delta_t) - 1

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
            theta_norm = (derived - self.param_means) / self.param_stds

            # build and flatten model input (B*L,1+D)
            t_blk = np.broadcast_to(time_norm, (B, length))
            p_blk = np.broadcast_to(theta_norm[:,None,:], (B, length, D))
            flat  = np.concatenate([t_blk[...,None], p_blk], axis=-1).reshape(-1,1+D).astype(np.float32)

            inp_t = torch.from_numpy(flat).to(self.device)
            with torch.no_grad():
                Amp_norm = self.amp_model(inp_t[:,:1], inp_t[:,1:]).cpu().numpy().reshape(B, length)
                phase  = self.phase_model(inp_t[:,:1], inp_t[:,1:]).cpu().numpy().reshape(B, length)

            all_amp.append(Amp_norm)
            all_phi.append(phase)

        Amp_mat   = np.vstack(all_amp)   # (N,L)
        phi_mat = np.vstack(all_phi)   # (N,L)

        # reconstruct plus/cross
        incls   = thetas_raw[:, 3]
        cosi    = np.cos(incls)[:,None]

        amp_mat = self.inverse_log_norm(Amp_mat)
        h_plus  = amp_mat * ((1+cosi**2)/2) * np.cos(phi_mat)
        h_cross = amp_mat * ( cosi ) * np.sin(phi_mat)

        # wrap per sample
        h_plus_list, h_cross_list = [], []
        for i in range(N):
            h_plus_list.append(TimeSeriesStrainData(
                data        = h_plus[i],
                uncertainty = h_plus[i],
                epoch       = real_time[0],
                sample_rate = delta_t,
                time        = time_norm,
                approximant = WAVEFORM
            ))
            h_cross_list.append(TimeSeriesStrainData(
                data        = h_cross[i],
                uncertainty = h_cross[i],
                epoch       = real_time[0],
                sample_rate = delta_t,
                time        = time_norm,
                approximant = WAVEFORM
            ))

        return h_plus_list, h_cross_list

    def predict_with_uncertainty(
        self,
        m1: float,
        m2: float,
        spin1_z: float,
        spin2_z: float,
        inclination: float,
        eccentricity: float,
        waveform_length: int = None,
        sampling_delta_t: float = None,
        sigma_level: int = 1
    ) -> tuple[TimeSeriesStrainData, TimeSeriesStrainData]:
        """
        Single‑sample prediction with uncertainty.
        Returns (h_plus_ts, h_cross_ts).
        """
        # Time arrays
        length   = waveform_length or self.waveform_length
        delta_t  = sampling_delta_t or self.delta_t
        time_array      = np.linspace(-length*delta_t, 0.0, length)
        normalized_time = 2*(time_array + length*delta_t)/(length*delta_t) - 1

        # derived features -> normalized
        derived_map = {
            "chirp_mass":           (m1*m2)**(3/5)/(m1+m2)**(1/5),
            "symmetric_mass_ratio": (m1*m2)/(m1+m2)**2,
            "effective_spin":       (m1*spin1_z + m2*spin2_z)/(m1+m2),
            "inclination":          inclination,
            "eccentricity":         eccentricity
        }
        D = len(TRAIN_FEATURES)
        derived = np.array([derived_map[f] for f in TRAIN_FEATURES], dtype=np.float32)  # (D,)
        normalized_derived = (derived - self.param_means) / self.param_stds                        # (D,)

        # Build input tensor
        feature_grid = np.tile(normalized_derived, (length, 1))  # (L,5)
        model_input  = np.hstack([
            normalized_time.reshape(-1,1),    # (L,1)
            feature_grid                      # (L,5)
        ]).astype(np.float32)
        inp = torch.from_numpy(model_input).to(self.device)

        # Hook final linear layers
        def find_last_linear(mod):
            return [m for m in mod.modules() if isinstance(m, nn.Linear) and m.out_features==1][-1]

        amp_lin = find_last_linear(self.amp_model)
        phs_lin = find_last_linear(self.phase_model)
        captured = {}

        amp_handle = amp_lin.register_forward_hook(lambda m,i,o: captured.update(amp_phi=i[0].detach()))
        phs_handle = phs_lin.register_forward_hook(lambda m,i,o: captured.update(phs_phi=i[0].detach()))

        # Forward to get means
        with torch.no_grad():
            mu_log_amp = self.amp_model(inp[:,:1], inp[:,1:])
            mu_phase   = self.phase_model(inp[:,:1], inp[:,1:])

        amp_handle.remove()
        phs_handle.remove()

        # Compute variances
        phi_amp   = captured['amp_phi']   # (L, H_amp)
        phi_phase = captured['phs_phi']   # (L, H_phase)

        var_log_amp = (phi_amp**2 * self.amp_last_weight_variances).sum(1,True) + self.amp_last_bias_variance
        var_phase   = (phi_phase**2 * self.phase_last_weight_variances).sum(1,True) + self.phase_last_bias_variance

        # To NumPy
        mu_log_amp_np = mu_log_amp.cpu().numpy().ravel()
        sigma_log_amp = np.sqrt(var_log_amp.cpu().numpy().ravel()) * sigma_level
        mu_phase_np   = mu_phase.cpu().numpy().ravel()
        sigma_phase   = np.sqrt(var_phase.cpu().numpy().ravel())    * sigma_level

        # Invert log‑norm
        linear_amp = self.inverse_log_norm(mu_log_amp_np)

        # Analytic propagation
        cos_i = np.cos(inclination)
        plus_mean  = linear_amp * ((1+cos_i**2)/2) * np.cos(mu_phase_np)
        plus_std   = np.sqrt(
            (plus_mean*sigma_log_amp)**2 +
            (linear_amp*((1+cos_i**2)/2)*np.sin(mu_phase_np)*sigma_phase)**2
        )
        cross_mean = linear_amp * cos_i * np.sin(mu_phase_np)
        cross_std  = np.sqrt(
            (cross_mean*sigma_log_amp)**2 +
            (linear_amp*cos_i*np.cos(mu_phase_np)*sigma_phase)**2
        )

        # Wrap
        h_plus = TimeSeriesStrainData(
            data        = plus_mean,
            uncertainty = plus_std,
            epoch       = time_array[0],
            sample_rate = delta_t,
            time        = normalized_time,
            approximant = WAVEFORM
        )
        h_cross = TimeSeriesStrainData(
            data        = cross_mean,
            uncertainty = cross_std,
            epoch       = time_array[0],
            sample_rate = delta_t,
            time        = normalized_time,
            approximant = WAVEFORM
        )
        return h_plus, h_cross


    def batch_predict_with_uncertainty(
        self,
        thetas_raw: np.ndarray,
        batch_size: int = None,
        sigma_level: int = 1
    ) -> tuple[list[TimeSeriesStrainData], list[TimeSeriesStrainData]]:
        """
        Returns two lists:
          - list of h_plus TimeSeriesStrainData
          - list of h_cross TimeSeriesStrainData
        with uncertainties scaled by sigma_level.
        """
        num_samples = thetas_raw.shape[0]
        length      = self.waveform_length
        delta_t     = self.delta_t
        if batch_size is None:
            batch_size = self.train_samples

        # Precompute times
        time_array      = np.linspace(-length*delta_t, 0.0, length)
        normalized_time = 2*(time_array + length*delta_t)/(length*delta_t) - 1

        # Identify final linear layers once
        def find_last_linear(mod):
            return [m for m in mod.modules() if isinstance(m, nn.Linear) and m.out_features==1][-1]
        amp_lin = find_last_linear(self.amp_model)
        phs_lin = find_last_linear(self.phase_model)

        all_plus, all_cross = [], []
        feature_dim = len(TRAIN_FEATURES)

        for start in range(0, num_samples, batch_size):
            end   = min(start + batch_size, num_samples)
            block = thetas_raw[start:end]  # shape (B,6)
            B     = end - start

            # Derived & normalize
            derived = np.stack([
                (b[0]*b[1])**(3/5)/(b[0]+b[1])**(1/5),
                (b[0]*b[1])/(b[0]+b[1])**2,
                (b[0]*b[2] + b[1]*b[3])/(b[0]+b[1]),
                b[4], b[5]
            ] for b in block).astype(np.float32)  # (B,5)
            norm_derived = (derived - self.param_means)/self.param_stds  # (B,5)

            # Build input tensor (B*L, 1+5)
            t_blk = np.broadcast_to(normalized_time, (B, length))
            p_blk = np.broadcast_to(norm_derived[:,None,:], (B, length, feature_dim))
            flat_in = np.concatenate([t_blk[...,None], p_blk], axis=-1) \
                      .reshape(-1, 1+feature_dim).astype(np.float32)
            inp_tensor = torch.from_numpy(flat_in).to(self.device)

            # Hook features
            captured = {}
            hA = amp_lin.register_forward_hook(lambda m,i,o: captured.update(amp_phi=i[0].detach()))
            hP = phs_lin.register_forward_hook(lambda m,i,o: captured.update(phs_phi=i[0].detach()))

            with torch.no_grad():
                mu_log_amp = self.amp_model(inp_tensor[:,:1], inp_tensor[:,1:])
                mu_phase   = self.phase_model(inp_tensor[:,:1], inp_tensor[:,1:])

            hA.remove(); hP.remove()

            # Extract & compute variances
            phi_A = captured['amp_phi']       # (B*L, H_amp)
            phi_P = captured['phs_phi']       # (B*L, H_phase)

            var_logA = (phi_A**2 * self.amp_last_weight_variances).sum(1,True) \
                       + self.amp_last_bias_variance
            var_phi  = (phi_P**2 * self.phase_last_weight_variances).sum(1,True) \
                       + self.phase_last_bias_variance

            # Reshape (B, L)
            mu_logA_np   = mu_log_amp.cpu().numpy().reshape(B, length)
            sigma_logA   = np.sqrt(var_logA.cpu().numpy().reshape(B, length)) * sigma_level
            mu_phase_np  = mu_phase.cpu().numpy().reshape(B, length)
            sigma_phase  = np.sqrt(var_phi.cpu().numpy().reshape(B, length))    * sigma_level

            # Invert log and propagate
            linear_amp = self.inverse_log_norm(mu_logA_np)  # (B,L)
            cos_inc    = np.cos(block[:,4])[:,None]         # (B,1)

            plus_mean  = linear_amp * ((1+cos_inc**2)/2) * np.cos(mu_phase_np)
            plus_std   = np.sqrt(
                (plus_mean*sigma_logA)**2 +
                (linear_amp*((1+cos_inc**2)/2)*np.sin(mu_phase_np)*sigma_phase)**2
            )
            cross_mean = linear_amp * cos_inc * np.sin(mu_phase_np)
            cross_std  = np.sqrt(
                (cross_mean*sigma_logA)**2 +
                (linear_amp*cos_inc*np.cos(mu_phase_np)*sigma_phase)**2
            )

            for i in range(B):
                all_plus.append(TimeSeriesStrainData(
                    data        = plus_mean[i],
                    uncertainty = plus_std[i],
                    epoch       = time_array[0],
                    sample_rate = delta_t,
                    time        = normalized_time,
                    approximant = WAVEFORM
                ))
                all_cross.append(TimeSeriesStrainData(
                    data        = cross_mean[i],
                    uncertainty = cross_std[i],
                    epoch       = time_array[0],
                    sample_rate = delta_t,
                    time        = normalized_time,
                    approximant = WAVEFORM
                ))

        return all_plus, all_cross

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
        # time grid
        L  = waveform_length or self.waveform_length
        dt = sampling_dt    or self.delta_t
        t_real = np.linspace(-L*dt, 0.0, L)
        t_norm = 2*(t_real + L*dt)/(L*dt) - 1

        # prepare derived_map
        derived_map = {
          "chirp_mass":           (m1*m2)**(3/5)/(m1+m2)**(1/5),
          "symmetric_mass_ratio": (m1*m2)/(m1+m2)**2,
          "effective_spin":       (m1*spin1_z + m2*spin2_z)/(m1+m2),
          "inclination":          inclination,
          "eccentricity":         eccentricity
        }

        # stack only TRAIN_FEATURES
        D = len(TRAIN_FEATURES)
        derived = np.array([derived_map[f] for f in TRAIN_FEATURES],
                           dtype=np.float32)             # (D,)

        # normalize
        theta_n = (derived - self.param_means) / self.param_stds  # (D,)

        # build model input (L,1+D)
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
