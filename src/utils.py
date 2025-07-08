import os
import json
import torch
import logging
import requests
import numpy as np

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

def notify_slack(message: str, url: str = None):
    """
    Send a notification message to Slack via Incoming Webhook.
    If url is None, falls back to the WEBHOOK_URL from config.
    """
    hook = url or WEBHOOK_URL
    if not hook:
        logger.warning("No WEBHOOK_URL configured—skipping Slack notification.")
        return

    payload = {"text": message}
    try:
        resp = requests.post(hook, json=payload, timeout=5)
        resp.raise_for_status()
        logger.info("Sent Slack notification.")
    except Exception as e:
        logger.error("Failed to send Slack notification: %s", e)

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

class WaveformPredictor:
    def __init__(self, checkpoint_dir, device="cpu"):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Initializing WaveformPredictor from checkpoint '{checkpoint_dir}'")

        self.device = torch.device(device)

        # Load normalization
        self.param_means    = np.load(f"{checkpoint_dir}/param_means.npy")
        self.param_stds     = np.load(f"{checkpoint_dir}/param_stds.npy")
        self.t_norm_array   = np.load(f"{checkpoint_dir}/t_norm_array.npy")
        meta_path = f"{checkpoint_dir}/meta.json"
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            self.log_amp_min = meta["log_amp_min"]
            self.log_amp_max = meta["log_amp_max"]
            self.waveform_length = meta["waveform_length"]
            self.delta_t         = meta["delta_t"]
            self.train_samples = int(meta.get("train_samples", NUM_SAMPLES))
            self.logger.debug(f"Loaded meta: waveform_length={self.waveform_length}, delta_t={self.delta_t}")
        else:
            self.waveform_length = YOUR_DEFAULT
            self.delta_t         = YOUR_DEFAULT
            self.logger.warning("Meta file missing, using default waveform_length and delta_t")

        # Build models (must match architecture used in training)
        self.amp_model = AmplitudeNet(in_dim=7, hidden_dims=[128,128,128])
        self.phase_model = PhaseDNN_Full(
            param_dim=6, time_dim=1,
            emb_hidden=[64,64],
            phase_hidden=[128,128,128,128],
            N_banks=1
        )

        # Load weights
        self.amp_model.load_state_dict(
            torch.load(f"{checkpoint_dir}/amp_model.pt", map_location=self.device)
        )
        self.phase_model.load_state_dict(
            torch.load(f"{checkpoint_dir}/phase_model.pt", map_location=self.device)
        )
        self.amp_model.to(self.device).eval()
        self.phase_model.to(self.device).eval()
        self.logger.info("Models loaded and set to eval mode.")

    def _normalize_params(self, raw_params):
        """raw_params: array-like of shape (6,)"""
        return (raw_params - self.param_means) / self.param_stds

    def inverse_log_norm(self, y_norm: np.ndarray) -> np.ndarray:
        # Denormalize log10(A):  y = y_norm*(U−L) + L
        y = y_norm * (self.log_amp_max - self.log_amp_min) + self.log_amp_min
        # Back to linear amplitude
        return 10 ** y

    def predict(self, m1, m2, chi1z, chi2z, incl, ecc):
        self.logger.debug(f"Predict called with params: m1={m1}, m2={m2}, chi1z={chi1z}, chi2z={chi2z}, incl={incl}, ecc={ecc}")
        # Normalize and build input tensor of shape (L,7)
        theta = np.array([m1,m2,chi1z,chi2z,incl,ecc])
        theta_n = self._normalize_params(theta)
        L = self.waveform_length

        # Replicate params across time steps
        t_n = self.t_norm_array  # shape (L,)
        params_grid = np.tile(theta_n, (L,1))  # (L,6)
        inp = np.concatenate([t_n[:,None], params_grid], axis=1).astype(np.float32)

        with torch.no_grad():
            inp_t = torch.from_numpy(inp).to(self.device)
            amp_pred_n = (
                self.amp_model(inp_t)
                .cpu().numpy()
                .ravel()
            )
            phi_pred = (
                self.phase_model(inp_t[:,:1], inp_t[:,1:])
                .cpu().numpy()
                .ravel()
            )
        self.logger.debug("Prediction completed.")
        return self.t_norm_array, amp_pred_n, phi_pred

    def batch_predict(self, thetas: np.ndarray, batch_size: int = None):
        """
        Vectorized prediction in chunks of `batch_size` (default = how many samples you trained on).
        """
        N = thetas.shape[0]
        L = self.waveform_length

        # Use training set size as default batch_size
        if batch_size is None:
            batch_size = self.train_samples

        results_amp = []
        results_phi = []
        # now chunk in slices of `batch_size`:
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            theta_batch = thetas[start:end]
            theta_n = (theta_batch - self.param_means) / self.param_stds

            t_grid = np.broadcast_to(self.t_norm_array, (end - start, L))
            theta_grid = np.broadcast_to(theta_n[:, None, :], (end - start, L, 6))
            inp = np.concatenate([t_grid[...,None], theta_grid], axis=-1) \
                     .reshape(-1, 7).astype(np.float32)

            with torch.no_grad():
                inp_t     = torch.from_numpy(inp).to(self.device)
                amp_n     = self.amp_model(inp_t).cpu().numpy() \
                              .reshape(end - start, L)
                phi_n     = self.phase_model(inp_t[:, :1], inp_t[:, 1:]) \
                              .cpu().numpy().reshape(end - start, L)

            results_amp.append(self.inverse_log_norm(amp_n))
            results_phi.append(phi_n)

        amp_preds = np.concatenate(results_amp, axis=0)
        phi_preds = np.concatenate(results_phi, axis=0)
        return self.t_norm_array, amp_preds, phi_preds
