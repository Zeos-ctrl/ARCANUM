import os
import json
import torch
import logging
import numpy as np

from src.model import *
from src.config import *

def save_checkpoint(checkpoint_dir, amp_model, phase_model,
                    param_means, param_stds, t_norm_array):
    logging.info(f"Saving checkpoint to '{checkpoint_dir}'")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Model weights
    torch.save(amp_model.state_dict(),
               os.path.join(checkpoint_dir, "amp_model.pt"))
    torch.save(phase_model.state_dict(),
               os.path.join(checkpoint_dir, "phase_model.pt"))
    logging.debug("Saved model weights.")

    # Normalization stats & constants
    np.save(os.path.join(checkpoint_dir, "param_means.npy"), param_means)
    np.save(os.path.join(checkpoint_dir, "param_stds.npy"),  param_stds)
    np.save(os.path.join(checkpoint_dir, "t_norm_array.npy"), t_norm_array)
    logging.debug("Saved normalization stats and constants.")

    # Save metadata JSON
    meta = {
        "waveform_length": WAVEFORM_LENGTH,
        "delta_t": DELTA_T,
        "device": str(DEVICE)
    }
    with open(os.path.join(checkpoint_dir, "meta.json"), "w") as f:
        json.dump(meta, f)
    logging.info("Saved metadata JSON.")

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
            self.waveform_length = meta["waveform_length"]
            self.delta_t         = meta["delta_t"]
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

    def predict(self, m1, m2, chi1z, chi2z, incl, ecc):
        self.logger.debug(f"Predict called with params: m1={m1}, m2={m2}, chi1z={chi1z}, chi2z={chi2z}, incl={incl}, ecc={ecc}")
        # 1. Normalize and build input tensor of shape (L,7)
        theta = np.array([m1,m2,chi1z,chi2z,incl,ecc])
        theta_n = self._normalize_params(theta)
        L = self.waveform_length

        # Replicate params across time steps
        t_n = self.t_norm_array  # shape (L,)
        params_grid = np.tile(theta_n, (L,1))  # (L,6)
        inp = np.concatenate([t_n[:,None], params_grid], axis=1).astype(np.float32)

        # 2. Predict
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
