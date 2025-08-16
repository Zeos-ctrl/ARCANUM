from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import requests
import torch
import torch.nn as nn
from numpy.fft import rfft
from numpy.fft import rfftfreq
from pycbc.psd import aLIGOZeroDetHighPower
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from src.data.config import *
from src.data.dataset import unscale_target
from src.models.model_factory import make_amp_model
from src.models.model_factory import make_phase_model

logger = logging.getLogger(__name__)


def save_checkpoint(
    checkpoint_dir, amp_model, phase_model, data,
    amp_weight_var, amp_bias_var,
    phase_weight_var, phase_bias_var,
    noise_variance,
):
    """
    Save important model information
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Model weights
    torch.save(
        amp_model.state_dict(),
        os.path.join(checkpoint_dir, 'amp_model.pt'),
    )
    torch.save(
        phase_model.state_dict(),
        os.path.join(checkpoint_dir, 'phase_model.pt'),
    )

    # Normalization stats & constants
    np.save(os.path.join(checkpoint_dir, 'param_means.npy'), data.param_means)
    np.save(os.path.join(checkpoint_dir, 'param_stds.npy'),  data.param_stds)
    np.save(os.path.join(checkpoint_dir, 't_norm_array.npy'), data.t_norm_array)

    # Hessian‐diag variances
    np.save(
        os.path.join(checkpoint_dir, 'amp_last_weight_variances.npy'),
        amp_weight_var.cpu().numpy(),
    )
    np.save(
        os.path.join(checkpoint_dir, 'amp_last_bias_variance.npy'),
        amp_bias_var.cpu().numpy(),
    )
    np.save(
        os.path.join(checkpoint_dir, 'phase_last_weight_variances.npy'),
        phase_weight_var.cpu().numpy(),
    )
    np.save(
        os.path.join(checkpoint_dir, 'phase_last_bias_variance.npy'),
        phase_bias_var.cpu().numpy(),
    )

    # Metadata JSON
    meta = {
        'waveform_length':   WAVEFORM_LENGTH,
        'delta_t':           DELTA_T,
        'amp_scale':       float(data.amp_scale),
        'train_samples':     NUM_SAMPLES,
        'hessian_noise_var': noise_variance,
        'in_dim': TRAIN_FEATURES,
        'in_dim_len': len(TRAIN_FEATURES)
    }
    with open(os.path.join(checkpoint_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f)


def notify_discord(message: str, url: str = None):
    """
    Send a notification message to Discord via webhook.
    """
    hook = url or DISCORD_WEBHOOK_URL
    if not hook:
        logger.warning(
            'No DISCORD_WEBHOOK_URL configured—skipping Discord notification.')
        return

    payload = {'content': message}
    headers = {'Content-Type': 'application/json'}
    try:
        resp = requests.post(hook, json=payload, headers=headers, timeout=5)
        resp.raise_for_status()
        logger.info('Sent Discord notification.')
    except Exception as e:
        logger.error('Failed to send Discord notification: %s', e)


def compute_last_layer_hessian_diag(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    noise_var: float = 1.0,
):
    """
    Approximate the diagonal of the Hessian of MSELoss wrt the final Linear layer’s
    weights and bias, using a forward‐hook to capture the penultimate features phi.
    Returns:
      weight_variances: Tensor of shape (1, in_features)
      bias_variance:    Tensor scalar shape (1,)
    """
    linears = [
        m for m in model.modules()
        if isinstance(m, nn.Linear) and m.out_features == 1
    ]
    assert linears, 'No final Linear layer with out_features=1 found!'
    final_linear = linears[-1]

    # prepare accumulators
    W = final_linear.weight
    B = final_linear.bias
    H_w = torch.zeros_like(W, device=device)
    H_b = torch.zeros_like(B, device=device)

    # hook to grab the input features phi whenever final_linear runs
    captured = {'phi': None}

    def hook_fn(module, inputs, output):
        captured['phi'] = inputs[0].detach()

    hook = final_linear.register_forward_hook(hook_fn)

    # sweep once over the loader
    model.eval()
    with torch.no_grad():
        for batch in loader:
            X, Y = batch[0].to(device), batch[1].to(device)
            _ = model(X[:, :1], X[:, 1:])

            phi = captured['phi']
            B_size = phi.shape[0]

            H_w += (phi ** 2).sum(dim=0, keepdim=True) / noise_var
            H_b += torch.ones_like(B) * (B_size / noise_var)

    hook.remove()

    weight_variances = 1.0 / H_w
    bias_variance = 1.0 / H_b

    return weight_variances, bias_variance


def compute_match(h1, h2, dt, f_low=20.0):
    from pycbc.types import TimeSeries
    from pycbc.filter import match
    h1 = np.asarray(h1, dtype=np.float64)
    h2 = np.asarray(h2, dtype=np.float64)

    h1_ts = TimeSeries(h1, delta_t=dt)
    h2_ts = TimeSeries(h2, delta_t=dt)

    # Match length
    tlen = max(len(h1_ts), len(h2_ts))
    h1_ts.resize(tlen)
    h2_ts.resize(tlen)

    # PSD
    delta_f = 1.0 / h1_ts.duration
    flen = tlen//2 + 1
    psd = aLIGOZeroDetHighPower(flen, delta_f, f_low).astype(np.float64)

    m, _ = match(h1_ts, h2_ts, psd=psd, low_frequency_cutoff=f_low)
    return m


@dataclass
class TimeSeriesStrainData:
    data: np.ndarray        # Data array of the waveform
    uncertainty: np.ndarray  # Data array of uncertainty for every datapoint
    epoch: float            # Start time for the waveform
    sample_rate: float      # delta_t
    time: np.ndarray        # Normalized time array
    approximant: str        # approximant used in training


class WaveformPredictor:
    def __init__(self, checkpoint_dir: str, model: str, device: str = DEVICE):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(
            f"Initializing WaveformPredictor from '{checkpoint_dir}'")
        self.device = torch.device(device)

        # Load normalization statistics
        self.param_means = np.load(os.path.join(
            checkpoint_dir, model, 'param_means.npy'))
        self.param_stds = np.load(os.path.join(
            checkpoint_dir, model, 'param_stds.npy'))
        self.time_norm_array = np.load(os.path.join(
            checkpoint_dir, model, 't_norm_array.npy'))

        # Load metadata
        meta_path = os.path.join(checkpoint_dir, model, 'meta.json')
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"No meta.json in {checkpoint_dir}")
        with open(meta_path) as meta_file:
            meta = json.load(meta_file)

        amp_path = os.path.join(checkpoint_dir, model, 'amp_params.json')
        if not os.path.exists(amp_path):
            raise FileNotFoundError(
                f"No amp_params.json in {checkpoint_dir}{model}")
        with open(amp_path) as meta_file:
            amp_params = json.load(meta_file)

        phase_path = os.path.join(checkpoint_dir, model, 'phase_params.json')
        if not os.path.exists(phase_path):
            raise FileNotFoundError(
                f"No phase_params.json in {checkpoint_dir}{model}")
        with open(phase_path) as meta_file:
            phase_params = json.load(meta_file)

        self.amp_scale = meta['amp_scale']
        self.waveform_length = int(meta['waveform_length'])
        self.delta_t = float(meta['delta_t'])
        self.train_samples = int(meta.get('train_samples', 0))
        self.in_dim_len = int(meta.get('in_dim_len'))

        # Build and load models
        self.amp_model = make_amp_model(
            in_param_dim=self.in_dim_len, params=amp_params).to(self.device)

        self.phase_model = make_phase_model(
            param_dim=self.in_dim_len, params=phase_params).to(self.device)

        self.amp_model.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, model,
                       'amp_model.pt'), map_location=self.device),
        )
        self.phase_model.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, model,
                       'phase_model.pt'), map_location=self.device),
        )

        # Load Laplace variances
        def load_variance_file(filename: str) -> torch.Tensor:
            return torch.from_numpy(
                np.load(os.path.join(checkpoint_dir, model, filename)),
            ).to(self.device)

        self.amp_last_weight_variances = load_variance_file(
            'amp_last_weight_variances.npy')
        self.amp_last_bias_variance = load_variance_file(
            'amp_last_bias_variance.npy')
        self.phase_last_weight_variances = load_variance_file(
            'phase_last_weight_variances.npy')
        self.phase_last_bias_variance = load_variance_file(
            'phase_last_bias_variance.npy')

        self.amp_model.eval()
        self.phase_model.eval()
        self.logger.info(f"Using {self.device}...")
        self.logger.info('Models and variances loaded; in eval mode.')

    def _compute_derived(self, m1: float, m2: float, chi1z: float, chi2z: float, inclination: float, eccentricity: float) -> np.ndarray:
        """Compute derived features dynamically based on TRAIN_FEATURES list."""
        # base parameter dict
        param_map = {
            'chirp_mass': (m1 * m2) ** (3.0/5.0) / (m1 + m2) ** (1.0/5.0),
            'symmetric_mass_ratio': (m1 * m2) / (m1 + m2) ** 2,
            'effective_spin': (m1 * chi1z + m2 * chi2z) / (m1 + m2),
            'inclination': inclination,
            'eccentricity': eccentricity,
        }
        # construct array in order of TRAIN_FEATURES
        return np.array([param_map[feat] for feat in TRAIN_FEATURES], dtype=np.float32)

    def _normalize_derived(self, derived_array: np.ndarray) -> np.ndarray:
        """Normalize derived parameters using stored means and standard deviations."""
        return (derived_array - self.param_means) / self.param_stds

    def predict(
        self,
        m1: float,
        m2: float,
        spin1_z: float,
        spin2_z: float,
        inclination: float,
        eccentricity: float,
        waveform_length: int | None = None,
        sampling_dt: float | None = None,
    ) -> tuple[TimeSeriesStrainData, TimeSeriesStrainData]:
        """
        Returns the predicted h_plus and h_cross waveforms for a single set of input parameters.
        """
        length = waveform_length or self.waveform_length
        delta_t = sampling_dt or self.delta_t

        # Create time arrays
        time_array = np.linspace(-length * delta_t, 0.0, length)
        normalized_time = 2 * (time_array + length *
                               delta_t) / (length * delta_t) - 1

        # Compute and normalize derived features
        derived_features = self._compute_derived(
            m1, m2, spin1_z, spin2_z, inclination, eccentricity)
        normalized_features = self._normalize_derived(derived_features)

        # Build model input tensor
        parameter_matrix = np.tile(normalized_features, (length, 1))
        model_input = np.concatenate(
            [
                normalized_time.reshape(-1, 1),
                parameter_matrix,
            ], axis=1,
        ).astype(np.float32)
        input_tensor = torch.from_numpy(model_input).to(self.device)

        # Model forward pass
        with torch.no_grad():
            log_amplitude_predictions = self.amp_model(
                input_tensor[:, :1], input_tensor[:, 1:]).cpu().numpy().ravel()
            phase_predictions = self.phase_model(
                input_tensor[:, :1], input_tensor[:, 1:]).cpu().numpy().ravel()

        # Unscale amplitude and compute polarizations
        amplitude = unscale_target(log_amplitude_predictions, self.amp_scale)
        cos_inclination = np.cos(inclination)

        h_plus_data = amplitude * \
            ((1 + cos_inclination**2) / 2) * np.cos(phase_predictions)
        h_cross_data = amplitude * cos_inclination * np.sin(phase_predictions)

        # Wrap into TimeSeriesStrainData objects
        plus_waveform = TimeSeriesStrainData(
            data=h_plus_data,
            uncertainty=None,
            epoch=time_array[0],
            sample_rate=delta_t,
            time=normalized_time,
            approximant=WAVEFORM,
        )
        cross_waveform = TimeSeriesStrainData(
            data=h_cross_data,
            uncertainty=None,
            epoch=time_array[0],
            sample_rate=delta_t,
            time=normalized_time,
            approximant=WAVEFORM,
        )

        return plus_waveform, cross_waveform

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

        amp_mat = unscale_target(Amp_mat, self.amp_scale)
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
        waveform_length: int | None = None,
        sampling_dt: float | None = None,
        sigma_level: int = 3,
    ) -> tuple[TimeSeriesStrainData, TimeSeriesStrainData]:
        """
        Single-sample prediction with uncertainty.
        Returns two TimeSeriesStrainData objects (plus and cross) including uncertainties.
        """
        length = waveform_length or self.waveform_length
        delta_t = sampling_dt or self.delta_t

        # Create time arrays
        time_array = np.linspace(-length * delta_t, 0.0, length)
        normalized_time = 2 * (time_array + length *
                               delta_t) / (length * delta_t) - 1

        # Compute and normalize derived features
        derived_features = self._compute_derived(
            m1, m2, spin1_z, spin2_z, inclination, eccentricity)
        normalized_features = self._normalize_derived(derived_features)

        # Prepare input tensor
        parameter_matrix = np.tile(normalized_features, (length, 1))
        model_input = np.concatenate(
            [
                normalized_time.reshape(-1, 1),
                parameter_matrix,
            ], axis=1,
        ).astype(np.float32)
        input_tensor = torch.from_numpy(model_input).to(self.device)

        # Hook last linear layers to capture features
        def find_last_linear_layer(model: nn.Module) -> nn.Linear:
            return [layer for layer in model.modules() if isinstance(layer, nn.Linear) and layer.out_features == 1][-1]

        amp_last_linear = find_last_linear_layer(self.amp_model)
        phase_last_linear = find_last_linear_layer(self.phase_model)
        captured_features: dict = {}
        amp_hook = amp_last_linear.register_forward_hook(
            lambda module, inp, out: captured_features.update(amp_hidden=inp[0].detach()))
        phase_hook = phase_last_linear.register_forward_hook(
            lambda module, inp, out: captured_features.update(phase_hidden=inp[0].detach()))

        # Forward pass
        with torch.no_grad():
            log_amp_means = self.amp_model(
                input_tensor[:, :1], input_tensor[:, 1:])
            phase_means = self.phase_model(
                input_tensor[:, :1], input_tensor[:, 1:])
        amp_hook.remove()
        phase_hook.remove()

        # Extract captured hidden features
        hidden_amp_features = captured_features['amp_hidden']
        hidden_phase_features = captured_features['phase_hidden']

        # Compute prediction variances
        variance_log_amp = (hidden_amp_features**2 * self.amp_last_weight_variances).sum(
            1, True) + self.amp_last_bias_variance
        variance_phase = (hidden_phase_features**2 * self.phase_last_weight_variances).sum(
            1, True) + self.phase_last_bias_variance

        # Convert to numpy arrays
        mean_log_amp = log_amp_means.cpu().numpy().ravel()
        std_log_amp = np.sqrt(
            variance_log_amp.cpu().numpy().ravel()) * sigma_level
        mean_phase = phase_means.cpu().numpy().ravel()
        std_phase = np.sqrt(variance_phase.cpu().numpy().ravel()) * sigma_level

        # Unscale amplitude and propagate uncertainty
        amplitude = unscale_target(mean_log_amp, self.amp_scale)
        cos_inclination = np.cos(inclination)
        plus_mean = amplitude * \
            ((1 + cos_inclination**2) / 2) * np.cos(mean_phase)
        plus_uncertainty = np.sqrt(
            (plus_mean * std_log_amp)**2 +
            (amplitude * ((1 + cos_inclination**2) / 2)
             * np.sin(mean_phase) * std_phase)**2,
        )
        cross_mean = amplitude * cos_inclination * np.sin(mean_phase)
        cross_uncertainty = np.sqrt(
            (cross_mean * std_log_amp)**2 +
            (amplitude * cos_inclination * np.cos(mean_phase) * std_phase)**2,
        )

        plus_waveform = TimeSeriesStrainData(
            data=plus_mean,
            uncertainty=plus_uncertainty,
            epoch=time_array[0],
            sample_rate=delta_t,
            time=normalized_time,
            approximant=WAVEFORM,
        )
        cross_waveform = TimeSeriesStrainData(
            data=cross_mean,
            uncertainty=cross_uncertainty,
            epoch=time_array[0],
            sample_rate=delta_t,
            time=normalized_time,
            approximant=WAVEFORM,
        )

        return plus_waveform, cross_waveform

    def batch_predict_with_uncertainty(
        self,
        thetas_raw: np.ndarray,
        batch_size: int | None = None,
        sigma_level: int = 3,
    ) -> tuple[list[TimeSeriesStrainData], list[TimeSeriesStrainData]]:
        """
        Using GPU batching, predict waveforms and uncertainties for multiple parameter sets.
        Returns lists of TimeSeriesStrainData for plus and cross polarizations.
        """
        num_samples = thetas_raw.shape[0]
        length = self.waveform_length
        delta_t = self.delta_t
        batch_size = batch_size or self.train_samples

        # Precompute time arrays
        time_array = np.linspace(-length * delta_t, 0.0, length)
        normalized_time = 2 * (time_array + length *
                               delta_t) / (length * delta_t) - 1

        def find_last_linear_layer(model: nn.Module) -> nn.Linear:
            return [layer for layer in model.modules() if isinstance(layer, nn.Linear) and layer.out_features == 1][-1]

        amp_last_linear = find_last_linear_layer(self.amp_model)
        phase_last_linear = find_last_linear_layer(self.phase_model)

        plus_waveforms: list[TimeSeriesStrainData] = []
        cross_waveforms: list[TimeSeriesStrainData] = []
        feature_count = len(TRAIN_FEATURES)

        for batch_start in range(0, num_samples, batch_size):
            batch_end = min(batch_start + batch_size, num_samples)
            parameter_batch = thetas_raw[batch_start:batch_end]
            current_batch_size = batch_end - batch_start

            # Compute and normalize derived features
            derived_matrix = np.stack(
                [
                    self._compute_derived(*params) for params in parameter_batch
                ], axis=0,
            )
            normalized_matrix = self._normalize_derived(derived_matrix)

            # Build flattened input
            time_block = np.broadcast_to(
                normalized_time, (current_batch_size, length))
            feature_block = np.broadcast_to(
                normalized_matrix[:, None, :],
                (current_batch_size, length, feature_count),
            )
            flat_input = np.concatenate(
                [time_block[..., None], feature_block], axis=-1)
            flat_input = flat_input.reshape(-1,
                                            1 + feature_count).astype(np.float32)
            input_tensor = torch.from_numpy(flat_input).to(self.device)

            # Hook and forward
            captured_features: dict = {}
            amp_hook = amp_last_linear.register_forward_hook(
                lambda module, inp, out: captured_features.update(amp_hidden=inp[0].detach()))
            phase_hook = phase_last_linear.register_forward_hook(
                lambda module, inp, out: captured_features.update(phase_hidden=inp[0].detach()))
            with torch.no_grad():
                log_amp_means = self.amp_model(
                    input_tensor[:, :1], input_tensor[:, 1:])
                phase_means = self.phase_model(
                    input_tensor[:, :1], input_tensor[:, 1:])
            amp_hook.remove()
            phase_hook.remove()

            # Reshape and compute variances
            hidden_amp = captured_features['amp_hidden']
            hidden_phase = captured_features['phase_hidden']
            variance_log_amp = (
                hidden_amp**2 * self.amp_last_weight_variances).sum(1, True) + self.amp_last_bias_variance
            variance_phase = (hidden_phase**2 * self.phase_last_weight_variances).sum(
                1, True) + self.phase_last_bias_variance

            mean_log_amp = log_amp_means.cpu().numpy().reshape(current_batch_size, length)
            std_log_amp = np.sqrt(variance_log_amp.cpu().numpy().reshape(
                current_batch_size, length)) * sigma_level
            mean_phase = phase_means.cpu().numpy().reshape(current_batch_size, length)
            std_phase = np.sqrt(variance_phase.cpu().numpy().reshape(
                current_batch_size, length)) * sigma_level

            amplitude_matrix = unscale_target(mean_log_amp, self.amp_scale)
            cos_inclinations = np.cos(parameter_batch[:, 4])[:, None]

            plus_mean_block = amplitude_matrix * \
                ((1 + cos_inclinations**2) / 2) * np.cos(mean_phase)
            plus_uncertainty_block = np.sqrt(
                (plus_mean_block * std_log_amp)**2 +
                (amplitude_matrix * ((1 + cos_inclinations**2) / 2)
                 * np.sin(mean_phase) * std_phase)**2,
            )
            cross_mean_block = amplitude_matrix * \
                cos_inclinations * np.sin(mean_phase)
            cross_uncertainty_block = np.sqrt(
                (cross_mean_block * std_log_amp)**2 +
                (amplitude_matrix * cos_inclinations *
                 np.cos(mean_phase) * std_phase)**2,
            )

            for i in range(current_batch_size):
                plus_waveforms.append(
                    TimeSeriesStrainData(
                        data=plus_mean_block[i], uncertainty=plus_uncertainty_block[i],
                        epoch=time_array[0], sample_rate=delta_t, time=normalized_time, approximant=WAVEFORM,
                    ),
                )
                cross_waveforms.append(
                    TimeSeriesStrainData(
                        data=cross_mean_block[i], uncertainty=cross_uncertainty_block[i],
                        epoch=time_array[0], sample_rate=delta_t, time=normalized_time, approximant=WAVEFORM,
                    ),
                )

        return plus_waveforms, cross_waveforms

    def predict_debug(
        self,
        m1: float,
        m2: float,
        spin1_z: float,
        spin2_z: float,
        inclination: float,
        eccentricity: float,
        waveform_length: int | None = None,
        sampling_dt: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Like predict(), but returns raw time array, linear amplitude matrix, and phase array.
        Returns: time_array (length,), amplitude_array (length,), phase_array (length,)
        """
        length = waveform_length or self.waveform_length
        delta_t = sampling_dt or self.delta_t
        time_array = np.linspace(-length * delta_t, 0.0, length)
        normalized_time = 2 * (time_array + length *
                               delta_t) / (length * delta_t) - 1

        derived_features = self._compute_derived(
            m1, m2, spin1_z, spin2_z, inclination, eccentricity)
        normalized_features = self._normalize_derived(derived_features)

        parameter_matrix = np.tile(normalized_features, (length, 1))
        model_input = np.concatenate(
            [
                normalized_time.reshape(-1, 1),
                parameter_matrix,
            ], axis=1,
        ).astype(np.float32)
        input_tensor = torch.from_numpy(model_input).to(self.device)

        with torch.no_grad():
            log_amplitude_predictions = self.amp_model(
                input_tensor[:, :1], input_tensor[:, 1:]).cpu().numpy().ravel()
            phase_predictions = self.phase_model(
                input_tensor[:, :1], input_tensor[:, 1:]).cpu().numpy().ravel()

        amplitude = unscale_target(log_amplitude_predictions, self.amp_scale)
        return time_array, amplitude, phase_predictions

    def batch_predict_debug(self, thetas_raw: np.ndarray, batch_size: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Like batch_predict, but returns raw time array, amplitude matrix, and phase matrix.
        Returns: time_array (length,), amplitude_matrix (num_samples, length), phase_matrix (num_samples, length)
        """
        num_samples = thetas_raw.shape[0]
        length = self.waveform_length
        delta_t = self.delta_t
        batch_size = batch_size or self.train_samples

        time_array = np.linspace(-length * delta_t, 0.0, length)
        normalized_time = 2 * (time_array + length *
                               delta_t) / (length * delta_t) - 1

        amplitude_list: list[np.ndarray] = []
        phase_list: list[np.ndarray] = []

        for batch_start in range(0, num_samples, batch_size):
            batch_end = min(batch_start + batch_size, num_samples)
            parameter_batch = thetas_raw[batch_start:batch_end]

            derived_matrix = np.stack(
                [
                    self._compute_derived(*params) for params in parameter_batch
                ], axis=0,
            )
            normalized_matrix = self._normalize_derived(derived_matrix)

            time_block = np.broadcast_to(
                normalized_time, (parameter_batch.shape[0], length))
            feature_block = np.broadcast_to(
                normalized_matrix[:, None, :],
                (parameter_batch.shape[0], length, len(TRAIN_FEATURES)),
            )
            flat_input = np.concatenate(
                [time_block[..., None], feature_block], axis=-1)
            flat_input = flat_input.reshape(-1, 1 +
                                            len(TRAIN_FEATURES)).astype(np.float32)
            input_tensor = torch.from_numpy(flat_input).to(self.device)

            with torch.no_grad():
                amp_output_block = self.amp_model(
                    input_tensor[:, :1], input_tensor[:, 1:]).cpu().numpy().reshape(-1, length)
                phase_output_block = self.phase_model(
                    input_tensor[:, :1], input_tensor[:, 1:]).cpu().numpy().reshape(-1, length)

            amplitude_list.append(unscale_target(
                amp_output_block, self.amp_scale))
            phase_list.append(phase_output_block)

        amplitude_matrix = np.vstack(amplitude_list)
        phase_matrix = np.vstack(phase_list)
        return time_array, amplitude_matrix, phase_matrix
