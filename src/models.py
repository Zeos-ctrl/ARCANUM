import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.data_generation import *
from src.config import *

# ---------------------------------------------
# Activation and Embedding Helpers
# ---------------------------------------------
class Sine(nn.Module):
    """
    Sine activation for implicit neural representations (SIREN).
    """
    def forward(self, x):
        return torch.sin(x)

class FourierFeature(nn.Module):
    """
    Fourier Feature Mapping: maps a scalar time t to a 2K-dimensional vector
    [sin(2 pi f1 t), cos(2 pi f1 t), ..., sin(2 pi fK t), cos(2 pi fK t)].
    """
    def __init__(self, K: int = 10, max_freq: float = 100.0):
        super().__init__()
        # Frequencies log-uniformly spaced between 1 Hz and max_freq
        freqs = torch.logspace(0.0, math.log10(max_freq), K)
        self.register_buffer('freqs', freqs.unsqueeze(0))  # (1, K)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (batch, 1)
        # output: (batch, 2*K)
        # Compute (batch, K) = t @ freqs
        args = 2 * math.pi * t @ self.freqs  # (batch, K)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

# ---------------------------------------------
# Residual Block
# ---------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout_p=0.2):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.act = nn.ReLU()
        self.do1 = nn.Dropout(p=dropout_p)

        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.do2 = nn.Dropout(p=dropout_p)

    def forward(self, x):
        out = self.act(self.bn1(self.fc1(x)))
        out = self.do1(out)
        out = self.act(self.bn2(self.fc2(out)))
        out = self.do2(out)
        return out + x

# ---------------------------------------------
# Multi-Head Gravitational Wave Surrogate
# ---------------------------------------------
class MultiHeadGWModel(nn.Module):
    """
    Shared backbone with amplitude and phase heads.
    Inputs: physical params (param_dim) + Fourier time features (2K).
    Heads: amplitude A(t), phase phi(t) and frequency freq(t).
    """
    def __init__(self,
                 param_dim: int,
                 fourier_K: int = 10,
                 fourier_max_freq: float = 100.0,
                 hidden_dims: list = [256, 256, 256, 256],
                 dropout_p: float = 0.2):
        super().__init__()
        # Fourier time embedding
        self.time_embed = FourierFeature(K=fourier_K, max_freq=fourier_max_freq)
        in_dim = param_dim + 2 * fourier_K

        # Shared SIREN-style backbone
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(Sine())
            layers.append(nn.Dropout(p=dropout_p))
            if h == prev:
                layers.append(ResidualBlock(h, dropout_p=dropout_p))
            prev = h
        self.backbone = nn.Sequential(*layers)

        # Amplitude head
        self.amp_head = nn.Sequential(
            nn.Linear(prev,128), nn.ReLU(), nn.Dropout(dropout_p),
            nn.Linear(128,64), nn.ReLU(), nn.Dropout(dropout_p),
            nn.Linear(64,1), nn.Sigmoid()
        )
        # Phase head
        self.phase_head = nn.Sequential(
            nn.Linear(prev,128), nn.ReLU(), nn.Dropout(dropout_p),
            nn.Linear(128,64), nn.ReLU(), nn.Dropout(dropout_p),
            nn.Linear(64,1)
        )
        # Frequency head
        self.freq_head = nn.Sequential(
            nn.Linear(prev,128), nn.ReLU(), nn.Dropout(dropout_p),
            nn.Linear(128,1)
        )

    def forward(self, params, t):
        t_feat = self.time_embed(t)
        x = torch.cat([params, t_feat], dim=-1)
        h = self.backbone(x)
        A_hat   = self.amp_head(h)
        phi_hat = self.phase_head(h)
        omega_hat = self.freq_head(h)
        return A_hat, phi_hat, omega_hat

class WaveformPredictor:
    def __init__(self,
                 model_checkpoint: str,
                 param_means: np.ndarray,
                 param_stds:  np.ndarray,
                 feat_means:  np.ndarray,
                 feat_stds:   np.ndarray):
        # Load pretrained model
        self.model = MultiHeadGWModel(
            param_dim=feat_means.shape[0],
            fourier_K=10,
            fourier_max_freq=1.0/(2*DELTA_T),
            hidden_dims=[256,256,256,256],
            dropout_p=0.2
        ).to(DEVICE)
        self.model.load_state_dict(torch.load(model_checkpoint, map_location=DEVICE))
        self.model.eval()

        # normalization stats
        self.feat_means = feat_means
        self.feat_stds  = feat_stds

        # build time axis
        self.common_times, self.N_common = build_common_times(DELTA_T, T_BEFORE, T_AFTER)
        # normalized time grid
        self.time_norm = ((2.0 * (self.common_times + T_BEFORE) / (T_BEFORE + T_AFTER)) - 1.0)

    def predict(self, raw15: np.ndarray) -> np.ndarray:
        """
        Predict strain waveform from a single raw 15-D parameter vector.
        Returns:
          common_times (np.ndarray), h_pred (np.ndarray)
        """
        # 1) feature-engineer and normalize
        feats = compute_engineered_features(raw15[None, :])  # shape (1,11)
        feats_norm = (feats - self.feat_means) / self.feat_stds

        # 2) expand to full time grid
        params = np.repeat(feats_norm, self.N_common, axis=0)
        t_norm = self.time_norm[:, None]

        # 3) convert normalized time to physical time relative to t0
        t_phys = ((t_norm + 1) / 2) * (T_BEFORE + T_AFTER) - T_BEFORE
        t_phys = t_phys.astype(np.float32)

        # 4) run model in batches
        t_norm_tensor = torch.from_numpy(t_norm).to(DEVICE)              # shape (N,1)
        params_tensor = torch.from_numpy(params).to(DEVICE)            # shape (N,11)
        t_phys_tensor = torch.from_numpy(t_phys).to(DEVICE)            # shape (N,1)

        with torch.no_grad():
            A_pred, phi_pred, omega_pred = self.model(params_tensor, t_phys_tensor)
            A_pred = A_pred.cpu().numpy().ravel()
            phi_cum = np.cumsum(phi_pred.cpu().numpy().ravel())
        
        # 5) reconstruct h(t)
        h_pred = A_pred * np.cos(phi_cum)
        return self.common_times, h_pred
