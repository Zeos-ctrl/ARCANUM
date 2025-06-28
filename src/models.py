from src.data_generation import *
from src.config import *
import math
import torch
import torch.nn as nn
import torch.fft
import torch.nn.functional as F
# ---------------------------------------------
# Activation & Embedding Helpers
# ---------------------------------------------
class SineActivation(nn.Module):
    def forward(self, x):
        return torch.sin(x)


class TimeEmbedding(nn.Module):
    def __init__(self, num_features: int = 10, max_frequency: float = 100.0):
        super().__init__()
        freqs = torch.logspace(0.0, math.log10(max_frequency), num_features)
        self.register_buffer('freqs', freqs.unsqueeze(0))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        angles = 2 * math.pi * t @ self.freqs
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


# ---------------------------------------------
# Spectral Convolution for FNO
# ---------------------------------------------
class SpectralConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat)
        )
        self.max_modes = modes

    def forward(self, x):
        batch, channels, length = x.shape
        x_freq = torch.fft.rfft(x)
        out_freq = torch.zeros(
            batch, self.weights.shape[1], x_freq.size(-1),
            dtype=torch.cfloat, device=x.device
        )
        modes = min(self.max_modes, x_freq.size(-1))
        out_freq[:, :, :modes] = torch.einsum(
            "bcm, iom -> bom",
            x_freq[:, :, :modes], self.weights[:, :, :modes]
        )
        return torch.fft.irfft(out_freq, n=length)


class FNO1D(nn.Module):
    def __init__(self, modes, width, depth=4):
        super().__init__()
        self.input_proj = nn.Linear(width, width)
        self.spectral_layers = nn.ModuleList([
            SpectralConv1D(width, width, modes) for _ in range(depth)
        ])
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(width, width, kernel_size=1) for _ in range(depth)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(width) for _ in range(depth)
        ])
        self.activation = nn.GELU()

    def forward(self, x):
        # x: (batch, seq_len, width)
        x = self.input_proj(x)
        x = x.permute(0, 2, 1)

        for spec, conv1, norm in zip(
            self.spectral_layers,
            self.conv_layers,
            self.norms
        ):
            s = spec(x)
            c = conv1(x)
            y = (s + c).permute(0, 2, 1)
            y = norm(y)
            y = self.activation(y)
            x = y.permute(0, 2, 1)

        return x.permute(0, 2, 1)


# ---------------------------------------------
# Shared Backbone
# ---------------------------------------------
class SharedBackbone(nn.Module):
    def __init__(
        self,
        param_dim: int,
        seq_len: int,
        time_feat_K: int,
        max_freq: float,
        hidden_width: int
    ):
        super().__init__()
        self.time_embedding = TimeEmbedding(time_feat_K, max_freq)
        self.time_proj = nn.Linear(2 * time_feat_K, hidden_width)
        self.param_proj = nn.Linear(param_dim, hidden_width)

    def forward(self, params, time_grid):
        # params: (batch, param_dim)
        # time_grid: (batch, seq_len, 1)
        batch, T = time_grid.shape
        p = self.param_proj(params).unsqueeze(1)

        tf = self.time_embedding(
            time_grid.reshape(-1, 1)
        ).view(batch, T, -1)

        return p + self.time_proj(tf)


# ---------------------------------------------
# Amplitude & Phase Heads
# ---------------------------------------------
class AmplitudeModel(nn.Module):
    def __init__(
        self,
        backbone: SharedBackbone,
        modes: int,
        width: int,
        depth: int,
        dropout_p: float
    ):
        super().__init__()
        self.backbone = backbone
        self.fno = FNO1D(modes, width, depth)
        self.dropout = nn.Dropout(dropout_p)
        self.out = nn.Sequential(
            nn.LayerNorm(width),
            nn.GELU(),
            nn.Linear(width, 1),
            nn.Sigmoid()
        )

    def forward(self, params, time_grid):
        x = self.backbone(params, time_grid)
        h = self.fno(x)
        return self.out(self.dropout(h)).squeeze(-1)


class PhaseModel(nn.Module):
    def __init__(
        self,
        backbone: SharedBackbone,
        modes: int,
        width: int,
        depth: int,
        dropout_p: float
    ):
        super().__init__()
        self.backbone = backbone
        self.fno = FNO1D(modes, width, depth)
        self.dropout = nn.Dropout(dropout_p)
        self.out = nn.Sequential(
            nn.LayerNorm(width),
            nn.GELU(),
            nn.Linear(width, 1)
        )

    def forward(self, params, time_grid):
        x = self.backbone(params, time_grid)
        h = self.fno(x)
        return self.out(self.dropout(h)).squeeze(-1)


# ---------------------------------------------
# Waveform Predictor
# ---------------------------------------------
class WaveformPredictor:
    def __init__(
        self,
        checkpoint_path: str,
        feature_means: np.ndarray,
        feature_stds: np.ndarray
    ):
        # Initialize amplitude model
        backbone_amp = SharedBackbone(
            param_dim=feature_means.shape[0],
            seq_len=int((T_BEFORE + T_AFTER) / DELTA_T),
            time_feat_K=10,
            max_freq=1/(2*DELTA_T),
            hidden_width=128
        )
        self.amp_model = AmplitudeModel(
            backbone=backbone_amp,
            modes=16,
            width=128,
            depth=4,
            dropout_p=0.1
        )

        # Initialize phase model
        backbone_phase = SharedBackbone(
            param_dim=feature_means.shape[0],
            seq_len=int((T_BEFORE + T_AFTER) / DELTA_T),
            time_feat_K=10,
            max_freq=1/(2*DELTA_T),
            hidden_width=128
        )
        self.phase_model = PhaseModel(
            backbone=backbone_phase,
            modes=16,
            width=128,
            depth=4,
            dropout_p=0.0
        )

        # Load checkpoint
        ckpt = torch.load(checkpoint_path, map_location=DEVICE)
        self.amp_model.load_state_dict(ckpt['amp_state'])
        self.phase_model.load_state_dict(ckpt['phase_state'])
        self.amp_model.to(DEVICE).eval()
        self.phase_model.to(DEVICE).eval()

        # Normalization stats
        self.feature_means = torch.tensor(feature_means, device=DEVICE)
        self.feature_stds = torch.tensor(feature_stds, device=DEVICE)

        # Time grid
        self.time_values, self.seq_len = build_common_times(DELTA_T, T_BEFORE, T_AFTER)
        self.time_norm = (
            (2 * (self.time_values + T_BEFORE) / (T_BEFORE + T_AFTER)) - 1
        ).astype(np.float32)

    def predict(self, raw_params: np.ndarray):
        # 1) Feature engineering & normalization
        feats = compute_engineered_features(raw_params[None, :])
        feats_norm = (feats - self.feature_means.cpu().numpy()) / self.feature_stds.cpu().numpy()

        # 2) Expand to full time grid
        params_grid = np.repeat(feats_norm, self.seq_len, axis=0)
        t_norm = self.time_norm[:, None]
        t_phys = ((t_norm + 1) / 2) * (T_BEFORE + T_AFTER) - T_BEFORE

        # 3) Model inference
        with torch.no_grad():
            amp_pred = self.amp_model(
                torch.from_numpy(params_grid).to(DEVICE),
                torch.from_numpy(t_phys).to(DEVICE)
            )
            dphi_pred = self.phase_model(
                torch.from_numpy(params_grid).to(DEVICE),
                torch.from_numpy(t_phys).to(DEVICE)
            )

        amp_pred = amp_pred.cpu().numpy().ravel()
        dphi_pred = dphi_pred.cpu().numpy().ravel()

        # 4) Reconstruct waveform
        phase = np.cumsum(dphi_pred)
        strain = amp_pred * np.cos(phase)

        return self.time_values, strain
