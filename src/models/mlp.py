import torch
import torch.nn as nn
from src.data.config import DEVICE


class FourierFeature(nn.Module):
    """
    Maps a D-dim input to a 2*D*B feature vector via sinusoids:
      x → [sin(2π B x), cos(2π B x)]
    where B is a learnable or fixed frequency matrix.
    """
    def __init__(self, in_dim, num_bands=16, max_freq=10.0, learnable=False):
        super().__init__()
        # Create frequency bands (log‐spaced)
        bands = torch.logspace(0., torch.log10(torch.tensor(max_freq)), num_bands)
        # shape: (in_dim, num_bands)
        self.register_buffer('bands', bands.unsqueeze(0).repeat(in_dim, 1))
        if learnable:
            self.bands = nn.Parameter(self.bands)
        self.in_dim = in_dim
        self.num_bands = num_bands

    def forward(self, x):
        # x: (B, in_dim)
        x_exp = x.unsqueeze(-1) * self.bands.unsqueeze(0) * 2 * torch.pi
        # flatten the sin/cos pair: (B, in_dim * num_bands * 2)
        return torch.cat([torch.sin(x_exp), torch.cos(x_exp)], dim=-1).flatten(1)

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return x + self.net(x)

class MLP(nn.Module):
    """
    MLP with BatchNorm, Dropout, and residual skips when dims match.
    """
    def __init__(self, in_dim, hidden_dims, out_dim, dropout=0.1):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
            # if incoming and outgoing dims match, add a residual block
            if prev == h:
                layers.append(ResidualBlock(h, dropout=dropout))
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class PhaseSubNet(nn.Module):
    """
    Phase subnetwork with BN, Dropout, and a residual block.
    """
    def __init__(self, input_dim, hidden_dims, dropout=0.1):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
            if prev == h:
                layers.append(ResidualBlock(h, dropout=dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class PhaseDNN_Full(nn.Module):
    """
    PhaseDNN with Fourier‐feature embedding of θ.
    """
    def __init__(self, param_dim=5, time_dim=1,
                 fourier_bands=16, fourier_max_freq=10.0, fourier_learnable=False,
                 phase_hidden=[128,128,128], N_banks=1, dropout=0.1):
        super().__init__()
        self.N_banks = N_banks

        fourier_dim = param_dim * fourier_bands * 2
        self.theta_ff = FourierFeature(param_dim,
                                       num_bands=fourier_bands,
                                       max_freq=fourier_max_freq,
                                       learnable=fourier_learnable)
        # project down to an embedding
        emb_dim = fourier_dim
        self.theta_proj = nn.Sequential(
            nn.Linear(fourier_dim, emb_dim),
            nn.BatchNorm1d(emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # per‐bank phase subnets
        for i in range(N_banks):
            net_i = PhaseSubNet(input_dim=time_dim + emb_dim,
                                hidden_dims=phase_hidden,
                                dropout=dropout)
            setattr(self, f"phase_net_{i}", net_i)

    def forward(self, t_norm, theta):
        # theta: (B, param_dim)
        ff = self.theta_ff(theta)            # (B, fourier_dim)
        emb = self.theta_proj(ff)            # (B, emb_dim)
        phi_total = torch.zeros_like(t_norm, device=DEVICE)

        x_i = torch.cat([t_norm, emb], dim=-1)
        for i in range(self.N_banks):
            net_i = getattr(self, f"phase_net_{i}")
            phi_total = phi_total + net_i(x_i)
        return phi_total

class AmpSubNet(nn.Module):
    """
    Amplitude subnetwork with BN, Dropout, and an optional residual block.
    """
    def __init__(self, input_dim, hidden_dims, dropout=0.1):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            # if dims match, add a residual skip
            if prev == h:
                layers.append(ResidualBlock(h, dropout=dropout))
            prev = h
        # Predict one amplitude component
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)  # (B,1) component

class AmplitudeDNN_Full(nn.Module):
    """
    AmplitudeDNN with Fourier‐feature embedding of θ.
    """
    def __init__(self,
        in_param_dim=5,
        time_dim=1,
        fourier_bands=16, fourier_max_freq=10.0, fourier_learnable=False,
        amp_hidden=(128,128,128),
        N_banks=2,
        dropout=0.1
    ):
        super().__init__()
        self.N_banks = N_banks

        # Fourier‐feature embed theta
        fourier_dim = in_param_dim * fourier_bands * 2
        self.theta_ff = FourierFeature(in_param_dim,
                                       num_bands=fourier_bands,
                                       max_freq=fourier_max_freq,
                                       learnable=fourier_learnable)
        emb_dim = fourier_dim
        self.theta_proj = nn.Sequential(
            nn.Linear(fourier_dim, emb_dim),
            nn.BatchNorm1d(emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # per‐bank amplitude subnets
        for i in range(N_banks):
            net_i = AmpSubNet(
                input_dim = time_dim + emb_dim,
                hidden_dims = list(amp_hidden),
                dropout = dropout
            )
            setattr(self, f"amp_net_{i}", net_i)

        self.out_act = nn.Sigmoid()

    def forward(self, t_norm, theta):
        ff = self.theta_ff(theta)     # (B, fourier_dim)
        emb = self.theta_proj(ff)     # (B, emb_dim)
        x = torch.cat([t_norm, emb], dim=-1)

        A_total = 0
        for i in range(self.N_banks):
            A_total = A_total + getattr(self, f"amp_net_{i}")(x)
        return self.out_act(A_total)
