import torch
import torch.nn as nn
from src.config import DEVICE

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
    PhaseDNN that learns φ(t; θ), with a BatchNorm+Dropout embeder and residual PhaseSubNets.
    """
    def __init__(self, param_dim=6, time_dim=1, emb_hidden=[64,64],
                 phase_hidden=[128,128,128,128], N_banks=1, dropout=0.1):
        super().__init__()
        self.N_banks = N_banks

        # embed θ → emb_dim, with BN and Dropout
        emb_dim = emb_hidden[-1]
        self.theta_embed = MLP(param_dim, emb_hidden, emb_dim, dropout=dropout)

        # per‐bank phase subnets
        for i in range(N_banks):
            net_i = PhaseSubNet(input_dim=time_dim+emb_dim,
                                hidden_dims=phase_hidden,
                                dropout=dropout)
            setattr(self, f"phase_net_{i}", net_i)

    def forward(self, t_norm, theta):
        emb = self.theta_embed(theta)           # (B, emb_dim)
        phi_total = torch.zeros_like(t_norm).to(DEVICE)
        for i in range(self.N_banks):
            net_i = getattr(self, f"phase_net_{i}")
            x_i = torch.cat([t_norm, emb], dim=-1)
            phi_i = net_i(x_i)
            phi_total = phi_total + phi_i
        return phi_total

class AmplitudeNet(nn.Module):
    """
    Amplitude MLP with BN, Dropout, and residual skips. Outputs in [0,1].
    """
    def __init__(self, in_dim=7, hidden_dims=[128,128,128], dropout=0.1):
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
            if prev == h:
                layers.append(ResidualBlock(h, dropout=dropout))
            prev = h
        layers += [
            nn.Linear(prev, 1),
            nn.Sigmoid()
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
