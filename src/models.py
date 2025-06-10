import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """
    A single MLP residual block:
      x → Linear → BatchNorm → ReLU → Dropout
        → Linear → BatchNorm → ReLU → Dropout
        → (+ x)
    Only adds the input back in if dims match.
    """
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
        return out + x  # only valid if input/output dims match

class MLP(nn.Module):
    """Feed-forward MLP with BatchNorm, ReLU, Dropout, and residual blocks."""
    def __init__(self, in_dim, hidden_dims, out_dim, dropout_p=0.2):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_p))
            # If the next hidden size equals the current, add a ResidualBlock
            if h == prev:
                layers.append(ResidualBlock(h, dropout_p=dropout_p))
            prev = h

        layers.append(nn.Linear(prev, out_dim))
        layers.append(nn.Sigmoid() if out_dim == 1 else nn.Identity())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class PhaseSubNet(nn.Module):
    """
    Takes [t_norm, theta_embed] → MLP with BatchNorm/ReLU/Dropout → scalar Δφ_i(t;θ).
    """
    def __init__(self, input_dim, hidden_dims, dropout_p=0.2):
        super().__init__()
        body = []
        prev = input_dim
        for h in hidden_dims:
            body.append(nn.Linear(prev, h))
            body.append(nn.BatchNorm1d(h))
            body.append(nn.ReLU())
            body.append(nn.Dropout(p=dropout_p))
            # Add residual if dims match
            if h == prev:
                body.append(ResidualBlock(h, dropout_p=dropout_p))
            prev = h

        self.net_body = nn.Sequential(*body)
        self.linear_out = nn.Linear(prev, 1)

    def forward(self, x):
        h = self.net_body(x)       # (batch, hidden_dims[-1])
        return self.linear_out(h)  # (batch, 1)

class PhaseDNN_Full(nn.Module):
    """
    PhaseDNN that learns Δφ(t; θ) with θ ∈ R^15, using N_banks = 3.
    Now with Dropout in each PhaseSubNet.
    """
    def __init__(self, param_dim=15, time_dim=1,
                 emb_hidden=[128,128], emb_dim=128,
                 phase_hidden=[256,256,256,256], N_banks=3,
                 dropout_p=0.2):
        super().__init__()
        self.theta_embed = MLP(param_dim, emb_hidden, emb_dim, dropout_p=dropout_p)
        self.N_banks = N_banks
        for i in range(N_banks):
            sub = PhaseSubNet(input_dim=time_dim + emb_dim,
                              hidden_dims=phase_hidden,
                              dropout_p=dropout_p)
            setattr(self, f"phase_net_{i}", sub)

    def forward(self, t_norm, theta):
        emb = self.theta_embed(theta)                     # (batch, emb_dim)
        dphi = torch.zeros_like(t_norm)                   # (batch,1)
        for i in range(self.N_banks):
            net_i = getattr(self, f"phase_net_{i}")
            x_i = torch.cat([t_norm, emb], dim=-1)        # (batch,1 + emb_dim)
            dphi = dphi + net_i(x_i)                      # (batch,1)
        return dphi

class AmplitudeNet(nn.Module):
    """
    MLP that learns normalized amplitude A_norm ∈ [0,1].
    Now with BatchNorm, Dropout, and residuals.
    """
    def __init__(self, in_dim=16, hidden_dims=[256,256,256], dropout_p=0.2):
        super().__init__()
        body = []
        prev = in_dim
        for h in hidden_dims:
            body.append(nn.Linear(prev, h))
            body.append(nn.BatchNorm1d(h))
            body.append(nn.ReLU())
            body.append(nn.Dropout(p=dropout_p))
            if h == prev:
                body.append(ResidualBlock(h, dropout_p=dropout_p))
            prev = h

        self.net_body = nn.Sequential(*body)
        self.linear_out = nn.Linear(prev, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h = self.net_body(x)
        return self.sigmoid(self.linear_out(h))
