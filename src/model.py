import torch
import torch.nn as nn

from src.config import DEVICE

# MODEL DEFINITIONS (with 6‐dim parameter embedding)
class MLP(nn.Module):
    """Simple feedforward MLP with ReLU activations."""
    def __init__(self, in_dim, hidden_dims, out_dim):
        super().__init__()
        layers = []
        prev_dim = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class PhaseSubNet(nn.Module):
    """
    Takes [t_norm, theta_embed] and outputs a scalar phi_i(t; θ).
    We use N_banks=1 here.
    """
    def __init__(self, input_dim, hidden_dims):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, 1))  # output: scalar phi_i
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)  # (batch_size, 1)

class PhaseDNN_Full(nn.Module):
    """
    PhaseDNN that learns phi(t; θ) with θ ∈ R^6 for (m1,m2,chi1z,chi2z,incl,ecc).
    """
    def __init__(self, param_dim=6, time_dim=1, emb_hidden=[64, 64],
                 phase_hidden=[128, 128, 128, 128], N_banks=1):
        super().__init__()
        self.param_dim = param_dim
        self.time_dim  = time_dim
        self.N_banks   = N_banks

        # theta‐embedding network: (m1,m2,chi1z,chi2z,incl,ecc) → emb_dim=64
        emb_dim = 64
        self.theta_embed = MLP(param_dim, emb_hidden, emb_dim)

        # Phase subnets: one per bank
        for i in range(N_banks):
            net_i = PhaseSubNet(input_dim = time_dim + emb_dim,
                                hidden_dims = phase_hidden)
            setattr(self, f"phase_net_{i}", net_i)

    def forward(self, t_norm, theta):
        """
        t_norm: (batch_size,1)
        theta:  (batch_size,6)
        returns phi_total: (batch_size,1)
        """
        emb = self.theta_embed(theta)  # (batch_size, emb_dim)

        phi_total = torch.zeros_like(t_norm).to(DEVICE)  # (batch_size,1)
        for i in range(self.N_banks):
            net_i = getattr(self, f"phase_net_{i}")
            x_i = torch.cat([t_norm, emb], dim=-1)      # (batch_size,1+emb_dim)
            phi_i = net_i(x_i)                            # (batch_size,1)
            w_i = torch.ones_like(t_norm)
            phi_total = phi_total + w_i * phi_i

        return phi_total  # shape (batch_size, 1)

class AmplitudeNet(nn.Module):
    """
    MLP that learns the normalized amplitude A_norm(t; θ) ∈ [0,1].
    Input: [t_norm, m1_norm, m2_norm, chi1z_norm, chi2z_norm, incl_norm, ecc_norm], Output: [0,1].
    """
    def __init__(self, in_dim=7, hidden_dims=[128, 128, 128]):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, 1))
        layers.append(nn.Sigmoid())  # force output in [0,1]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)  # (batch_size, 1)

