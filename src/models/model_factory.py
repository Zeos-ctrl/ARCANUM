import torch.nn as nn
from src.data.config import *
from src.models.mlp import AmplitudeDNN_Full, PhaseDNN_Full

def make_amp_model(
    in_param_dim: int,
) -> nn.Module:
    """
    Returns the amplitude model specified by MODEL.amp_model_type,
    instantiated with the hyperparameters in MODEL.*.
    """
    mtype = MODEL_TYPE
    if mtype == "mlp":
        return AmplitudeDNN_Full(
            in_param_dim=in_param_dim,
            time_dim=1,
            fourier_bands=AMP_FOURIER_BANDS,         # e.g. 16
            fourier_max_freq=AMP_FOURIER_MAX_FREQ,   # e.g. 10.0
            fourier_learnable=AMP_FOURIER_LEARNABLE, # True/False
            amp_hidden=AMP_HIDDEN,
            N_banks=AMP_BANKS,
            dropout=AMP_DROPOUT
        )
    else:
        raise ValueError(f"Unknown amp_model_type '{MODEL_TYPE}'")


def make_phase_model(
    param_dim: int,
) -> nn.Module:
    """
    Returns the phase model specified by MODEL.phase_model_type,
    instantiated with the hyperparameters in MODEL.*.
    """
    mtype = MODEL_TYPE
    if mtype == "mlp":
        return PhaseDNN_Full(
            param_dim=param_dim,
            time_dim=1,
            fourier_bands=PHASE_FOURIER_BANDS,
            fourier_max_freq=PHASE_FOURIER_MAX_FREQ,
            fourier_learnable=PHASE_FOURIER_LEARNABLE,
            phase_hidden=PHASE_HIDDEN,
            N_banks=PHASE_BANKS,
            dropout=PHASE_DROPOUT
        )
    else:
        raise ValueError(f"Unknown phase_model_type '{MODEL_TYPE}'")

