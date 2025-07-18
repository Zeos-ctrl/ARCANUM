import torch.nn as nn
from src.data.config import MODEL_TYPE, AMP_EMB_HIDDEN, AMP_HIDDEN, AMP_BANKS, AMP_DROPOUT, PHASE_EMB_HIDDEN, PHASE_HIDDEN, PHASE_BANKS, PHASE_DROPOUT
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
            emb_hidden=AMP_EMB_HIDDEN,
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
            emb_hidden=PHASE_EMB_HIDDEN,
            phase_hidden=PHASE_HIDDEN,
            N_banks=PHASE_BANKS,
            dropout=PHASE_DROPOUT
        )
    else:
        raise ValueError(f"Unknown phase_model_type '{MODEL_TYPE}'")

