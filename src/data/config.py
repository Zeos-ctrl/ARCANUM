from __future__ import annotations

import os
from enum import Enum
from types import SimpleNamespace

import torch
import yaml


class ModelType(str, Enum):
    MLP = 'mlp'


MODEL_TYPE: ModelType = ModelType.MLP

# Load YAML
_CONFIG_PATH = os.path.join(os.path.dirname(__file__), '../..', 'config.yaml')
with open(_CONFIG_PATH) as f:
    _cfg = yaml.safe_load(f)

# Pick which approximant profile to use
# PROFILE = os.getenv("WF_PROFILE", "seobnrv4")
PROFILE = os.getenv('WF_PROFILE', 'phenomd')
PYCBC = _cfg['pycbc'][PROFILE]
SAMPLING_RANGES = SimpleNamespace(**_cfg['sampling_ranges'])
TRAINING = SimpleNamespace(**_cfg['training'])
MODEL = SimpleNamespace(**_cfg['model'])
CHECKPOINT_DIR = _cfg['paths']['checkpoint_dir']

# Device
if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

# PyCBC parameters
WAVEFORM = PYCBC['waveform_name']
DELTA_T = 1.0 / float(PYCBC['delta_t'])
F_LOWER = float(PYCBC['f_lower'])
DETECTOR = PYCBC['detector_name']
WAVEFORM_LENGTH = int(PYCBC['waveform_length'])

# Sampling ranges
MASS_MIN = SAMPLING_RANGES.mass_min
MASS_MAX = SAMPLING_RANGES.mass_max
SPIN_MIN = SAMPLING_RANGES.spin_mag_min
SPIN_MAX = SAMPLING_RANGES.spin_mag_max
INCLINATION_MIN = SAMPLING_RANGES.incl_min
INCLINATION_MAX = SAMPLING_RANGES.incl_max
ECC_MIN = SAMPLING_RANGES.ecc_min
ECC_MAX = SAMPLING_RANGES.ecc_max
CLEAN = SAMPLING_RANGES.clean
SNR_MIN = SAMPLING_RANGES.snr_min
SNR_MAX = SAMPLING_RANGES.snr_max

# Training / optimizer
NUM_SAMPLES = TRAINING.num_samples
VAL_SPLIT = TRAINING.val_split
RANDOM_SEED = TRAINING.random_seed

OPTIMIZER_NAME = TRAINING.optimizer
LEARNING_RATE = TRAINING.learning_rate
WEIGHT_DECAY = TRAINING.weight_decay

BATCH_SIZE = TRAINING.batch_size
NUM_EPOCHS = TRAINING.num_epochs
PATIENCE = TRAINING.patience
MIN_DELTA = float(TRAINING.min_delta)

# Model params
AMP_EMB_HIDDEN = MODEL.amp_emb_hidden
AMP_HIDDEN = MODEL.amp_hidden
AMP_BANKS = MODEL.amp_banks
AMP_DROPOUT = MODEL.amp_dropout
AMP_LR = MODEL.amp_lr
AMP_CLIP = MODEL.amp_clip

PHASE_EMB_HIDDEN = MODEL.phase_emb_hidden
PHASE_HIDDEN = MODEL.phase_hidden
PHASE_BANKS = MODEL.phase_banks
PHASE_DROPOUT = MODEL.phase_dropout
PHASE_LR = MODEL.phase_lr
PHASE_CLIP = MODEL.phase_clip

# Fourier‐feature embedding for both amp & phase models
AMP_FOURIER_BANDS = 16      # number of sin/cos pairs per θ‐dimension
AMP_FOURIER_MAX_FREQ = 10.0    # highest frequency in log‐spaced band
AMP_FOURIER_LEARNABLE = False   # whether to tune the band frequencies

PHASE_FOURIER_BANDS = 16
PHASE_FOURIER_MAX_FREQ = 10.0
PHASE_FOURIER_LEARNABLE = False

# Scheduler sub‑block
SCHEDULER_CFG = SimpleNamespace(**TRAINING.scheduler)

# HPO sub‑block
HPO_CFG = SimpleNamespace(**TRAINING.hpo)

# Misc
GRADIENT_CLIP = TRAINING.gradient_clip
SEED_EVERYTHING = TRAINING.seed_everything

NOTIFICATIONS = _cfg.get('notifications', {})
DISCORD_WEBHOOK_URL = NOTIFICATIONS.get('webhook_url', None)
DISCORD_BOT_TOKEN = NOTIFICATIONS.get('discord_bot_token')
STATUS_CHANNEL_ID = int(NOTIFICATIONS.get('status_channel_id', 0))
GUILD_ID = int(NOTIFICATIONS.get('guild_id', 0))

# Paths
CHECKPOINT_DIR = _cfg['paths']['checkpoint_dir']

# Inputs to train the network, comment out to disable
TRAIN_FEATURES = [
    'chirp_mass',
    'symmetric_mass_ratio',
    #    "effective_spin",
    #    "eccentricity",
]
