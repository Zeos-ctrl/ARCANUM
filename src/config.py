import os
import yaml
import torch
from types import SimpleNamespace

# Load YAML
_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
with open(_CONFIG_PATH, "r") as f:
    _cfg = yaml.safe_load(f)

# Pick which approximant profile to use
PROFILE = os.getenv("WF_PROFILE", "seobnrv4")
PYCBC = _cfg["pycbc"][PROFILE]
SAMPLING_RANGES = SimpleNamespace(**_cfg["sampling_ranges"])
TRAINING = SimpleNamespace(**_cfg["training"])
CHECKPOINT_DIR = _cfg["paths"]["checkpoint_dir"]

# Device
if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# PyCBC parameters
WAVEFORM         = PYCBC["waveform_name"]
DELTA_T          = 1.0 / float(PYCBC["delta_t"])
F_LOWER          = float(PYCBC["f_lower"])
DETECTOR         = PYCBC["detector_name"]
WAVEFORM_LENGTH  = int(PYCBC["waveform_length"])

# Sampling ranges
MASS_MIN         = SAMPLING_RANGES.mass_min
MASS_MAX         = SAMPLING_RANGES.mass_max
SPIN_MIN         = SAMPLING_RANGES.spin_mag_min
SPIN_MAX         = SAMPLING_RANGES.spin_mag_max
INCLINATION_MIN  = SAMPLING_RANGES.incl_min
INCLINATION_MAX  = SAMPLING_RANGES.incl_max
ECC_MIN          = SAMPLING_RANGES.ecc_min
ECC_MAX          = SAMPLING_RANGES.ecc_max

# Training / optimizer
NUM_SAMPLES      = TRAINING.num_samples
VAL_SPLIT        = TRAINING.val_split
RANDOM_SEED      = TRAINING.random_seed

OPTIMIZER_NAME   = TRAINING.optimizer
LEARNING_RATE    = TRAINING.learning_rate
WEIGHT_DECAY     = TRAINING.weight_decay

BATCH_SIZE       = TRAINING.batch_size
NUM_EPOCHS       = TRAINING.num_epochs
PATIENCE         = TRAINING.patience
MIN_DELTA        = float(TRAINING.min_delta)

# Scheduler sub‑block
SCHEDULER_CFG    = SimpleNamespace(**TRAINING.scheduler)

# Fine‑tuning sub‑block
FINE_TUNE_CFG    = SimpleNamespace(**TRAINING.fine_tune)

# HPO sub‑block
HPO_CFG          = SimpleNamespace(**TRAINING.hpo)

# Misc
GRADIENT_CLIP    = TRAINING.gradient_clip
SEED_EVERYTHING  = TRAINING.seed_everything

NOTIFICATIONS = _cfg.get("notifications", {})
WEBHOOK_URL = NOTIFICATIONS.get("webhook_url", None)

# Paths
CHECKPOINT_DIR   = _cfg["paths"]["checkpoint_dir"]
