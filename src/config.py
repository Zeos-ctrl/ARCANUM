# src/config.py
import os
import yaml
import torch

# locate the YAML file (assuming you're running from the repo root)
_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
with open(_CONFIG_PATH, "r") as f:
    _cfg = yaml.safe_load(f)

# Expose top‐level sections as attributes:
PYCBC = _cfg["pycbc"]
TIME_WINDOW = _cfg["time_window"]
SAMPLING_RANGES = _cfg["sampling_ranges"]
TRAINING = _cfg["training"]
CHECKPOINT_DIR = _cfg["paths"]["checkpoint_dir"]

if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# Convenience shortcuts:
WAVEFORM_NAME = PYCBC["waveform_name"]
DELTA_T       = float(PYCBC["delta_t"])
F_LOWER       = float(PYCBC["f_lower"])
DETECTOR_NAME = PYCBC["detector_name"]
PSI_FIXED     = float(PYCBC["psi_fixed"])

T_BEFORE      = float(TIME_WINDOW["t_before"])
T_AFTER       = float(TIME_WINDOW["t_after"])

MASS_MIN      = float(SAMPLING_RANGES["mass_min"])
MASS_MAX      = float(SAMPLING_RANGES["mass_max"])
SPIN_MAG_MIN  = float(SAMPLING_RANGES["spin_mag_min"])
SPIN_MAG_MAX  = float(SAMPLING_RANGES["spin_mag_max"])
INCL_MIN      = float(SAMPLING_RANGES["incl_min"])
INCL_MAX      = float(SAMPLING_RANGES["incl_max"])
ECC_MIN       = float(SAMPLING_RANGES["ecc_min"])
ECC_MAX       = float(SAMPLING_RANGES["ecc_max"])
RA_MIN        = float(SAMPLING_RANGES["ra_min"])
RA_MAX        = float(SAMPLING_RANGES["ra_max"])
DEC_MIN       = float(SAMPLING_RANGES["dec_min"])
DEC_MAX       = float(SAMPLING_RANGES["dec_max"])
DIST_MIN      = float(SAMPLING_RANGES["dist_min"])
DIST_MAX      = float(SAMPLING_RANGES["dist_max"])
COAL_MIN      = float(SAMPLING_RANGES["coal_min"])
COAL_MAX      = float(SAMPLING_RANGES["coal_max"])

NUM_SAMPLES   = int(TRAINING["num_samples"])
BATCH_SIZE    = int(TRAINING["batch_size"])
NUM_EPOCHS    = int(TRAINING["num_epochs"])
LEARNING_RATE = float(TRAINING["learning_rate"])
PATIENCE      = int(TRAINING["patience"])
FINE_TUNE_EPOCHS = int(TRAINING["fine_tune_epochs"])
FINE_TUNE_LR     = float(TRAINING["fine_tune_lr"])
