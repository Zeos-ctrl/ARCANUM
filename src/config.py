import os
import yaml
import torch

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
with open(_CONFIG_PATH, "r") as f:
    _cfg = yaml.safe_load(f)

PYCBC = _cfg["pycbc"]
SAMPLING_RANGES = _cfg["sampling_ranges"]
TRAINING = _cfg["training"]
CHECKPOINT_DIR = _cfg["paths"]["checkpoint_dir"]

if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

WAVEFORM = PYCBC["waveform_name"] # approximant
DELTA_T       = float(PYCBC["delta_t"]) # time spacing for get_td_waveform
F_LOWER       = float(PYCBC["f_lower"]) # starting GW frequency [Hz]
DETECTOR = PYCBC["detector_name"] # detector generated for
WAVEFORM_LENGTH     = int(PYCBC["waveform_length"]) # number of time samples per waveform

MASS_MIN      = float(SAMPLING_RANGES["mass_min"]) # black‐hole masses
MASS_MAX      = float(SAMPLING_RANGES["mass_max"])
SPIN_MIN  = float(SAMPLING_RANGES["spin_mag_min"]) # aligned‐spin chi_z range
SPIN_MAX  = float(SAMPLING_RANGES["spin_mag_max"])
INCLINATION_MIN      = float(SAMPLING_RANGES["incl_min"]) # inclination [rad]
INCLINATION_MAX      = float(SAMPLING_RANGES["incl_max"])
ECC_MIN       = float(SAMPLING_RANGES["ecc_min"]) # eccentricity range
ECC_MAX       = float(SAMPLING_RANGES["ecc_max"])

NUM_SAMPLES   = int(TRAINING["num_samples"]) # number of (m1,m2,chi1z,chi2z,incl,ecc) examples
BATCH_SIZE    = int(TRAINING["batch_size"])
NUM_EPOCHS    = int(TRAINING["num_epochs"])
LEARNING_RATE = float(TRAINING["learning_rate"])
PATIENCE      = int(TRAINING["patience"])
FINE_TUNE_EPOCHS = int(TRAINING["fine_tune_epochs"])
FINE_TUNE_LR     = float(TRAINING["fine_tune_lr"])
