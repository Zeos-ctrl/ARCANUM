import pytest
import numpy as np
from pycbc.psd import aLIGOZeroDetHighPower
from src.config import WAVEFORM_LENGTH, DELTA_T
from src.dataset import make_waveform, make_noisy_waveform, sample_parameters

def test_make_waveform_length():
    theta = sample_parameters(1)[0]
    h = make_waveform(theta)
    assert h.shape == (WAVEFORM_LENGTH,), "Waveform length mismatch."

def test_make_noisy_waveform_reproducibility():
    theta = sample_parameters(1)[0]
    psd = aLIGOZeroDetHighPower(WAVEFORM_LENGTH, 1.0 / (WAVEFORM_LENGTH * DELTA_T), 20)
    h1 = make_noisy_waveform(theta, psd, seed=42)
    h2 = make_noisy_waveform(theta, psd, seed=42)
    assert np.allclose(h1, h2), "Noise generation is not reproducible with same seed."
