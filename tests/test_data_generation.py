import pytest
import numpy as np
from src.dataset import generate_data
from src.config import WAVEFORM_LENGTH, NUM_SAMPLES

def test_generate_data_shapes():
    dataset = generate_data(clean=True)
    S, L = NUM_SAMPLES, WAVEFORM_LENGTH
    assert dataset.inputs.shape == (S * L, 7), "Incorrect input shape."
    assert dataset.targets_A.shape == (S * L, 1), "Incorrect amplitude target shape."
    assert dataset.targets_phi.shape == (S * L, 1), "Incorrect phase target shape."

def test_generate_data_normalization_range():
    dataset = generate_data(clean=True)
    assert np.all(dataset.targets_A >= 0) and np.all(dataset.targets_A <= 1), "Amplitude not in [0, 1]."
