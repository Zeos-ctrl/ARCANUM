import pytest
import numpy as np
from src.data.dataset import generate_data
from src.data.config import WAVEFORM_LENGTH, NUM_SAMPLES, TRAIN_FEATURES

def test_generate_data_shapes():
    dataset = generate_data()
    S, L = NUM_SAMPLES, WAVEFORM_LENGTH
    assert dataset.inputs.shape == (S * L, len(TRAIN_FEATURES) + 1), "Incorrect input shape."
    assert dataset.targets_A.shape == (S * L, 1), "Incorrect amplitude target shape."
    assert dataset.targets_phi.shape == (S * L, 1), "Incorrect phase target shape."

def test_generate_data_normalization_range():
    dataset = generate_data()
    assert np.all(dataset.targets_A >= 0) and np.all(dataset.targets_A <= 1), "Amplitude not in [0, 1]."
