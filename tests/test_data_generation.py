from __future__ import annotations

import numpy as np
import pytest

from src.data.config import NUM_SAMPLES
from src.data.config import TRAIN_FEATURES
from src.data.config import WAVEFORM_LENGTH
from src.data.dataset import generate_data


def test_generate_data_shapes():
    dataset = generate_data(samples=10)
    S, L = 10, WAVEFORM_LENGTH
    assert dataset.inputs.shape == (
        S * L, len(TRAIN_FEATURES) + 1), 'Incorrect input shape.'
    assert dataset.targets_A.shape == (
        S * L, 1), 'Incorrect amplitude target shape.'
    assert dataset.targets_phi.shape == (
        S * L, 1), 'Incorrect phase target shape.'


def test_generate_data_normalization_range():
    dataset = generate_data()
    assert np.all(dataset.targets_A >= 0) and np.all(
        dataset.targets_A <= 1), 'Amplitude not in [0, 1].'
