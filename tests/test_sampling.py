from __future__ import annotations

import numpy as np
import pytest

from src.data.config import MASS_MAX
from src.data.config import MASS_MIN
from src.data.config import NUM_SAMPLES
from src.data.dataset import sample_parameters


def test_sample_parameters_shape():
    n = 100
    params = sample_parameters(n)
    assert params.shape == (n, 6), 'Incorrect parameter shape.'


def test_sample_parameters_bounds():
    n = 1000
    params = sample_parameters(n)
    assert np.all(params[:, 0] >= MASS_MIN) and np.all(
        params[:, 1] <= MASS_MAX), 'Masses out of bounds.'
