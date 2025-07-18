import pytest
import numpy as np
from src.data.dataset import sample_parameters
from src.data.config import MASS_MIN, MASS_MAX, NUM_SAMPLES

def test_sample_parameters_shape():
    n = 100
    params = sample_parameters(n)
    assert params.shape == (n, 6), "Incorrect parameter shape."

def test_sample_parameters_bounds():
    n = 1000
    params = sample_parameters(n)
    assert np.all(params[:, 0] >= MASS_MIN) and np.all(params[:, 1] <= MASS_MAX), "Masses out of bounds."
