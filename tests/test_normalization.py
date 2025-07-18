import pytest
import numpy as np
from src.data.dataset import generate_data

def test_parameter_normalization():
    dataset = generate_data(clean=True)
    means = dataset.theta_norm.mean(axis=0)
    stds = dataset.theta_norm.std(axis=0)
    # Tolerate some small numerical error
    assert np.all(np.abs(means) < 1e-2), "Normalized parameter means not near zero."
    assert np.all(np.abs(stds - 1) < 1e-2), "Normalized parameter stds not near one."
