import pytest
import numpy as np

@pytest.fixture(autouse=True)
def fixed_seed():
    np.random.seed(0)
