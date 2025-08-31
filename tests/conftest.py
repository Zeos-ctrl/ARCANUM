from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def fixed_seed():
    np.random.seed(0)
