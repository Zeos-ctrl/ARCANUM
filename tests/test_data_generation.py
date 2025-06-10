import pytest
import numpy as np

from src.data_generation import (
    sample_parameters,
    build_common_times,
    build_waveform_chunks,
    T_BEFORE,
    T_AFTER,
    DELTA_T,
)

def test_sample_parameters_shapes_and_types():
    num_samples = 5
    param_list, thetas = sample_parameters(num_samples)
    # Check lengths and shapes
    assert isinstance(param_list, list)
    assert len(param_list) == num_samples
    assert isinstance(thetas, np.ndarray)
    assert thetas.shape == (num_samples, 15)
    assert thetas.dtype == np.float32
    # Check that each element of param_list is a tuple of length 15
    for params in param_list:
        assert isinstance(params, tuple)
        assert len(params) == 15
        for value in params:
            assert isinstance(value, float)

def test_build_common_times_values():
    delta_t = 1.0
    t_before = 1.0
    t_after = 2.0
    common_times, n_common = build_common_times(delta_t=delta_t, t_before=t_before, t_after=t_after)
    # Expect array from -1.0 to +2.0 inclusive, step 1.0 → [-1, 0, 1, 2] → length 4
    expected = np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    assert isinstance(common_times, np.ndarray)
    assert n_common == expected.size
    assert np.allclose(common_times, expected)
    assert common_times.dtype == np.float32

@pytest.mark.parametrize("outside_t0", [100.0, -100.0])
def test_build_waveform_chunks_outside_time_window(outside_t0):
    # Build a single-parameter tuple very far outside the merge window
    params = (
        10.0, 10.0,       # m1, m2
        0.0, 0.0, 0.0,    # S1x, S1y, S1z
        0.0, 0.0, 0.0,    # S2x, S2y, S2z
        np.pi/3, 0.0,     # incl, ecc
        0.0, 0.0,         # ra, dec
        100.0, outside_t0, 0.0  # distance, t0 far away, phi0
    )
    num_samples = 1
    # Create param_list and thetas manually
    param_list = [params]
    # Build the common_times array where no waveform overlaps
    common_times, n_common = build_common_times(delta_t=DELTA_T, t_before=T_BEFORE, t_after=T_AFTER)
    waveform_chunks = build_waveform_chunks(param_list, common_times, n_common)
    assert isinstance(waveform_chunks, list)
    assert len(waveform_chunks) == num_samples
    chunk = waveform_chunks[0]
    # Because t0 is far outside, expect empty arrays
    assert isinstance(chunk, dict)
    assert chunk["start_idx"] == 0
    assert isinstance(chunk["amp_chunk"], np.ndarray)
    assert isinstance(chunk["dphi_chunk"], np.ndarray)

def test_build_waveform_chunks_basic_run():
    # Use randomly sampled parameters, but limit to 2 samples to keep runtime small
    num_samples = 2
    param_list, thetas = sample_parameters(num_samples)
    # Build the common grid
    common_times, n_common = build_common_times(delta_t=DELTA_T, t_before=T_BEFORE, t_after=T_AFTER)
    # Generate waveform chunks
    waveform_chunks = build_waveform_chunks(param_list, common_times, n_common)
    assert isinstance(waveform_chunks, list)
    assert len(waveform_chunks) == num_samples
    for chunk in waveform_chunks:
        # Each chunk should have the correct keys
        assert set(chunk.keys()) == {"start_idx", "amp_chunk", "dphi_chunk"}
        # start_idx should be int, amp_chunk and dphi_chunk are numpy arrays
        assert isinstance(chunk["start_idx"], int)
        assert isinstance(chunk["amp_chunk"], np.ndarray)
        assert isinstance(chunk["dphi_chunk"], np.ndarray)
        # amp_chunk and dphi_chunk should have dtype float32
        assert chunk["amp_chunk"].dtype == np.float32
        assert chunk["dphi_chunk"].dtype == np.float32
        # Values in amp_chunk should be between 0 and 1 (normalized amplitude)
        if chunk["amp_chunk"].size > 0:
            assert np.all(chunk["amp_chunk"] >= 0.0)
            assert np.all(chunk["amp_chunk"] <= 1.0)
