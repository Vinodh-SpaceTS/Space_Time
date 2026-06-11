# tests/test_clock_model.py
import numpy as np
import pytest
from models.clock_model import simulate_rubidium_clock

def test_rubidium_simulation_lengths_and_bias():
    duration = 100
    dt = 1.0
    bias = 1e-3
    rw_step = 1e-14
    white_noise = 1e-12
    aging = 1e-13
    
    t, error, phase_rw, freq_offset, aging_err = simulate_rubidium_clock(
        duration=duration,
        dt=dt,
        bias=bias,
        random_walk_step=rw_step,
        white_noise_std=white_noise,
        aging_rate=aging,
        seed=42
    )
    
    # Verify lengths
    assert len(t) == 100
    assert len(error) == 100
    assert len(phase_rw) == 100
    assert len(freq_offset) == 100
    assert len(aging_err) == 100
    
    # Verify start time and step
    assert t[0] == 0.0
    assert t[1] == 1.0
    assert t[-1] == 99.0
    
    # Verify initial bias (without white noise, should be close)
    assert abs(error[0] - bias) < 5e-12

def test_rubidium_aging():
    duration = 100
    dt = 1.0
    bias = 0.0
    rw_step = 0.0
    white_noise = 0.0
    aging = 1e-10
    
    t, error, phase_rw, freq_offset, aging_err = simulate_rubidium_clock(
        duration=duration,
        dt=dt,
        bias=bias,
        random_walk_step=rw_step,
        white_noise_std=white_noise,
        aging_rate=aging,
        seed=42
    )
    
    # Verify quadratic aging phase error: 0.5 * aging * t^2
    expected_aging = 0.5 * aging * (t ** 2)
    assert np.allclose(aging_err, expected_aging)
    assert np.allclose(error, expected_aging)

def test_rubidium_random_walk():
    duration = 100
    dt = 1.0
    bias = 0.0
    rw_step = 1e-12
    white_noise = 0.0
    aging = 0.0
    
    t, error, phase_rw, freq_offset, aging_err = simulate_rubidium_clock(
        duration=duration,
        dt=dt,
        bias=bias,
        random_walk_step=rw_step,
        white_noise_std=white_noise,
        aging_rate=aging,
        seed=42
    )
    
    # Frequency offset should be the cumsum of random walk steps
    # Phase random walk should be the cumsum of freq_offset * dt
    # Since bias, aging, and white noise are 0, error should equal phase_rw
    assert np.allclose(error, phase_rw)
    assert freq_offset[0] == 0.0
