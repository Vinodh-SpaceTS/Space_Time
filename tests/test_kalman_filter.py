# tests/test_kalman_filter.py
import numpy as np
import pytest
from models.kalman_filter import run_kalman_filter_v2, build_Q_matrix
from receivers.simulated_receiver import SimulatedReceiver

def test_kalman_disciplining_performance():
    # Simulates a scenario where Kalman filter disciplines a drifting clock
    t = np.arange(0, 500, 1.0)
    N = len(t)
    
    # Rubidium clock: small random walk + bias
    np.random.seed(42)
    rb_err = 1e-6 + 1e-12 * np.cumsum(np.random.normal(0, 1.0, N))
    
    # GNSS: receiver bias + noise (standard deviation = 30 ns)
    gnss_err = 100e-9 + np.random.normal(0, 30e-9, N)
    
    # Run the filter
    res = run_kalman_filter_v2(
        true_time=t,
        rubidium_error=rb_err,
        gnss_error=gnss_err,
        Q_bias=1e-20,
        Q_drift=1e-22,
        R_val=(30e-9)**2
    )
    
    bias_est, drift_est, kal_var, master_err, innov = res
    
    # The disciplined master clock error should track the receiver bias (100 ns),
    # meaning the offset relative to the receiver's time base is near zero.
    assert np.std(master_err) < 25e-9
    assert np.mean(np.abs(master_err - 100e-9)) < 25e-9


def test_receiver_integration():
    t = np.arange(0, 100, 1.0)
    N = len(t)
    rb_err = 1e-6 + np.random.normal(0, 1e-12, N)
    
    rx = SimulatedReceiver(
        receiver_bias=100e-9,
        sat_clock_std=20e-9,
        prop_std=30e-9,
        meas_std=50e-9,
        total_counts=[12],
        dynamic_r_enabled=False,
        seed=42
    )
    
    # Run filter with simulated receiver object
    res = run_kalman_filter_v2(
        true_time=t,
        rubidium_error=rb_err,
        receiver=rx,
        Q_bias=1e-20,
        Q_drift=1e-22
    )
    
    bias_est, drift_est, kal_var, master_err, innov = res
    assert len(bias_est) == 100
    assert len(res.diverged_flags) == 100
    assert not np.any(res.diverged_flags) # Should not diverge under nominal noise

def test_divergence_detection():
    t = np.arange(0, 100, 1.0)
    N = len(t)
    
    # Normal clock error
    rb_err = np.zeros(N)
    
    # Normal GNSS error for first 40 steps, then a massive step offset of 10 microseconds (10000 ns)
    gnss_err = np.random.normal(0, 10e-9, N)
    gnss_err[40:] += 10e-6
    
    # Run the filter with low measurement noise (trust GNSS)
    res = run_kalman_filter_v2(
        true_time=t,
        rubidium_error=rb_err,
        gnss_error=gnss_err,
        Q_bias=1e-24,
        Q_drift=1e-26,
        R_val=(10e-9)**2,
        threshold_divergence=3.0,
        consecutive_limit=5
    )
    
    # Check that divergence gets flagged after the step change
    assert not np.any(res.diverged_flags[:40])
    assert np.any(res.diverged_flags[45:])  # Should diverge after 5 consecutive steps of large innovations
