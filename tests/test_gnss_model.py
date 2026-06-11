# tests/test_gnss_model.py
import numpy as np
import pytest
from receivers.simulated_receiver import SimulatedReceiver
from models.gnss_time_model import simulate_gnss_time

def test_simulated_receiver_measurements():
    rx = SimulatedReceiver(
        receiver_bias=100e-9,
        sat_clock_std=20e-9,
        prop_std=30e-9,
        meas_std=50e-9,
        use_gauss_markov=False,
        total_counts=[10, 15, 20],
        dynamic_r_enabled=True,
        seed=42
    )
    
    # Check first measurement
    res = rx.measure(t=0.0, dt=1.0)
    assert "gnss_error" in res
    assert "sat_count" in res
    assert "R" in res
    assert "propagation_error" in res
    assert "measurement_noise" in res
    
    # Sat count should match first index
    assert res["sat_count"] == 10
    
    # Check that a low satellite count (e.g. < 4) triggers out of lock (infinite covariance R)
    rx_outage = SimulatedReceiver(
        receiver_bias=100e-9,
        sat_clock_std=20e-9,
        prop_std=30e-9,
        meas_std=50e-9,
        total_counts=[2],  # Outage
        dynamic_r_enabled=True,
        seed=42
    )
    res_outage = rx_outage.measure(t=0.0, dt=1.0)
    assert res_outage["sat_count"] == 2
    assert res_outage["R"] >= 1e12

def test_gauss_markov_propagation():
    # Verify Gauss-Markov state changes are correlated, unlike white noise
    rx_gm = SimulatedReceiver(
        receiver_bias=0.0,
        sat_clock_std=0.0,
        prop_std=30e-9,
        meas_std=0.0,
        use_gauss_markov=True,
        correlation_time=3600.0,
        seed=42
    )
    
    errs = []
    for t in range(10):
        res = rx_gm.measure(float(t), 1.0)
        errs.append(res["propagation_error"])
        
    # Standard deviation of consecutive steps should be small due to high correlation
    diffs = np.diff(errs)
    # The step size driving noise std is: 30e-9 * sqrt(2 * (1/3600) * 1) ≈ 30e-9 * 0.023 ≈ 0.7e-9
    assert np.std(diffs) < 5e-9

def test_simulate_gnss_time_backward_compatibility():
    gnss_error, sat_clock, prop, meas, R_profile = simulate_gnss_time(
        duration=100,
        dt=1.0,
        receiver_bias=100e-9,
        sat_clock_std=20e-9,
        prop_std=30e-9,
        meas_std=50e-9,
        use_gauss_markov=False,
        sat_counts=[12],
        dynamic_r_enabled=True,
        seed=42
    )
    
    assert len(gnss_error) == 100
    assert len(R_profile) == 100
    assert np.all(R_profile > 0)
