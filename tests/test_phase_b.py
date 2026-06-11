# tests/test_phase_b.py
import numpy as np
import pytest
from models.kalman_filter import run_kalman_filter_v2, calculate_recovery_time

def test_kalman_gains_outage():
    t = np.arange(0, 100, 1.0)
    N = len(t)
    
    # Rubidium clock error
    rb_err = np.zeros(N)
    
    # GNSS error
    gnss_err = np.zeros(N)
    
    # Run the filter with an outage from t=40 to t=70
    res = run_kalman_filter_v2(
        true_time=t,
        rubidium_error=rb_err,
        gnss_error=gnss_err,
        outage_enabled=True,
        outage_start=40.0,
        outage_end=70.0,
        Q_bias=1e-20,
        Q_drift=1e-22,
        R_val=(50e-9)**2
    )
    
    # Verify gains collection
    assert hasattr(res, "kalman_gains")
    assert len(res.kalman_gains) == N
    
    # During normal tracking (before outage), gain should be non-zero (since Q is positive)
    assert np.all(res.kalman_gains[:40] > 0.0)
    
    # During outage, gain must be exactly 0.0
    assert np.all(res.kalman_gains[40:70] == 0.0)
    
    # After outage, gain should resume being non-zero
    assert np.all(res.kalman_gains[70:] > 0.0)

def test_holdover_metrics_extraction():
    # Construct a deterministic mock time series and master error profile to test recovery time
    t = np.arange(0, 200, 1.0)
    outage_end = 100.0
    
    # Pre-outage error: 0
    # Outage error: grows linearly up to 150 ns at t=100
    # Post-outage error: decays exponentially back to 0.
    # At t=100 (outage end), master error = 150e-9 s (150 ns)
    # Decay rate: e^(-0.1 * (t - 100))
    # We want to find when it falls and remains below 50 ns (threshold_ns = 50)
    # master_error(t) = 150e-9 * exp(-0.1 * (t - 100))
    # 150 * exp(-0.1 * dt) < 50 => exp(-0.1 * dt) < 1/3 => -0.1 * dt < ln(1/3) => dt > ln(3)/0.1 approx 10.986 s
    # Since dt is integer steps (1s resolution), at dt = 11s:
    # 150 * exp(-1.1) = 150 * 0.3328 = 49.93 ns < 50 ns
    # Let's verify with the helper function
    
    master_error = np.zeros_like(t, dtype=float)
    for i, time_val in enumerate(t):
        if time_val < outage_end:
            master_error[i] = (time_val / outage_end) * 150e-9
        else:
            master_error[i] = 150e-9 * np.exp(-0.1 * (time_val - outage_end))
            
    # Compute recovery time with window_seconds=5.0
    rec_time = calculate_recovery_time(t, master_error, outage_end, threshold_ns=50.0, dt=1.0, window_seconds=5.0)

    
    # Expectation: at dt=11.0 s, error is ~49.93 ns, and it decays further.
    # So all subsequent points are smaller than 50 ns.
    assert rec_time is not None
    assert abs(rec_time - 11.0) < 1e-5
    
    # Test average drift rate calculation
    # Let's verify the formula for mean frequency drift rate during holdover
    # average frequency offset = (error_end - error_start) / duration
    outage_start = 50.0
    outage_duration = outage_end - outage_start
    phase_start = master_error[t == outage_start][0]
    phase_end = master_error[t == outage_end][0]
    mean_drift_s_s = (phase_end - phase_start) / outage_duration
    mean_drift_ps_s = mean_drift_s_s * 1e12
    
    # phase_start = 75e-9, phase_end = 150e-9
    # drift = (150 - 75)e-9 / 50 = 75e-9 / 50 = 1.5e-9 s/s = 1500 ps/s
    assert abs(mean_drift_ps_s - 1500.0) < 1e-5
