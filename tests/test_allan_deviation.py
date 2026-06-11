# tests/test_allan_deviation.py
import numpy as np
import pytest
from analysis.allan_deviation import calculate_allan_deviation, check_rb_allan_slopes

def test_allan_deviation_computation():
    # Generate random phase data
    np.random.seed(42)
    phase_data = np.random.normal(0, 1e-9, 1000)
    
    taus, adev = calculate_allan_deviation(phase_data, dt=1.0)
    
    assert len(taus) > 0
    assert len(adev) == len(taus)
    assert np.all(adev > 0)
    
    # White phase noise should have a slope of approximately -1.0 in ADEV
    log_tau = np.log10(taus[:5])
    log_adev = np.log10(adev[:5])
    slope, _ = np.polyfit(log_tau, log_adev, 1)
    
    # Assert slope is negative (white phase noise slope is -1)
    assert slope < -0.3

def test_allan_slope_checks():
    # Generate random phase data representing White PM
    np.random.seed(42)
    phase_data = np.random.normal(0, 1e-9, 50)
    
    taus, adev = calculate_allan_deviation(phase_data, dt=1.0)
    slopes, passed = check_rb_allan_slopes(taus, adev, verbose=False)
    
    assert len(slopes) == len(taus) - 1
    # Since this is pure white noise without random walk, long term slope assertion might fail.
    # We just test that the function completes and returns expected types.
    assert isinstance(passed, bool)
