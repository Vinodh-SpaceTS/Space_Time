# receivers/simulated_receiver.py
import numpy as np
from receivers.base_receiver import BaseReceiver

class SimulatedReceiver(BaseReceiver):
    def __init__(
        self,
        receiver_bias: float,
        sat_clock_std: float,
        prop_std: float,
        meas_std: float,
        use_gauss_markov: bool = False,
        correlation_time: float = 3600.0,
        total_counts = None,
        gps_counts = None,
        galileo_counts = None,
        glonass_counts = None,
        sbas_counts = None,
        dynamic_r_enabled: bool = True,
        sat_scale: float = 1.0,
        seed: int = None
    ):
        self.receiver_bias = receiver_bias
        self.sat_clock_std = sat_clock_std
        self.prop_std = prop_std
        self.meas_std = meas_std
        self.use_gauss_markov = use_gauss_markov
        self.correlation_time = correlation_time
        
        # Fallback counts if none provided (e.g. static simulations)
        self.total_counts = total_counts if total_counts is not None else [32]
        self.gps_counts = gps_counts if gps_counts is not None else [12]
        self.galileo_counts = galileo_counts if galileo_counts is not None else [10]
        self.glonass_counts = glonass_counts if glonass_counts is not None else [8]
        self.sbas_counts = sbas_counts if sbas_counts is not None else [2]
        
        self.dynamic_r_enabled = dynamic_r_enabled
        self.sat_scale = sat_scale
        self.rng = np.random.default_rng(seed)
        
        self.prev_prop_err = None

    def reset(self, seed=None):
        self.prev_prop_err = None
        if seed is not None:
            self.rng = np.random.default_rng(seed)


    def measure(self, t: float, dt: float = 1.0) -> dict:
        k = int(round(t / dt))
        
        # 1. Fetch satellite counts for this epoch
        n_epochs = len(self.total_counts)
        raw_sats = self.total_counts[k % n_epochs]
        gps_sats = max(0, int(self.gps_counts[k % len(self.gps_counts)] * self.sat_scale))
        gal_sats = max(0, int(self.galileo_counts[k % len(self.galileo_counts)] * self.sat_scale))
        glo_sats = max(0, int(self.glonass_counts[k % len(self.glonass_counts)] * self.sat_scale))
        sbas_sats = max(0, int(self.sbas_counts[k % len(self.sbas_counts)] * self.sat_scale))
        sats = max(0, int(raw_sats * self.sat_scale))
        
        # 2. Determine covariance R
        if sats < 4:
            # Out of lock / Holdover
            R_k = 1e12  # virtually infinite variance
        else:
            if self.dynamic_r_enabled:
                # Continuous constellation-driven covariance scaling
                n_avg = np.mean(self.total_counts)
                raw_mult = np.sqrt(n_avg / max(1.0, sats))
                
                # Normalize multiplier based on expected average to maintain config scaling
                unscaled_raw_mults = np.sqrt(n_avg / np.maximum(1.0, np.array(self.total_counts)))
                mean_mult = np.mean(unscaled_raw_mults)
                multiplier = raw_mult / (mean_mult if mean_mult > 0 else 1.0)
                R_k = (self.meas_std * multiplier) ** 2
            else:
                R_k = self.meas_std ** 2

        # 3. Simulate errors
        # Satellite clock error (independent white noise)
        sat_clock_k = self.rng.normal(0, self.sat_clock_std)
        
        # Propagation error
        if self.use_gauss_markov:
            beta = 1.0 / self.correlation_time
            driving_noise_std = self.prop_std * np.sqrt(2.0 * beta * dt)
            prop_step = self.rng.normal(0, driving_noise_std)
            
            if self.prev_prop_err is None or k == 0:
                self.prev_prop_err = self.rng.normal(0, self.prop_std)
            else:
                self.prev_prop_err = (1.0 - beta * dt) * self.prev_prop_err + prop_step
            prop_error_k = self.prev_prop_err
        else:
            prop_error_k = self.rng.normal(0, self.prop_std)
            
        # Measurement noise
        meas_noise_std = np.sqrt(R_k) if R_k < 1e10 else self.meas_std * 100.0
        meas_noise_k = self.rng.normal(0, meas_noise_std)
        
        # Total GNSS time error
        gnss_error_k = self.receiver_bias + sat_clock_k + prop_error_k + meas_noise_k
        
        return {
            "gnss_error": gnss_error_k,
            "sat_count": sats,
            "gps_count": gps_sats,
            "galileo_count": gal_sats,
            "glonass_count": glo_sats,
            "sbas_count": sbas_sats,
            "sat_clock_error": sat_clock_k,
            "propagation_error": prop_error_k,
            "measurement_noise": meas_noise_k,
            "R": R_k
        }
