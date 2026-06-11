# models/kalman_filter.py
# Spacecraft Clock Kalman Filter — Single Production File
#
# Merged from:
#   kalman_filter_v2.py  (FilterPy, plotting functions, Streamlit production)
#   kalman_filter.py     (NumPy baseline — corrected Q matrix, clean docstring)
#
# What was kept from each:
#   FROM kalman_filter_v2.py : FilterPy KalmanFilter, all three plot functions,
#                              outage logic, Streamlit-compatible styling
#   FROM kalman_filter.py    : corrected full Q matrix (Paper R17, Zhang 2012),
#                              cleaner docstring, explicit parameter documentation
#   DELETED                  : the NumPy manual filter loop (replaced by FilterPy)
#
# References:
#   R13 — Hauschild & Montenbruck (2019) — GPS clock Kalman estimation
#   R17 — Zhang et al. (2012) — corrected Q matrix formula (cross-coupling terms)

import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter

try:
    from clock_model import H0, H_NEG1, H_NEG2
except ImportError:
    try:
        from models.clock_model import H0, H_NEG1, H_NEG2
    except ImportError:
        H0     = 2e-19
        H_NEG1 = 7e-23
        H_NEG2 = 2e-30


# ── Q matrix builder (corrected formula — Paper R17) ──────────────────────────

def build_Q_matrix(dt, h0=H0, h_neg1=H_NEG1, h_neg2=H_NEG2):
    """
    Build the 2x2 process noise covariance matrix Q for a [bias, drift] state.

    Uses the corrected formula from Paper R17 (Zhang 2012).
    The diagonal-only Q used in prior versions underestimated the bias-drift
    cross-coupling and produced a filter that was overconfident in its drift
    estimate during GPS outages.

    Parameters
    ----------
    dt     : float — time step in seconds
    h0     : float — white FM Allan coefficient
    h_neg1 : float — flicker FM Allan coefficient
    h_neg2 : float — random walk FM Allan coefficient

    Returns
    -------
    Q : 2x2 ndarray
    """
    q_bias  = h0 * dt + (2.0/3.0)*h_neg1*dt**2 + (1.0/3.0)*h_neg2*dt**3
    q_cross = 0.5*h_neg1*dt**2 + 0.5*h_neg2*dt**3
    q_drift = (2.0/3.0)*h_neg1*dt + h_neg2*dt**2
    return np.array([
        [q_bias,  q_cross],
        [q_cross, q_drift],
    ])


# ── Main filter function ───────────────────────────────────────────────────────

class KalmanOutput:
    """
    Custom wrapper class that behaves like a tuple of length 5 for backward-compatible
    unpacking, while exposing extra diagnostics (like divergence flags, innovation ratios,
    and Kalman gains) as properties.
    """
    def __init__(self, bias_estimates, drift_estimates, kalman_variance, master_error, innovations, diverged_flags, innovation_ratios, kalman_gains):
        self.bias_estimates = bias_estimates
        self.drift_estimates = drift_estimates
        self.kalman_variance = kalman_variance
        self.master_error = master_error
        self.innovations = innovations
        self.diverged_flags = diverged_flags
        self.innovation_ratios = innovation_ratios
        self.kalman_gains = kalman_gains

    def __iter__(self):
        yield self.bias_estimates
        yield self.drift_estimates
        yield self.kalman_variance
        yield self.master_error
        yield self.innovations

    def __getitem__(self, index):
        return [self.bias_estimates, self.drift_estimates, self.kalman_variance, self.master_error, self.innovations][index]

    def __len__(self):
        return 5


def run_kalman_filter_v2(
    true_time,
    rubidium_error,
    gnss_error=None,
    outage_enabled=False,
    outage_start=0,
    outage_end=0,
    Q_bias=None,
    Q_drift=None,
    R_val=(50e-9)**2,
    receiver=None,
    threshold_divergence=5.0,
    consecutive_limit=5
):
    """
    2-state Kalman filter using FilterPy — estimates [clock_bias, clock_drift].

    The measurement at each epoch is z = rubidium_error - gnss_error, which
    represents the total offset between the Rb clock and the GNSS reference.
    The filter tracks this offset and its rate of change (drift), then
    subtracts the bias estimate from the raw Rb error to produce the
    disciplined master clock output.

    Q matrix: built from Allan deviation coefficients (Paper R17 formula).
    If Q_bias / Q_drift are passed, they override the Allan-derived values —
    retained for backwards compatibility with existing call sites.

    Parameters
    ----------
    true_time      : ndarray — simulation epoch array (seconds)
    rubidium_error : ndarray — Rb clock phase error (seconds)
    gnss_error     : ndarray — GNSS measurement error (seconds) [optional if receiver is provided]
    outage_enabled : bool    — suppress GNSS updates during outage window
    outage_start   : float   — outage start time (seconds)
    outage_end     : float   — outage end time (seconds)
    Q_bias         : float   — override bias process noise variance (optional)
    Q_drift        : float   — override drift process noise variance (optional)
    R_val          : float   — measurement noise variance (default (50 ns)^2)
    receiver       : BaseReceiver — optional receiver abstraction layer to obtain measurements
    threshold_divergence : float — innovation ratio threshold for divergence detection (default 5.0)
    consecutive_limit : int — consecutive epochs exceeding threshold to flag divergence (default 5)

    Returns
    -------
    output : KalmanOutput — custom object containing:
             - bias_estimates (ndarray)
             - drift_estimates (ndarray)
             - kalman_variance (ndarray)
             - master_error (ndarray)
             - innovations (ndarray)
             and extra properties:
             - diverged_flags (ndarray of bool)
             - innovation_ratios (ndarray)
    """
    N  = len(true_time)
    dt = true_time[1] - true_time[0] if N > 1 else 1.0

    # ── Build filter ──────────────────────────────────────────────────────────
    kf = KalmanFilter(dim_x=2, dim_z=1)

    # Initial GNSS error baseline
    if receiver is not None:
        init_res = receiver.measure(true_time[0], dt)
        gnss_error_0 = init_res["gnss_error"]
        R_0 = init_res["R"]
    else:
        gnss_error_0 = gnss_error[0] if gnss_error is not None else 0.0
        is_r_seq = hasattr(R_val, "__len__") and not isinstance(R_val, (str, bytes))
        R_0 = R_val[0] if is_r_seq else R_val

    # Initial state: [bias, drift] — bias from first measurement, drift = 0
    kf.x = np.array([
        [rubidium_error[0] - gnss_error_0],
        [0.0]
    ])

    # Initial covariance P
    kf.P = np.array([
        [1e-6,  0.0],
        [0.0,  1e-12],
    ])

    # State transition F: bias += drift * dt
    kf.F = np.array([
        [1.0, dt],
        [0.0, 1.0],
    ])

    # Measurement matrix H: observe bias only
    kf.H = np.array([[1.0, 0.0]])

    # Process noise Q — use corrected Allan-derived matrix by default
    Q_phys = build_Q_matrix(dt)
    if Q_bias is not None and Q_drift is not None:
        # Legacy override: replace diagonal elements only
        Q_phys[0, 0] = Q_bias
        Q_phys[1, 1] = Q_drift
    kf.Q = Q_phys

    # Measurement noise R (can be static scalar or time-varying vector)
    if receiver is not None:
        kf.R = np.array([[R_0]])
    else:
        is_r_seq = hasattr(R_val, "__len__") and not isinstance(R_val, (str, bytes))
        if is_r_seq:
            kf.R = np.array([[R_val[0]]])
        else:
            kf.R = np.array([[R_val]])

    # ── Run filter ────────────────────────────────────────────────────────────
    bias_estimates  = np.zeros(N)
    drift_estimates = np.zeros(N)
    kalman_variance = np.zeros(N)
    innovations     = np.zeros(N)
    diverged_flags  = np.zeros(N, dtype=bool)
    innovation_ratios = np.zeros(N)
    kalman_gains    = np.zeros(N)

    consecutive_divergent_count = 0

    for k in range(N):
        # Determine current measurements and noise R
        if receiver is not None:
            res = receiver.measure(true_time[k], dt)
            gnss_error_k = res["gnss_error"]
            R_k = res["R"]
            kf.R = np.array([[R_k]])
            in_outage = (outage_enabled and (outage_start <= true_time[k] < outage_end)) or (R_k > 1e10)
        else:
            gnss_error_k = gnss_error[k] if gnss_error is not None else 0.0
            if is_r_seq:
                R_k = R_val[k]
                kf.R = np.array([[R_k]])
            else:
                R_k = R_val
            in_outage = (outage_enabled and (outage_start <= true_time[k] < outage_end)) or (R_k > 1e10)

        kf.predict()
        x_pred_bias = kf.x[0, 0]
        P_pred_bias = kf.P[0, 0]
        S_k = P_pred_bias + R_k

        if not in_outage:
            z = rubidium_error[k] - gnss_error_k
            innov_k = z - x_pred_bias
            innovations[k] = innov_k
            
            # Divergence Check
            # ratio = |innovation| / sqrt(S)
            ratio_k = abs(innov_k) / np.sqrt(S_k) if S_k > 0 else 0.0
            innovation_ratios[k] = ratio_k
            
            if ratio_k > threshold_divergence:
                consecutive_divergent_count += 1
            else:
                consecutive_divergent_count = 0
                
            if consecutive_divergent_count >= consecutive_limit:
                diverged_flags[k] = True
            else:
                diverged_flags[k] = False

            # Collect Kalman gain for the phase bias state
            kalman_gains[k] = P_pred_bias / S_k if S_k > 0 else 0.0

            kf.update(z)
        else:
            innovations[k] = np.nan
            innovation_ratios[k] = np.nan
            diverged_flags[k] = False
            consecutive_divergent_count = 0
            kalman_gains[k] = 0.0

        bias_estimates[k]  = kf.x[0, 0]
        drift_estimates[k] = kf.x[1, 0]
        kalman_variance[k] = kf.P[0, 0]

    master_error = rubidium_error - bias_estimates

    return KalmanOutput(
        bias_estimates,
        drift_estimates,
        kalman_variance,
        master_error,
        innovations,
        diverged_flags,
        innovation_ratios,
        kalman_gains
    )


# Alias — allows old import `from kalman_filter import run_kalman_filter` to work
run_kalman_filter = run_kalman_filter_v2


# ── Plotting functions ─────────────────────────────────────────────────────────

def plot_disciplined_clock_v2(
    true_time, gnss_error, master_error,
    outage_enabled, outage_start, outage_end,
    auto_scale=False
):
    """
    Disciplined Master Clock Phase Error vs Raw GNSS Error.
    """
    fig_m, ax_m = plt.subplots(figsize=(14, 4.2))
    ax_m.plot(true_time, gnss_error  * 1e9, label="Raw GNSS Error",
              color="#059669", alpha=0.3)
    ax_m.plot(true_time, master_error * 1e9, label="Disciplined Master Clock Error",
              color="#4f46e5", linewidth=1.8)
    if outage_enabled:
        ax_m.axvspan(outage_start, outage_end,
                     color="#ef4444", alpha=0.1, label="Outage Window")
    ax_m.set_ylabel("Error (ns)", color="#334155")
    ax_m.set_xlabel("Time (s)",   color="#334155")
    ax_m.grid(True, linestyle=":", color="#cbd5e1")
    ax_m.legend(loc="upper right", framealpha=0.8, labelcolor="#334155")
    if auto_scale:
        all_err = np.concatenate([gnss_error * 1e9, master_error * 1e9])
        span = max(np.max(all_err) - np.min(all_err), 10.0)
        ax_m.set_ylim(np.min(all_err) - 0.1*span, np.max(all_err) + 0.1*span)
    else:
        ax_m.set_ylim(-300, 300)
    ax_m.patch.set_facecolor('#ffffff')
    fig_m.patch.set_facecolor('#f8fafc')
    ax_m.tick_params(colors='#334155')
    plt.tight_layout()
    return fig_m


def plot_kalman_diagnostics_v2(
    true_time, rubidium_error, kalman_estimate, kalman_variance,
    gnss_bias, outage_enabled, outage_start, outage_end
):
    """
    Kalman filter estimation error (zero-mean) with 3-sigma confidence bounds.
    """
    fig_kf, ax_kf = plt.subplots(figsize=(14, 4.2))
    sigma = np.sqrt(kalman_variance)
    estimation_error = (rubidium_error - kalman_estimate - gnss_bias) * 1e9
    ax_kf.plot(true_time, estimation_error, color="#4f46e5",
               label="Filter Estimation Error (Zero-Mean)", linewidth=1.5)
    ax_kf.fill_between(
        true_time,
        -3 * sigma * 1e9, 3 * sigma * 1e9,
        color="#888888", alpha=0.2, label="3-Sigma Uncertainty Bounds"
    )
    if outage_enabled:
        ax_kf.axvspan(outage_start, outage_end,
                      color="#ef4444", alpha=0.1, label="Outage Window")
    ax_kf.set_ylabel("Error (ns)", color="#334155")
    ax_kf.set_xlabel("Time (s)",   color="#334155")
    ax_kf.grid(True, linestyle=":", color="#cbd5e1")
    ax_kf.legend(loc="upper right", framealpha=0.8, labelcolor="#334155")
    ax_kf.patch.set_facecolor('#ffffff')
    fig_kf.patch.set_facecolor('#f8fafc')
    ax_kf.tick_params(colors='#334155')
    plt.tight_layout()
    return fig_kf


def plot_estimated_drift(true_time, drift_estimates):
    """
    Estimated frequency drift over time.
    """
    fig_d, ax_d = plt.subplots(figsize=(14, 4.2))
    ax_d.plot(true_time, drift_estimates * 1e9, color="#9333ea",
              label="Estimated Frequency Drift", linewidth=1.5)
    ax_d.set_ylabel("Drift Rate (ns/s)", color="#334155")
    ax_d.set_xlabel("Time (s)",          color="#334155")
    ax_d.set_title("Estimated Frequency Drift",
                   color="#0f172a", fontsize=11, fontweight="bold")
    ax_d.grid(True, linestyle=":", color="#cbd5e1")
    ax_d.legend(loc="upper right", framealpha=0.8, labelcolor="#334155")
    ax_d.patch.set_facecolor('#ffffff')
    fig_d.patch.set_facecolor('#f8fafc')
    ax_d.tick_params(colors='#334155')
    plt.tight_layout()
    return fig_d


def plot_kalman_innovation(
    true_time, innovations,
    outage_enabled, outage_start, outage_end
):
    """
    Plot the Kalman filter measurement innovation over time.
    """
    fig_in, ax_in = plt.subplots(figsize=(14, 4.2))
    
    # Plot innovations in nanoseconds for readability
    ax_in.plot(
        true_time, innovations * 1e9, 
        color="#f59e0b", label="Filter Innovation (z - H @ x_pred)", 
        linewidth=1.5
    )
    
    if outage_enabled:
        ax_in.axvspan(
            outage_start, outage_end,
            color="#ef4444", alpha=0.1, label="Outage Window (No Updates)"
        )
        
    ax_in.set_ylabel("Innovation (ns)", color="#334155")
    ax_in.set_xlabel("Time (s)", color="#334155")
    ax_in.set_title("Kalman Filter Measurement Innovation",
                   color="#0f172a", fontsize=11, fontweight="bold")
    ax_in.grid(True, linestyle=":", color="#cbd5e1")
    ax_in.legend(loc="upper right", framealpha=0.8, labelcolor="#334155")
    ax_in.patch.set_facecolor('#ffffff')
    fig_in.patch.set_facecolor('#f8fafc')
    ax_in.tick_params(colors='#334155')
    plt.tight_layout()
    return fig_in


def plot_dynamic_r_diagnostics(
    true_time, sat_counts, measurement_noise_stds, kalman_gains,
    outage_enabled, outage_start, outage_end
):
    """
    Plots the relationship between visible satellite count, dynamic measurement noise R, and Kalman Gain.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

    # 1. Satellite Count
    ax1.plot(true_time, sat_counts, color="#2563eb", linewidth=1.5, label="Visible Satellites")
    ax1.set_ylabel("Satellite Count", color="#2563eb")
    ax1.tick_params(axis='y', labelcolor="#2563eb")
    ax1.set_title("Constellation-Driven Covariance Adaptation & Kalman Gain Diagnostics",
                 color="#0f172a", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle=":", color="#cbd5e1")
    
    # 2. Dynamic R (std dev in ns)
    # Capped for lock loss scenarios to prevent infinite scaling on plot
    capped_stds = np.minimum(measurement_noise_stds * 1e9, 500.0)
    ax2.plot(true_time, capped_stds, color="#dc2626", linewidth=1.5, label="Dynamic R (σ_R)")
    ax2.fill_between(true_time, capped_stds, color="#dc2626", alpha=0.1)
    ax2.set_ylabel("Measurement Std Dev σ_R (ns)", color="#dc2626")
    ax2.tick_params(axis='y', labelcolor="#dc2626")
    ax2.grid(True, linestyle=":", color="#cbd5e1")
    
    # 3. Kalman Gain
    ax3.plot(true_time, kalman_gains, color="#059669", linewidth=1.8, label="Kalman Gain (K_bias)")
    ax3.fill_between(true_time, kalman_gains, color="#059669", alpha=0.1)
    ax3.set_ylabel("Kalman Gain K_bias", color="#059669")
    ax3.tick_params(axis='y', labelcolor="#059669")
    ax3.set_xlabel("Time (s)", color="#334155")
    ax3.grid(True, linestyle=":", color="#cbd5e1")

    # Draw outage window on all subplots
    if outage_enabled:
        for ax in (ax1, ax2, ax3):
            ax.axvspan(outage_start, outage_end, color="#ef4444", alpha=0.08, label="GNSS Outage")

    for ax in (ax1, ax2, ax3):
        ax.patch.set_facecolor('#ffffff')
        ax.tick_params(colors='#334155')

    fig.patch.set_facecolor('#f8fafc')
    plt.tight_layout()
    return fig


def calculate_holdover_uncertainties(kf_P, dt, Q_matrix):
    """
    Propagates kf.P covariance matrix forward in time without measurement updates
    to predict 3-sigma holdover uncertainty bounds.
    """
    F_step = np.array([[1.0, dt],
                       [0.0, 1.0]])
    
    # Propagate 1 min (60 seconds)
    P_temp = kf_P.copy()
    steps_1m = max(1, int(60.0 / dt))
    for _ in range(steps_1m):
        P_temp = F_step @ P_temp @ F_step.T + Q_matrix
    sigma_1m = 3.0 * np.sqrt(max(0.0, P_temp[0, 0]))
    
    # Propagate 9 min more (10 min total)
    steps_9m = max(1, int(540.0 / dt))
    for _ in range(steps_9m):
        P_temp = F_step @ P_temp @ F_step.T + Q_matrix
    sigma_10m = 3.0 * np.sqrt(max(0.0, P_temp[0, 0]))
    
    # Propagate 50 min more (60 min total)
    steps_50m = max(1, int(3000.0 / dt))
    for _ in range(steps_50m):
        P_temp = F_step @ P_temp @ F_step.T + Q_matrix
    sigma_1h = 3.0 * np.sqrt(max(0.0, P_temp[0, 0]))
    
    return sigma_1m, sigma_10m, sigma_1h


def calculate_recovery_time(true_time, master_error, outage_end, threshold_ns=50.0, dt=1.0, window_seconds=30.0):
    """
    Calculates the time in seconds taken for the disciplined master clock error to fall
    and stabilize below the threshold_ns accuracy target after the outage ends.
    """
    threshold_s = threshold_ns * 1e-9
    post_outage_mask = true_time >= outage_end
    post_t = true_time[post_outage_mask]
    post_err = np.abs(master_error[post_outage_mask])
    
    # Convert window in seconds to number of epochs
    window_len = max(1, int(round(window_seconds / dt)))
    
    n = len(post_err)
    for i in range(n):
        end_idx = min(i + window_len, n)
        if np.all(post_err[i:end_idx] < threshold_s):
            return post_t[i] - outage_end
    return None



# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    from models.clock_model import simulate_rubidium_clock

    duration = 10000
    dt       = 1.0

    # Unpack all 5 return values from upgraded clock_model
    t, rb_err, rb_rw, rb_freq_offset, rb_aging = simulate_rubidium_clock(
        duration=duration, dt=dt,
        bias=1e-3, random_walk_step=1e-14,
        white_noise_std=1e-12, aging_rate=1e-13,
        seed=42,
    )

    rng      = np.random.default_rng(7)
    gnss_err = rng.normal(0, 30e-9, size=len(t))

    bias_est, drift_est, var, master_err, innovs = run_kalman_filter_v2(
        true_time=t, rubidium_error=rb_err, gnss_error=gnss_err,
        outage_enabled=True, outage_start=3000, outage_end=5000,
        R_val=(30e-9)**2,
    )

    print("kalman_filter.py - merged production file - test complete")
    print(f"  Final bias estimate : {bias_est[-1]*1e9:.3f} ns")
    print(f"  Final drift estimate: {drift_est[-1]*1e12:.6f} ps/s")
    print(f"  Master error (std)  : {master_err.std()*1e9:.3f} ns")
    print(f"  Final innovation    : {np.nanmean(np.abs(innovs))*1e9:.3f} ns (average magnitude)")
    print(f"  Q matrix (corrected Paper R17):")
    for row in build_Q_matrix(dt):
        print(f"    {row}")

    import matplotlib
    matplotlib.use('Agg')
    fig1 = plot_disciplined_clock_v2(t, gnss_err, master_err, True, 3000, 5000)
    fig2 = plot_kalman_diagnostics_v2(t, rb_err, bias_est, var, 0.0, True, 3000, 5000)
    fig3 = plot_estimated_drift(t, drift_est)
    fig4 = plot_kalman_innovation(t, innovs, True, 3000, 5000)
    print("  Plots generated (Agg backend - no display in headless mode)")


def calculate_autocorrelation(x, max_lag=20):
    """
    Computes the autocorrelation coefficients for lag 1 to max_lag.
    """
    N = len(x)
    if N == 0:
        return np.array([]), np.array([])
    mean = np.mean(x)
    var = np.var(x)
    if var < 1e-30:
        return np.arange(1, max_lag + 1), np.zeros(max_lag)
    
    lags = np.arange(1, max_lag + 1)
    acf = []
    for lag in lags:
        cov = np.mean((x[:-lag] - mean) * (x[lag:] - mean))
        acf.append(cov / var)
    return lags, np.array(acf)


def plot_innovation_autocorrelation(innovations, max_lag=20):
    """
    Plots the autocorrelation of innovations (correlogram) with 95% confidence bounds.
    """
    fig, ax = plt.subplots(figsize=(14, 4.2))
    
    lags, acf = calculate_autocorrelation(innovations, max_lag)
    N = len(innovations)
    conf_limit = 1.96 / np.sqrt(N) if N > 0 else 0.0
    
    # Draw bars for autocorrelation coefficients
    ax.bar(lags, acf, color="#4f46e5", alpha=0.8, width=0.4, label="Autocorrelation")
    ax.axhline(0, color="#64748b", linewidth=1.0)
    
    # Draw confidence bounds
    ax.axhline(conf_limit, color="#ef4444", linestyle="--", linewidth=1.2, label="95% Confidence Limit")
    ax.axhline(-conf_limit, color="#ef4444", linestyle="--", linewidth=1.2)
    
    ax.set_ylim(-1.0, 1.0)
    ax.set_xlim(0, max_lag + 1)
    ax.set_xticks(lags)
    ax.set_ylabel("Autocorrelation", color="#334155")
    ax.set_xlabel("Lag (epochs)", color="#334155")
    ax.set_title("Innovation Autocorrelation (Correlogram) & Whiteness Diagnostic",
                 color="#0f172a", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle=":", color="#cbd5e1")
    ax.legend(loc="upper right", framealpha=0.8, labelcolor="#334155")
    ax.patch.set_facecolor('#ffffff')
    fig.patch.set_facecolor('#f8fafc')
    ax.tick_params(colors='#334155')
    plt.tight_layout()
    return fig

