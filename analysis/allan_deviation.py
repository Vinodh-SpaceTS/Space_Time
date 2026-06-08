# analysis/allan_deviation.py
# Allan Deviation Comparison — Rb vs GNSS vs Disciplined Master Clock
#
# Updated for clock_model.py upgrade:
#   simulate_rubidium_clock now returns 5 values, not 4.
#   Unpack: true_time, rubidium_error, rb_rw, rb_freq_offset, rb_aging
#
# References:
#   R5  — Allan (1987) — defines Allan deviation and noise slope taxonomy
#   R13 — Hauschild & Montenbruck (2019) — Kalman clock estimation

import numpy as np
import matplotlib.pyplot as plt
import allantools


# ─── Core Allan deviation computation ────────────────────────────────────────

def calculate_allan_deviation(phase_errors_s, dt=None, rate=None):
    """
    Calculate overlapping Allan deviation (OADEV) for phase error data.

    Parameters
    ----------
    phase_errors_s : array-like
        Clock phase errors in SECONDS. Must be phase data (time offset),
        not frequency data. Verify the units before calling.
    dt : float, optional
        Sample interval in seconds.
    rate : float, optional
        Sample rate in Hz. If specified, dt = 1.0 / rate.

    Returns
    -------
    taus : ndarray — averaging times in seconds
    adev : ndarray — Allan deviation values (dimensionless, s/s)

    Notes
    -----
    Uses overlapping ADEV (oadev) not standard ADEV (adev).
    OADEV uses more data and produces lower-variance estimates.
    Both are valid; OADEV is preferred for shorter datasets.
    """
    if dt is None and rate is None:
        dt = 1.0
        r_val = 1.0
    elif dt is not None and rate is None:
        r_val = 1.0 / dt
    elif rate is not None and dt is None:
        r_val = rate
        dt = 1.0 / rate
    else:
        r_val = rate

    taus, adev, _, _ = allantools.oadev(
        np.asarray(phase_errors_s, dtype=float),
        rate=r_val,
        data_type="phase"
    )
    return taus, adev


# ─── Slope validation ─────────────────────────────────────────────────────────

def check_rb_allan_slopes(taus, adev, verbose=True):
    """
    Compute log-log slopes between consecutive tau points and analyze
    expected Rubidium clock noise taxonomy (Paper R5 — Allan 1987).

    We analyze three key regions of the Rubidium clock stability curve:
        1. Very Short-term: tau <= 3 s       slope ≈ -1.0   White Phase Modulation (WPM)
        2. Mid-term:        3 < tau < 10 s   crossover / transition region
        3. Long-term:       tau >= 10 s      slope ≈ +0.5 to +1.0  Random Walk FM or Aging

    Returns
    -------
    slopes : ndarray of log-log slopes between consecutive tau points
    passed : bool — True if dominant regions have approximately correct slopes
    """
    taus  = np.asarray(taus)
    adev  = np.asarray(adev)
    log_t = np.log10(taus)
    log_a = np.log10(adev)
    
    # Calculate consecutive slopes for backward compatibility
    slopes = np.diff(log_a) / np.diff(log_t)

    # Segmented analysis using linear fits
    regions = [
        {"name": "White PM (Short-term)", "min_tau": 0.0, "max_tau": 3.0, "target": "~ -1.0"},
        {"name": "Transition (Mid-term)", "min_tau": 3.0, "max_tau": 10.0, "target": "variable"},
        {"name": "Random Walk FM / Aging (Long-term)", "min_tau": 10.0, "max_tau": np.inf, "target": "+0.5 to +1.0"}
    ]

    fits = {}
    for reg in regions:
        mask = (taus >= reg["min_tau"]) & (taus < reg["max_tau"])
        if np.sum(mask) >= 2:
            slope, _ = np.polyfit(log_t[mask], log_a[mask], 1)
            fits[reg["name"]] = slope
        else:
            fits[reg["name"]] = None

    if verbose:
        print("\n========== Allan Deviation Slope Check (Rb Clock) ==========")
        print(f"  {'Region':<32} {'Tau Range (s)':<18} {'Fitted Slope':<14} {'Expected':<14} {'Status'}")
        print("  " + "-" * 88)
        
        # 1. Short-term White PM
        val = fits["White PM (Short-term)"]
        if val is not None:
            status = "PASS" if val < -0.4 else "CHECK"
            print(f"  {'White PM (Short-term)':<32} {'tau <= 3':<18} {val:<+14.3f} {'~ -1.0':<14} {status}")
        else:
            print(f"  {'White PM (Short-term)':<32} {'tau <= 3':<18} {'N/A':<14} {'~ -1.0':<14} {'NO DATA'}")
            
        # 2. Transition
        val = fits["Transition (Mid-term)"]
        if val is not None:
            print(f"  {'Transition (Mid-term)':<32} {'3 < tau < 10':<18} {val:<+14.3f} {'variable':<14} {'OK'}")
        else:
            print(f"  {'Transition (Mid-term)':<32} {'3 < tau < 10':<18} {'N/A':<14} {'variable':<14} {'NO DATA'}")
            
        # 3. Long-term
        val = fits["Random Walk FM / Aging (Long-term)"]
        if val is not None:
            # Expected is between +0.5 (RWFM) and +1.0 (Aging)
            status = "PASS" if 0.2 < val < 1.3 else "CHECK"
            print(f"  {'RW FM / Aging (Long-term)':<32} {'tau >= 10':<18} {val:<+14.3f} {'+0.5 to +1.0':<14} {status}")
        else:
            print(f"  {'RW FM / Aging (Long-term)':<32} {'tau >= 10':<18} {'N/A':<14} {'+0.5 to +1.0':<14} {'NO DATA'}")

    # Pass criterion:
    # 1. Short-term slope is negative (White PM is present)
    # 2. Long-term slope is positive (Random Walk FM or Aging is present)
    passed = True
    if fits["White PM (Short-term)"] is not None and fits["White PM (Short-term)"] >= -0.2:
        passed = False
    if fits["Random Walk FM / Aging (Long-term)"] is not None and fits["Random Walk FM / Aging (Long-term)"] <= 0.2:
        passed = False
        
    if verbose:
        print(f"\n  Overall Slope Check: {'PASS' if passed else 'FAIL - check noise parameters'}")
        if not passed:
            print("  Hint: white_noise_std is probably too large relative to random_walk_step.")
            print("  For a realistic Rb clock: white_noise_std ~ 1e-12, random_walk_step ~ 1e-14")

    return slopes, passed


# ─── Comparison plot ──────────────────────────────────────────────────────────

def plot_allan_deviation_comparison(
    taus_rb, adev_rb,
    taus_gnss, adev_gnss,
    taus_master, adev_master
):
    """
    Log-log comparison of Allan deviation for Rb, GNSS, and disciplined master clock.

    The physically correct relationship between the three curves:
      • Rb curve:     below GNSS at SHORT tau (Rb is more stable short-term)
      • GNSS curve:   below Rb at LONG tau (GNSS does not drift long-term)
      • Master curve: follows Rb short-term, follows GNSS long-term (Kalman crossover)

    If GNSS is below Rb at short tau, the simulation parameters are unrealistic.
    """
    # Configure modern typography
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Inter', 'DejaVu Sans', 'Arial', 'Helvetica']

    fig_ad, ax_ad = plt.subplots(figsize=(14, 5.5))

    # Plot the main error curves
    ax_ad.loglog(
        taus_rb, adev_rb,
        marker='o', markersize=4.5,
        color="#db2777", linewidth=2.0,
        label="Raw Rubidium"
    )
    ax_ad.loglog(
        taus_gnss, adev_gnss,
        marker='s', markersize=4.5,
        color="#059669", linewidth=1.5,
        label="Raw GNSS"
    )
    ax_ad.loglog(
        taus_master, adev_master,
        marker='^', markersize=4.5,
        color="#4f46e5", linewidth=2.5,
        label="Disciplined Master Clock (Kalman)"
    )

    # ─── Crossover Detection & Stability Domain Shading ──────────────────────
    crossover_tau = None
    if len(taus_rb) > 0 and len(adev_rb) == len(adev_gnss):
        # Find the first tau where Rb stability becomes worse than GNSS stability
        idx_above = np.where(adev_rb > adev_gnss)[0]
        if len(idx_above) > 0:
            crossover_tau = taus_rb[idx_above[0]]

    if crossover_tau is not None:
        # Highlight stability regimes with soft fills
        ax_ad.axvspan(
            taus_rb[0], crossover_tau,
            color="#db2777", alpha=0.03,
            label="Rb Stability Domain"
        )
        ax_ad.axvspan(
            crossover_tau, taus_rb[-1],
            color="#059669", alpha=0.03,
            label="GNSS Stability Domain"
        )
        # Add vertical crossover line
        ax_ad.axvline(
            x=crossover_tau,
            color="#64748b", linestyle="--", linewidth=1.2, alpha=0.5
        )
        # Labeled annotation using a blended coordinate system
        import matplotlib.transforms as mtransforms
        trans = mtransforms.blended_transform_factory(ax_ad.transData, ax_ad.transAxes)
        ax_ad.text(
            crossover_tau * 1.1, 0.05,  # x in data coords, y at 5% height from axes bottom
            f"Stability Crossover\nτ ≈ {crossover_tau:.0f} s",
            transform=trans,
            fontsize=8.5,
            color="#475569",
            fontweight="semibold",
            verticalalignment="bottom"
        )

    # ─── Reference Slopes ───────────────────────────────────────────────────
    if len(taus_rb) > 0:
        # White PM slope reference line (-1.0 slope)
        tau_ref_pm = np.array([taus_rb[0], min(10.0, taus_rb[-1])])
        adev_ref_pm = adev_rb[0] * (tau_ref_pm / tau_ref_pm[0]) ** (-1.0)
        ax_ad.loglog(
            tau_ref_pm, adev_ref_pm,
            color="#94a3b8", linewidth=1.0, linestyle="--",
            label="Slope -1.0 (White PM)"
        )
        
        # Random Walk FM slope reference line (+0.5 slope)
        if taus_rb[-1] > 10.0:
            tau_ref_rw = np.array([max(10.0, taus_rb[-1] / 10.0), taus_rb[-1]])
            adev_ref_rw = adev_rb[-1] * (tau_ref_rw / tau_ref_rw[-1]) ** (0.5) * 0.4
            ax_ad.loglog(
                tau_ref_rw, adev_ref_rw,
                color="#64748b", linewidth=1.0, linestyle="-.",
                label="Slope +0.5 (Random Walk FM)"
            )

    ax_ad.set_title(
        "Allan Deviation Comparison\n"
        "Rubidium  vs  GNSS  vs  GNSSDO Master Clock",
        fontsize=11.5, fontweight="bold", color="#0f172a", pad=12
    )
    ax_ad.set_xlabel("Averaging Time τ (s)", color="#334155", labelpad=8)
    ax_ad.set_ylabel("Allan Deviation (s/s)", color="#334155", labelpad=8)
    ax_ad.grid(True, which="both", linestyle=":", color="#cbd5e1")
    ax_ad.legend(loc="lower left", framealpha=0.9, labelcolor="#334155", fontsize=9)
    ax_ad.patch.set_facecolor('#ffffff')
    fig_ad.patch.set_facecolor('#f8fafc')
    ax_ad.tick_params(colors='#334155')
    plt.tight_layout()
    return fig_ad


# ─── Standalone test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    from models.clock_model import simulate_rubidium_clock

    # ── Import Kalman filter if available ─────────────────────────────────────
    try:
        from models.kalman_filter import run_kalman_filter_v2
        kalman_available = True
    except ImportError:
        kalman_available = False
        print("kalman_filter.py not found — master clock ADEV will be skipped")

    # ── Simulation settings ───────────────────────────────────────────────────
    duration = 10000
    dt       = 1.0

    # ── Rubidium Clock — UNPACK ALL 5 RETURN VALUES ───────────────────────────
    true_time, rubidium_error, rb_rw, rb_freq_offset, rb_aging = simulate_rubidium_clock(
        duration=duration,
        dt=dt,
        bias=1e-3,
        random_walk_step=1e-14,   # physically realistic for ISRO Rb class
        white_noise_std=1e-12,    # physically realistic for ISRO Rb class
        aging_rate=1e-13,
        seed=42,
    )

    # ── GNSS Receiver (simulated Gaussian noise — Phase 1 model) ──────────────
    # gps_time = true_time + np.random.normal(0, 30e-9)
    # GNSS error is white noise — ADEV slope should be -0.5 everywhere
    rng_gnss  = np.random.default_rng(7)
    gnss_error = true_time + rng_gnss.normal(0, 30e-9, size=len(true_time))
    # Subtract true_time to get only the error component for ADEV
    gnss_error_only = gnss_error - true_time

    # ── Kalman master clock ───────────────────────────────────────────────────
    if kalman_available:
        _, _, _, master_error, _ = run_kalman_filter_v2(
            true_time=true_time,
            rubidium_error=rubidium_error,
            gnss_error=gnss_error_only,
            outage_enabled=False,
            outage_start=0,
            outage_end=0,
            Q_bias=1e-20,
            Q_drift=1e-22,
            R_val=(30e-9)**2,
        )
    else:
        # Placeholder: average of Rb and GNSS
        master_error = 0.5 * (rubidium_error + gnss_error_only)

    # ── Compute Allan deviations — pass dt explicitly ─────────────────────────
    taus_rb,     adev_rb     = calculate_allan_deviation(rubidium_error,  dt=dt)
    taus_gnss,   adev_gnss   = calculate_allan_deviation(gnss_error_only, dt=dt)
    taus_master, adev_master = calculate_allan_deviation(master_error,    dt=dt)

    # ── Print results ─────────────────────────────────────────────────────────
    print("\n=========== ALLAN DEVIATION RESULTS ===========")
    print("\nRubidium (first 10 tau values):")
    for t_val, a in zip(taus_rb[:10], adev_rb[:10]):
        print(f"  Tau = {t_val:8.1f} s    ADEV = {a:.3e}")

    print("\nGNSS (first 10 tau values):")
    for t_val, a in zip(taus_gnss[:10], adev_gnss[:10]):
        print(f"  Tau = {t_val:8.1f} s    ADEV = {a:.3e}")

    print("\nMaster Clock (first 10 tau values):")
    for t_val, a in zip(taus_master[:10], adev_master[:10]):
        print(f"  Tau = {t_val:8.1f} s    ADEV = {a:.3e}")

    # ── Rb slope validation ───────────────────────────────────────────────────
    slopes, passed = check_rb_allan_slopes(taus_rb, adev_rb, verbose=True)

    # ── Cross-over sanity check ───────────────────────────────────────────────
    # At tau=1s, Rb should be MORE stable (lower ADEV) than GNSS
    if len(adev_rb) > 0 and len(adev_gnss) > 0:
        print("\n=========== SANITY CHECK ===========")
        print(f"  Rb   ADEV @ tau~1s : {adev_rb[0]:.3e}")
        print(f"  GNSS ADEV @ tau~1s : {adev_gnss[0]:.3e}")
        if adev_rb[0] < adev_gnss[0]:
            print("  PASS: Rb is more stable than GNSS at short tau (physically correct)")
        else:
            print("  FAIL: GNSS appears more stable than Rb at tau=1s")
            print("  Hint: Rb white_noise_std is too large. Reduce to ~1e-12 s.")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig = plot_allan_deviation_comparison(
        taus_rb, adev_rb,
        taus_gnss, adev_gnss,
        taus_master, adev_master,
    )
    plt.show()