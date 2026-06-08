# models/clock_model.py
# Rubidium Atomic Clock Simulation Model
# Upgrade: Phase 1 — Frequency Random Walk Integration
#
# Mathematical change from previous version:
#   OLD: phase_error += random_step   (t^0.5 growth — WRONG for Rb)
#   NEW: freq_offset += random_step   (t^0.5 growth in frequency)
#        phase_error  = cumsum(freq_offset) * dt  (t^1.5 growth in phase — CORRECT for Rb)
#
# References:
#   R5  — Allan (1987) — defines the noise taxonomy and Allan deviation slopes
#   R1  — Khatri et al. (2014) — ISRO IRNSS Rb clock DVM parameters
#   R17 — Zhang et al. (2012) — corrected Q matrix formula for Kalman filter

import numpy as np
import matplotlib.pyplot as plt

# ─── Physical noise coefficients for ISRO IRNSS-class Rb clock ───────────────
# Source: Paper R1 (Khatri 2014 — ISRO IRNSS DVM)
# These govern the three noise regimes visible in the Allan deviation plot
H0     = 2e-19   # White Frequency Modulation   — slope -0.5 in ADEV
H_NEG1 = 7e-23   # Flicker Frequency Modulation — slope  0.0 in ADEV
H_NEG2 = 2e-30   # Random Walk Frequency Mod.   — slope +0.5 in ADEV

# ─── Main simulation function ─────────────────────────────────────────────────

def simulate_rubidium_clock(
    duration,
    dt,
    bias,
    random_walk_step,
    white_noise_std,
    aging_rate=0.0,
    seed=None,
):
    """
    Simulate a Rubidium atomic clock using a frequency random walk model.

    Mathematical model
    ------------------
    The clock error is the sum of four physical components:

        clock_error(t) = bias
                       + phase_random_walk(t)   [from integrated frequency walk]
                       + aging_error(t)          [quadratic — linearly growing frequency]
                       + white_noise(t)          [independent per sample]

    Frequency random walk (the key upgrade):
        freq_offset[n]    = cumsum(random_walk_steps)      <- frequency drifts
        phase_random_walk = cumsum(freq_offset) * dt       <- phase integrates that drift

    This produces phase error growth proportional to t^(3/2), which matches
    real-world atomic clock random walk FM behaviour (Paper R5, Allan 1987).
    The old model accumulated phase steps directly, giving t^(1/2) growth —
    characteristic of a quartz oscillator, not a Rubidium standard.

    Parameters
    ----------
    duration         : float  — total simulation time in seconds
    dt               : float  — sample interval in seconds
    bias             : float  — fixed initial phase offset (seconds)
    random_walk_step : float  — std dev of each frequency random walk step (s/s per step)
    white_noise_std  : float  — std dev of short-term white phase noise (seconds)
    aging_rate       : float  — linear frequency aging rate (s/s^2), produces 0.5*D*t^2 phase error
    seed             : int    — optional random seed for reproducibility

    Returns  (5 values — unpack all five at every call site)
    -------
    true_time        : ndarray — simulation epoch array (seconds)
    clock_error      : ndarray — total clock timing error (seconds)
    phase_random_walk: ndarray — phase error from integrated frequency walk only (seconds)
    freq_offset      : ndarray — accumulated frequency offset (seconds/second) — NEW
    aging_error      : ndarray — phase error from linear frequency aging (seconds)
    """
    rng = np.random.default_rng(seed)

    true_time = np.arange(0, duration, dt)
    N = len(true_time)

    # ── 1. Frequency Random Walk → Phase ─────────────────────────────────────
    # Step 1: frequency takes a random step each epoch
    freq_steps = rng.normal(0, random_walk_step, size=N)
    freq_steps[0] = 0.0                        # clean initial condition

    # Step 2: accumulate frequency steps → frequency offset time series
    freq_offset = np.cumsum(freq_steps)        # units: seconds/second (fractional freq)

    # Step 3: integrate frequency offset → phase error
    # phase = integral of frequency, approximated as cumulative sum * dt
    phase_random_walk = np.cumsum(freq_offset) * dt   # units: seconds

    # ── 2. Linear Frequency Aging → Quadratic Phase ───────────────────────────
    # A linearly growing frequency offset produces a quadratic phase error
    # phase_aging(t) = 0.5 * aging_rate * t^2
    aging_error = 0.5 * aging_rate * (true_time ** 2)

    # ── 3. White Phase Noise ──────────────────────────────────────────────────
    # Independent Gaussian sample per epoch — no temporal correlation
    white_noise = rng.normal(0, white_noise_std, size=N)

    # ── 4. Total Clock Error ──────────────────────────────────────────────────
    clock_error = bias + phase_random_walk + aging_error + white_noise

    return true_time, clock_error, phase_random_walk, freq_offset, aging_error

# ─── Plotting functions ───────────────────────────────────────────────────────

def plot_rubidium_clock(true_time, rubidium_error, rb_aging, rb_aging_ns_s):
    """
    Plot total Rubidium clock phase error with optional aging overlay.
    Used in: Streamlit Tab 3, standalone test.
    """
    fig_rb, ax_rb = plt.subplots(figsize=(7, 3.8))
    ax_rb.plot(
        true_time, rubidium_error * 1e6,
        color="#db2777", label="Total Rubidium Error", linewidth=2.0
    )
    if rb_aging_ns_s > 0.0:
        ax_rb.plot(
            true_time, rb_aging * 1e6,
            color="#ea580c", label="Linear Frequency Aging Component",
            linestyle="--", linewidth=1.5
        )
    ax_rb.set_ylabel("Error (µs)", color="#334155")
    ax_rb.set_xlabel("Time (s)", color="#334155")
    ax_rb.set_title(
        "Rubidium Clock Total Phase Error",
        color="#0f172a", fontsize=11, fontweight="bold"
    )
    ax_rb.grid(True, linestyle=":", color="#cbd5e1")
    ax_rb.legend(loc="upper left", framealpha=0.8, labelcolor="#334155")
    ax_rb.patch.set_facecolor('#ffffff')
    fig_rb.patch.set_facecolor('#f8fafc')
    ax_rb.tick_params(colors='#334155')
    plt.tight_layout()
    return fig_rb

def plot_rubidium_frequency_wander(t, freq_offset):
    """
    Plot the instantaneous frequency offset (wander) of the Rb oscillator over time.

    This is the NEW plot introduced in the upgrade. It shows the random walk in
    frequency — the physical process that drives long-term phase drift. In a real
    Rb clock this corresponds to the output of the servo electronics tracking the
    hyperfine resonance frequency.

    Units: frequency offset is dimensionless (fractional frequency, s/s), plotted
    in parts-per-trillion (ppt = 1e-12) for readability.

    Used in: Streamlit Tab 3 expander, standalone test.
    """
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(
        t, freq_offset * 1e12,
        color="#7c3aed", label="Frequency Offset", linewidth=1.5
    )
    ax.axhline(0, color="#94a3b8", linewidth=0.8, linestyle="--")
    ax.set_ylabel("Frequency Offset (ppt = ×10⁻¹²)", color="#334155")
    ax.set_xlabel("Time (s)", color="#334155")
    ax.set_title(
        "Rubidium Frequency Wander (Random Walk FM)",
        color="#0f172a", fontsize=11, fontweight="bold"
    )
    ax.grid(True, linestyle=":", color="#cbd5e1")
    ax.legend(loc="upper left", framealpha=0.8, labelcolor="#334155")
    ax.patch.set_facecolor('#ffffff')
    fig.patch.set_facecolor('#f8fafc')
    ax.tick_params(colors='#334155')
    plt.tight_layout()
    return fig

def plot_error_without_bias(t, error, bias):
    """Plot clock error with constant bias subtracted."""
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(t, (error - bias) * 1e6, color="#4f46e5", label="Error without Bias", linewidth=1.5)
    ax.set_ylabel("Error (µs)", color="#334155")
    ax.set_xlabel("Time (s)", color="#334155")
    ax.set_title("Clock Error without Bias", color="#0f172a", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle=":", color="#cbd5e1")
    ax.legend(loc="upper left", framealpha=0.8, labelcolor="#334155")
    ax.patch.set_facecolor('#ffffff')
    fig.patch.set_facecolor('#f8fafc')
    ax.tick_params(colors='#334155')
    plt.tight_layout()
    return fig

def plot_rubidium_random_walk(t, rw):
    """Plot the accumulated phase random walk (integrated from frequency walk)."""
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(t, rw * 1e6, color="#2563eb", label="Phase Random Walk (∫ freq_offset · dt)", linewidth=1.5)
    ax.set_ylabel("Error (µs)", color="#334155")
    ax.set_xlabel("Time (s)", color="#334155")
    ax.set_title("Integrated Phase Random Walk Component", color="#0f172a", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle=":", color="#cbd5e1")
    ax.legend(loc="upper left", framealpha=0.8, labelcolor="#334155")
    ax.patch.set_facecolor('#ffffff')
    fig.patch.set_facecolor('#f8fafc')
    ax.tick_params(colors='#334155')
    plt.tight_layout()
    return fig

def plot_rubidium_aging(t, aging):
    """Plot the quadratic frequency aging phase error component."""
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(t, aging * 1e6, color="#ea580c", label="Aging Component (0.5·D·t²)", linewidth=1.5)
    ax.set_ylabel("Error (µs)", color="#334155")
    ax.set_xlabel("Time (s)", color="#334155")
    ax.set_title("Aging Component", color="#0f172a", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle=":", color="#cbd5e1")
    ax.legend(loc="upper left", framealpha=0.8, labelcolor="#334155")
    ax.patch.set_facecolor('#ffffff')
    fig.patch.set_facecolor('#f8fafc')
    ax.tick_params(colors='#334155')
    plt.tight_layout()
    return fig

def plot_rubidium_distribution(error_without_bias):
    """Plot histogram of clock phase error (bias removed)."""
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.hist(error_without_bias * 1e6, bins=50, color="#64748b", edgecolor="#475569")
    ax.set_ylabel("Count", color="#334155")
    ax.set_xlabel("Error (µs)", color="#334155")
    ax.set_title("Clock Error Distribution", color="#0f172a", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle=":", color="#cbd5e1")
    ax.patch.set_facecolor('#ffffff')
    fig.patch.set_facecolor('#f8fafc')
    ax.tick_params(colors='#334155')
    plt.tight_layout()
    return fig

# ─── Standalone test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    # Use the config if available, otherwise use safe test defaults
    try:
        from config import (
            SIM_DURATION, SIM_DT, RB_BIAS,
            RB_RANDOM_WALK_STEP, RB_WHITE_NOISE_STD, RB_AGING_RATE
        )
    except ImportError:
        print("config.py not found — using test defaults")
        SIM_DURATION        = 10000
        SIM_DT              = 1.0
        RB_BIAS             = 1e-3
        RB_RANDOM_WALK_STEP = 1e-14   # physically realistic for Rb
        RB_WHITE_NOISE_STD  = 1e-12   # physically realistic for Rb
        RB_AGING_RATE       = 1e-13

    print("Testing models/clock_model.py (upgraded frequency random walk model)...")

    # ── Run simulation — UNPACK ALL 5 RETURN VALUES ──────────────────────────
    t, error, phase_rw, freq_offset, aging = simulate_rubidium_clock(
        duration=SIM_DURATION,
        dt=SIM_DT,
        bias=RB_BIAS,
        random_walk_step=RB_RANDOM_WALK_STEP,
        white_noise_std=RB_WHITE_NOISE_STD,
        aging_rate=RB_AGING_RATE,
        seed=42,
    )

    print(f"Simulation complete. Epochs: {len(t)}")

    # ── Statistics ────────────────────────────────────────────────────────────
    print("\n========== RUBIDIUM CLOCK STATISTICS ==========")
    print(f"Start error   (µs): {error[0]  * 1e6:.4f}")
    print(f"Middle error  (µs): {error[len(error)//2] * 1e6:.4f}")
    print(f"End error     (µs): {error[-1] * 1e6:.4f}")
    print(f"Mean error    (µs): {np.mean(error) * 1e6:.4f}")
    print(f"Std error     (µs): {np.std(error)  * 1e6:.4f}")
    print(f"Total wander  (µs): {(error[-1] - error[0]) * 1e6:.4f}")
    print(f"\nFrequency offset range (ppt): "
          f"{freq_offset.min()*1e12:.4f}  to  {freq_offset.max()*1e12:.4f}")

    # ── Allan deviation slope check ───────────────────────────────────────────
    try:
        import allantools
        phase = error
        rate  = 1.0 / SIM_DT
        taus, adev, _, _ = allantools.oadev(phase, rate=rate, data_type="phase")

        print("\n========== ALLAN DEVIATION SLOPE CHECK ==========")
        print(f"  {'Tau (s)':<12} {'ADEV':<16} {'Slope':<10} {'Expected'}")
        print("  " + "-" * 58)
        log_tau  = np.log10(taus)
        log_adev = np.log10(adev)
        slopes   = np.diff(log_adev) / np.diff(log_tau)

        for i in range(min(15, len(slopes))):
            tau   = taus[i]
            a     = adev[i]
            slope = slopes[i]
            if tau < 100:
                expected = "-0.5 (white FM)"
            elif tau < 5000:
                expected = " 0.0 (flicker FM)"
            else:
                expected = "+0.5 (random walk FM)"
            print(f"  {tau:<12.1f} {a:<16.3e} {slope:<+10.3f} {expected}")
    except ImportError:
        print("allantools not installed — skipping slope check")

    # ── Generate all plots ────────────────────────────────────────────────────
    print("\nGenerating plots...")
    fig1 = plot_rubidium_clock(t, error, aging, RB_AGING_RATE * 1e9)
    fig2 = plot_error_without_bias(t, error, RB_BIAS)
    fig3 = plot_rubidium_random_walk(t, phase_rw)
    fig4 = plot_rubidium_frequency_wander(t, freq_offset)   # NEW plot
    fig5 = plot_rubidium_aging(t, aging)
    fig6 = plot_rubidium_distribution(error - RB_BIAS)

    print("Close plot windows to exit.")
    plt.show()
