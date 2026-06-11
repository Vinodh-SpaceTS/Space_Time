# app.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import time
from datetime import datetime, timedelta

# ─── Live Telemetry Playback Helpers (Defined early for use in tabs) ───────────

def format_telemetry_time(dt_obj, seconds_offset=0.0, date_only=False, time_only=False):
    """
    Formats a datetime object with a floating-point seconds offset added to it,
    rendering it as a UTC string representation with nanosecond sub-seconds.
    """
    whole_sec = int(seconds_offset)
    frac_sec = seconds_offset - whole_sec
    
    dt_at_epoch = dt_obj + timedelta(seconds=whole_sec)
    ns = int(round(frac_sec * 1e9))
    if ns < 0:
        dt_at_epoch -= timedelta(seconds=1)
        ns += 1000000000
        
    dt_str = dt_at_epoch.strftime("%Y-%m-%d %H:%M:%S")
    if date_only:
        return dt_at_epoch.strftime("%Y-%m-%d")
    elif time_only:
        return f"{dt_at_epoch.strftime('%H:%M:%S')}.{ns:09d}"
    else:
        return f"{dt_str}.{ns:09d}"

def format_offset(val_sec):
    val_ns = val_sec * 1e9
    if abs(val_ns) < 1000:
        return f"{val_ns:+.0f} ns"
    elif abs(val_ns) < 1e6:
        return f"{val_ns/1e3:+.2f} µs"
    else:
        return f"{val_ns/1e6:+.2f} ms"

def format_uncertainty(val_sec):
    val_ns = val_sec * 1e9
    if val_ns < 1000:
        return f"±{val_ns:.0f} ns"
    else:
        return f"±{val_ns/1e3:.1f} µs"

# Configure Streamlit page
st.set_page_config(
    page_title="GNSSDO Synchronization Telemetry",
    layout="wide"
)

# Add custom styled components
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [data-testid="stAppViewContainer"], .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Sidebar styling refinement */
    [data-testid="stSidebar"] {
        border-right: 1px solid #cbd5e1;
    }
    
    /* Custom metric card */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 18px 12px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.04);
        margin-bottom: 10px;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(99, 102, 241, 0.4);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 2px 0;
        letter-spacing: -0.03em;
    }
    .metric-label {
        font-size: 0.75rem;
        color: #64748b !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 500;
    }
    
    /* Custom colors for values to override global black color */
    .value-blue { color: #1d4ed8 !important; }
    .value-green { color: #047857 !important; }
    .value-orange { color: #c2410c !important; }
    .value-red { color: #b91c1c !important; }
    .value-purple { color: #6d28d9 !important; }

    /* Telemetry Panel Styles (Light/White Theme) */
    .telemetry-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px;
        color: #0f172a;
        font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        margin-bottom: 15px;
    }
    .telemetry-header {
        border-bottom: 1px solid #f1f5f9;
        padding-bottom: 8px;
        margin-bottom: 12px;
        font-weight: 700;
        color: #4f46e5;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    .telemetry-value-large {
        font-size: 2.4rem;
        font-weight: 700;
        color: #4f46e5;
        letter-spacing: 0.02em;
        margin: 12px 0;
    }
    .telemetry-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
        font-size: 0.95rem;
        border-bottom: 1px dotted #f1f5f9;
        padding-bottom: 4px;
    }
    .telemetry-row:last-child {
        border-bottom: none;
        margin-bottom: 0;
        padding-bottom: 0;
    }
    .telemetry-label {
        color: #64748b;
    }
    .telemetry-value {
        color: #0f172a;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Add custom paths to sys.path so modules can find config.py and each other
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'models'))
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'analysis'))

# Import functions from modules
from clock_model import simulate_rubidium_clock, plot_rubidium_clock, plot_error_without_bias, plot_rubidium_random_walk, plot_rubidium_aging, plot_rubidium_distribution, plot_rubidium_frequency_wander
from gnss_time_model import simulate_gnss_time, plot_gnss_time_components, plot_full_gnss_error, plot_gnss_component_breakdown
from kalman_filter import run_kalman_filter_v2, plot_disciplined_clock_v2, plot_kalman_diagnostics_v2, plot_estimated_drift, plot_kalman_innovation, calculate_holdover_uncertainties, build_Q_matrix, plot_dynamic_r_diagnostics, calculate_recovery_time
from gnss_analysis import parse_rinex_file, plot_satellite_visibility, parse_rinex_metadata
from analysis.allan_deviation import calculate_allan_deviation, plot_allan_deviation_comparison, allan_plot_lock
from config import RINEX_FILEPATH, GLOBAL_RANDOM_SEED, validate_config
from receivers.simulated_receiver import SimulatedReceiver

# Run config validation at startup
validate_config()


# ====================================================
# TITLE BLOCK
# ====================================================
st.title("GNSSDO Synchronization Telemetry")

# ====================================================
# SCENARIO SELECTOR
# ====================================================
st.subheader("Scenario Control")
scenario = st.radio(
    "Select operation profile:",
    ["Normal Operation", "GNSS Outage", "GNSS Degraded"],
    horizontal=True
)

# Default simulation settings
duration = 10000
dt = 1

# Scenario presets mapping
if scenario == "Normal Operation":
    outage_enabled = False
    outage_start = 3000
    outage_end = 5000
    rb_bias_ns = 1000000.0        # 1 ms
    rb_rw_step_ns = 0.1           # 0.1 ns (1e-10 s)
    rb_noise_ns = 5.0             # 5 ns
    rb_aging_ns_s = 0.0001        # 0.0001 ns/s^2 (1e-13 s/s^2)
    gnss_bias_ns = 100.0          # 100 ns
    gnss_sat_ns = 20.0
    gnss_prop_ns = 30.0
    gnss_meas_ns = 50.0
    use_gauss_markov = False
    
elif scenario == "GNSS Outage":
    outage_enabled = True
    outage_start = 3000
    outage_end = 7000             # Default to 3000s -> 7000s outage for Experiment testing
    rb_bias_ns = 1000000.0
    rb_rw_step_ns = 0.1
    rb_noise_ns = 5.0
    rb_aging_ns_s = 0.0001
    gnss_bias_ns = 100.0
    gnss_sat_ns = 20.0
    gnss_prop_ns = 30.0
    gnss_meas_ns = 50.0
    use_gauss_markov = False
    
elif scenario == "GNSS Degraded":
    outage_enabled = False
    outage_start = 3000
    outage_end = 5000
    rb_bias_ns = 1000000.0
    rb_rw_step_ns = 0.1
    rb_noise_ns = 5.0
    rb_aging_ns_s = 0.0001
    gnss_bias_ns = 100.0
    gnss_sat_ns = 40.0
    gnss_prop_ns = 150.0          # Intense atmospheric storm delay
    gnss_meas_ns = 200.0          # High measurement noise (jamming/multipath)
    use_gauss_markov = True       # Correlated slowly-varying propagation error

# ====================================================
# SIDEBAR PARAMETER OVERRIDES
# ====================================================
st.sidebar.header("Configuration")

with st.sidebar.expander("Simulation Control", expanded=True):
    duration = st.sidebar.slider("Duration (s)", 1000, 20000, duration, 1000)
    dt = st.sidebar.slider("Time Step (s)", 1, 10, dt, 1)
    target_accuracy_ns = st.sidebar.number_input("Target Accuracy (ns)", 5.0, 500.0, 50.0, 5.0, help="Target disciplined master clock error limit under active tracking.")
    auto_scale_y = st.sidebar.checkbox("Auto-scale Plot Y-Axis", value=False)


with st.sidebar.expander("Atomic Standard Settings", expanded=False):
    rb_bias_ns = st.sidebar.number_input("Rubidium Initial Bias (ns)", 0.0, 5000000.0, rb_bias_ns, 50000.0)
    rb_rw_step_ns = st.sidebar.number_input("Rubidium Random Walk Step (ns)", 0.0, 100.0, rb_rw_step_ns, 0.1, format="%.3f")
    rb_noise_ns = st.sidebar.number_input("Rubidium White Noise (ns)", 0.0, 100.0, rb_noise_ns, 0.5)
    rb_aging_ns_s = st.sidebar.number_input("Rubidium Aging Rate (ns/s^2)", 0.0, 1.0, rb_aging_ns_s, 0.01, format="%.3f")

with st.sidebar.expander("GNSS Environment Settings", expanded=False):
    gnss_bias_ns = st.sidebar.number_input("GNSS Bias (ns)", 0.0, 1000.0, gnss_bias_ns, 10.0)
    gnss_meas_ns = st.sidebar.number_input("GNSS Measurement Noise (ns)", 0.0, 500.0, gnss_meas_ns, 10.0)
    gnss_prop_ns = st.sidebar.number_input("GNSS Propagation Delay (ns)", 0.0, 500.0, gnss_prop_ns, 10.0)
    gnss_sat_ns = st.sidebar.number_input("GNSS Sat Clock Error (ns)", 0.0, 500.0, gnss_sat_ns, 5.0)
    dynamic_r_enabled = st.sidebar.checkbox(
        "Constellation-Driven Covariance Adaptation", 
        value=True, 
        help="Scale GNSS measurement noise dynamically based on the ratio of average satellite count to current satellite count."
    )
    sat_scale = st.sidebar.slider(
        "Satellite Visibility Scale", 
        0.1, 1.0, 1.0, 0.05, 
        help="Scale down the number of visible satellites to simulate blockage, foliage, or antenna degradation."
    )

# Convert all manual parameter overrides to seconds for backend simulation
rb_bias = rb_bias_ns * 1e-9
rb_rw_step = rb_rw_step_ns * 1e-9
rb_noise = rb_noise_ns * 1e-9
rb_aging = rb_aging_ns_s * 1e-9

gnss_bias = gnss_bias_ns * 1e-9
gnss_meas = gnss_meas_ns * 1e-9
gnss_prop = gnss_prop_ns * 1e-9
gnss_sat = gnss_sat_ns * 1e-9

# ====================================================
# SYSTEM STATE CALCULATION
# ====================================================
if outage_enabled and outage_start == 0 and outage_end >= duration:
    operating_mode = "Autonomous Holdover (Open-Loop)"
    reference_source = "GNSS Outage (Continuous)"
    system_status = "Outage"
    status_desc = "GNSS receiver signal loss. Autonomous Rubidium oscillator holdover active."
elif outage_enabled:
    operating_mode = "Autonomous Holdover (Open-Loop)"
    reference_source = f"GNSS Outage (t={outage_start}s to t={outage_end}s)"
    system_status = "Holdover"
    status_desc = f"GNSS outage active. Time synchronization maintained via local atomic standard."
else:
    operating_mode = "GNSS-Disciplined (Closed-Loop)"
    reference_source = "GNSS Locked (RINEX)"
    system_status = "Nominal"
    status_desc = "GNSS lock active. Local Rubidium oscillator disciplined via Kalman estimation."

# ====================================================
# CACHED RINEX PARSER
# ====================================================
@st.cache_data
def load_rinex_data(filepath):
    # Safe load, falls back to simulated data if RINEX file is absent
    times, tot, gps, gal, glo, sbas = parse_rinex_file(filepath)
    if len(times) == 0:
        # Fallback simulated dataset spanning 24h
        times = [datetime(2025, 4, 10, h, m, 0) for h in range(24) for m in range(0, 60, 30)]
        np.random.seed(42)
        tot = list(np.random.randint(25, 40, len(times)))
        gps = list(np.random.randint(10, 15, len(times)))
        gal = list(np.random.randint(8, 12, len(times)))
        glo = list(np.random.randint(5, 10, len(times)))
        sbas = list(np.random.randint(1, 4, len(times)))
    return times, tot, gps, gal, glo, sbas

times_analysis, total_count, gps_count, galileo_count, glonass_count, sbas_count = load_rinex_data(RINEX_FILEPATH)
rinex_metadata = parse_rinex_metadata(RINEX_FILEPATH)

rinex_epoch_count = len(times_analysis)
rinex_sample_interval = (
    (times_analysis[1] - times_analysis[0]).total_seconds()
    if len(times_analysis) > 1
    else 0.0
)

# ====================================================
# RUN SIMULATION BACKEND (CACHED)
# ====================================================
@st.cache_data
def get_system_simulation(duration, dt, rb_bias, rb_rw_step, rb_noise, rb_aging,
                          gnss_bias, gnss_sat, gnss_prop, gnss_meas, use_gauss_markov,
                          outage_enabled, outage_start, outage_end,
                          dynamic_r_enabled, sat_scale, total_count_tuple, seed):
    # Scale total counts based on satellite visibility scale
    scaled_sats = [max(0, int(c * sat_scale)) for c in total_count_tuple]

    # 1. Simulate Rubidium clock
    t, rb_err, rw, fw, ag_err = simulate_rubidium_clock(
        duration, dt, rb_bias, rb_rw_step, rb_noise, rb_aging, seed=seed
    )

    # 2. Simulate GNSS error components
    gnss_err, sat_err, prop_err, meas_ns, R_profile = simulate_gnss_time(
        duration, dt, gnss_bias, gnss_sat, gnss_prop, gnss_meas, use_gauss_markov,
        sat_counts=scaled_sats, dynamic_r_enabled=dynamic_r_enabled, seed=seed
    )

    # 3. Run Kalman Filter (2D V2 tracking bias and drift - Default parameters)
    Q_val = rb_rw_step ** 2
    R_val = R_profile if dynamic_r_enabled else gnss_meas ** 2
    kf_out = run_kalman_filter_v2(
        t, rb_err, gnss_err, outage_enabled, outage_start, outage_end, Q_bias=Q_val, Q_drift=1e-22, R_val=R_val
    )
    kal_est, drift_est, kal_var, m_err, innov = kf_out
    diverged_flags = kf_out.diverged_flags
    kalman_gains = kf_out.kalman_gains
    
    return (t, rb_err, rw, fw, ag_err, gnss_err, sat_err, prop_err, meas_ns,
            kal_est, drift_est, kal_var, m_err, innov, diverged_flags, kalman_gains, R_profile)

# Run or retrieve cached simulation
true_time, rubidium_error, rb_rw, rb_freq_offset, rb_aging_error, \
gnss_error, sat_clock_error, propagation_error, measurement_noise, \
kalman_estimate, drift_estimates, kalman_variance, master_error, innovations, diverged_flags, kalman_gains, R_profile = get_system_simulation(
    duration, dt, rb_bias, rb_rw_step, rb_noise, rb_aging,
    gnss_bias, gnss_sat, gnss_prop, gnss_meas, use_gauss_markov,
    outage_enabled, outage_start, outage_end,
    dynamic_r_enabled, sat_scale, tuple(total_count), GLOBAL_RANDOM_SEED
)

# ====================================================
# TABS SYSTEM
# ====================================================
tab_telemetry, tab_constellation, tab_diagnostics, tab_tuning, tab_allan, tab_holdover, tab_playback = st.tabs([
    "Real-Time Telemetry & Diagnostics",
    "Constellation Analysis (RINEX)",
    "Signal Source Characterization",
    "Kalman Tuning",
    "Allan Deviation Analysis",
    "Holdover Verification",
    "Live GNSSDO Playback"
])

# ----------------------------------------------------
# TAB 1: REAL-TIME TELEMETRY & DIAGNOSTICS
# ----------------------------------------------------
with tab_telemetry:
    # Render operational status
    cols_status = st.columns(3)
    with cols_status[0]:
        st.markdown(f"**Operating Mode:** `{operating_mode}`")
    with cols_status[1]:
        st.markdown(f"**Reference Source:** `{reference_source}`")
    with cols_status[2]:
        st.markdown(f"**System Status:** `{system_status}`")
    st.markdown(f"*{status_desc}*")
    
    if np.any(diverged_flags):
        st.error("⚠️ **Kalman Filter Divergence Detected!** Innovations have grown unbounded relative to theoretical covariance bounds. Check filter tuning or signal environment.")
        
    st.markdown("---")
    
    # UTC Master Clock Card
    idx_end = min(len(times_analysis) - 1, int(duration / dt))
    epoch_end_dt = times_analysis[idx_end]
    final_offset = master_error[-1]
    final_utc_str = format_telemetry_time(epoch_end_dt, final_offset)
    
    st.markdown(f"""
    <div class="telemetry-card" style="text-align: center; border-color: #e2e8f0; background-color: #f8fafc; margin-bottom: 20px;">
        <div class="telemetry-header" style="color: #4f46e5; font-size: 1.1rem; border-bottom: none; margin-bottom: 0;">GNSSDO MASTER CLOCK (UTC)</div>
        <div class="telemetry-value-large" style="font-size: 2.8rem; color: #1e1b4b; margin: 10px 0;">{final_utc_str}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Calculate performance scorecard metrics
    if outage_enabled:
        active_indices = (true_time < outage_start) | (true_time >= outage_end)
    else:
        active_indices = np.ones(len(true_time), dtype=bool)
        
    calibrated_gnss_error = gnss_error - gnss_bias
    calibrated_master_error = master_error - gnss_bias
    gnss_rms = np.sqrt(np.mean(calibrated_gnss_error[active_indices]**2)) * 1e9
    rb_rms = np.sqrt(np.mean(rubidium_error[active_indices]**2)) * 1e9
    master_rms = np.sqrt(np.mean(calibrated_master_error[active_indices]**2)) * 1e9
    master_max = np.max(np.abs(calibrated_master_error[active_indices])) * 1e9
    
    passed_assessment = master_max < target_accuracy_ns
    status_text = "PASS" if passed_assessment else "FAIL"
    status_color = "#059669" if passed_assessment else "#dc2626"
    
    st.markdown("### Timing System Performance Scorecard")
    cols_scorecard = st.columns([1, 1, 1, 1, 1.2])
    with cols_scorecard[0]:
        st.markdown(f'<div class="metric-card"><div class="metric-value value-green">{gnss_rms:.1f} ns</div><div class="metric-label">GNSS RMS Error</div></div>', unsafe_allow_html=True)
    with cols_scorecard[1]:
        st.markdown(f'<div class="metric-card"><div class="metric-value value-orange">{rb_rms/1e6:.2f} ms</div><div class="metric-label">Rubidium RMS Error</div></div>', unsafe_allow_html=True)
    with cols_scorecard[2]:
        st.markdown(f'<div class="metric-card"><div class="metric-value value-blue">{master_rms:.2f} ns</div><div class="metric-label">Master RMS Error</div></div>', unsafe_allow_html=True)
    with cols_scorecard[3]:
        st.markdown(f'<div class="metric-card"><div class="metric-value value-purple">{master_max:.2f} ns</div><div class="metric-label">Master Peak Error</div></div>', unsafe_allow_html=True)
    with cols_scorecard[4]:
        st.markdown(
            f'<div class="metric-card" style="background-color: {status_color}10; border-color: {status_color}50;">'
            f'<div class="metric-value" style="color: {status_color};">{status_text}</div>'
            f'<div class="metric-label">Target: {target_accuracy_ns:.0f} ns</div></div>',
            unsafe_allow_html=True
        )
        
    st.markdown("---")
    
    st.markdown("### Disciplined Master Clock Phase Error")
    fig_m = plot_disciplined_clock_v2(true_time, calibrated_gnss_error, calibrated_master_error, outage_enabled, outage_start, outage_end, auto_scale=auto_scale_y)
    st.pyplot(fig_m)
    plt.close(fig_m)
    
    st.markdown("---")
    st.markdown("### Performance Metrics")
    
    # Indices logic
    if outage_enabled:
        active_indices = (true_time < outage_start) | (true_time >= outage_end)
        outage_indices = (true_time >= outage_start) & (true_time < outage_end)
    else:
        active_indices = np.ones(len(true_time), dtype=bool)
        outage_indices = np.zeros(len(true_time), dtype=bool)
        
    col_stats1, col_stats2 = st.columns(2)
    
    with col_stats1:
        st.markdown("#### Active Tracking Performance")
        tracking_data = {
            "System Parameter": [
                "Initial Clock Offset (Atomic Bias)",
                "GNSS Receiver Fixed Bias",
                "GNSS Signal Measurement Noise (1-Sigma)",
                "Disciplined Clock Accuracy (Active)",
                "Average Synchronization Error"
            ],
            "Value": [
                f"{rb_bias*1e6:.2f} µs",
                f"{gnss_bias*1e9:.1f} ns",
                f"{np.std(gnss_error[active_indices])*1e9:.2f} ns",
                f"{np.std(master_error[active_indices])*1e9:.2f} ns",
                f"{np.mean(master_error[active_indices])*1e9:.2f} ns"
            ]
        }
        st.table(tracking_data)
        
    with col_stats2:
        st.markdown("#### Outage Holdover Performance")
        if outage_enabled:
            outage_duration = outage_end - outage_start
            accumulated_drift = (master_error[outage_indices][-1] - master_error[outage_indices][0]) * 1e9
            final_sigma = np.sqrt(kalman_variance[outage_indices][-1])
            
            outage_data = {
                "Holdover Metric": [
                    "Duration of GNSS Outage",
                    "Accumulated Drift during Outage",
                    "Estimated Max Time Error (3-Sigma Confidence)"
                ],
                "Value": [
                    f"{outage_duration} seconds",
                    f"{accumulated_drift:.2f} ns",
                    f"{3 * final_sigma * 1e9:.2f} ns ({3 * final_sigma * 1e6:.4f} µs)"
                ]
            }
            st.table(outage_data)
        else:
            st.write("GNSS receiver operating under continuous lock. Holdover statistics inactive.")
            
    st.markdown("---")
    st.markdown("### Telemetry Observation Log")
    sim_data = pd.DataFrame({
        "Time Epoch (s)": true_time,
        "Oscillator Error (ns)": rubidium_error * 1e9,
        "Raw GNSS Error (ns)": gnss_error * 1e9,
        "Disciplined System Error (ns)": master_error * 1e9,
        "Filter Estimate (ns)": kalman_estimate * 1e9,
        "Frequency Drift Estimate (ns/s)": drift_estimates * 1e9
    })
    st.dataframe(sim_data, width="stretch")

# ----------------------------------------------------
# TAB 2: CONSTELLATION ANALYSIS (RINEX)
# ----------------------------------------------------
with tab_constellation:
    st.markdown("### Satellite Status (RINEX Data: Station AB04)")
    st.caption(
        f"Historical source: {RINEX_FILEPATH} | "
        f"{rinex_epoch_count:,} epochs | "
        f"{rinex_sample_interval:.0f}s sample interval"
    )
    
    # Render RINEX Header Metadata
    with st.expander("RINEX Header Metadata & Station Details", expanded=True):
        col_meta1, col_meta2, col_meta3 = st.columns(3)
        with col_meta1:
            st.markdown(f"**Station / Marker Name:** `{rinex_metadata.get('marker_name', 'N/A')}`")
            st.markdown(f"**RINEX Version & Type:** `{rinex_metadata.get('version', 'N/A')} ({rinex_metadata.get('type', 'N/A')})`")
            st.markdown(f"**First Observation Time:** `{rinex_metadata.get('first_obs_time', 'N/A')}`")
        with col_meta2:
            st.markdown(f"**Observer / Agency:** `{rinex_metadata.get('observer_agency', 'N/A')}`")
            st.markdown(f"**Receiver Model:** `{rinex_metadata.get('receiver_info', 'N/A')}`")
            st.markdown(f"**Antenna Model:** `{rinex_metadata.get('antenna_info', 'N/A')}`")
        with col_meta3:
            st.markdown(f"**Approx Position (XYZ):** `{rinex_metadata.get('approx_position', 'N/A')}`")
            st.markdown(f"**Leap Seconds:** `{rinex_metadata.get('leap_seconds', 'N/A')} seconds`")
            obs_types_str = ", ".join(rinex_metadata.get('obs_types', []))
            st.markdown(f"**Observed Bands:** `{obs_types_str}`")
            
    st.markdown("---")
    
    avg_total = f"{np.mean(total_count):.1f}"
    avg_gps = f"{np.mean(gps_count):.1f}"
    avg_gal = f"{np.mean(galileo_count):.1f}"
    avg_glo = f"{np.mean(glonass_count):.1f}"
    avg_sbas = f"{np.mean(sbas_count):.1f}"
    
    cols_sat = st.columns(5)
    with cols_sat[0]:
        st.markdown(f'<div class="metric-card"><div class="metric-value value-blue">{avg_total}</div><div class="metric-label">Avg Visible Satellites</div></div>', unsafe_allow_html=True)
    with cols_sat[1]:
        st.markdown(f'<div class="metric-card"><div class="metric-value" style="color: #059669;">{avg_gps}</div><div class="metric-label">GPS</div></div>', unsafe_allow_html=True)
    with cols_sat[2]:
        st.markdown(f'<div class="metric-card"><div class="metric-value" style="color: #ea580c;">{avg_gal}</div><div class="metric-label">Galileo</div></div>', unsafe_allow_html=True)
    with cols_sat[3]:
        st.markdown(f'<div class="metric-card"><div class="metric-value" style="color: #dc2626;">{avg_glo}</div><div class="metric-label">GLONASS</div></div>', unsafe_allow_html=True)
    with cols_sat[4]:
        st.markdown(f'<div class="metric-card"><div class="metric-value" style="color: #7c3aed;">{avg_sbas}</div><div class="metric-label">SBAS</div></div>', unsafe_allow_html=True)
        
    # RINEX Visibility Plot
    fig_sat = plot_satellite_visibility(times_analysis, total_count, gps_count, galileo_count, glonass_count, sbas_count)
    st.pyplot(fig_sat)
    plt.close(fig_sat)
    
    st.markdown("---")
    st.markdown("### Raw Observation Epoch Data")
    rinex_df = pd.DataFrame({
        "Epoch Time (UTC)": times_analysis,
        "Total Visible Satellites": total_count,
        "GPS": gps_count,
        "Galileo": galileo_count,
        "GLONASS": glonass_count,
        "SBAS": sbas_count
    })
    st.dataframe(rinex_df, width="stretch")

# ----------------------------------------------------
# TAB 3: SIGNAL SOURCE CHARACTERIZATION
# ----------------------------------------------------
with tab_diagnostics:
    st.markdown("### Signal Source Characterization")
    
    col_src1, col_src2 = st.columns(2)
    with col_src1:
        st.markdown("#### Rubidium Standard Phase Error")
        fig_rb = plot_rubidium_clock(true_time, rubidium_error, rb_aging_error, rb_aging_ns_s)
        st.pyplot(fig_rb)
        plt.close(fig_rb)
        
        with st.expander("Show Separate Components & Distribution", expanded=True):
            fig_nobias = plot_error_without_bias(true_time, rubidium_error, rb_bias)
            st.pyplot(fig_nobias)
            plt.close(fig_nobias)
            
            fig_rw = plot_rubidium_random_walk(true_time, rb_rw)
            st.pyplot(fig_rw)
            plt.close(fig_rw)
            
            fig_fw = plot_rubidium_frequency_wander(true_time, rb_freq_offset)
            st.pyplot(fig_fw)
            plt.close(fig_fw)
            
            fig_aging_comp = plot_rubidium_aging(true_time, rb_aging_error)
            st.pyplot(fig_aging_comp)
            plt.close(fig_aging_comp)
            
            fig_dist = plot_rubidium_distribution(rubidium_error - rb_bias)
            st.pyplot(fig_dist)
            plt.close(fig_dist)
            
        with st.expander("Show Raw Oscillator Statistics", expanded=True):
            rb_start_us = rubidium_error[0] * 1e6
            rb_mid_us = rubidium_error[len(rubidium_error)//2] * 1e6
            rb_end_us = rubidium_error[-1] * 1e6
            rb_mean_us = np.mean(rubidium_error) * 1e6
            rb_std_us = np.std(rubidium_error) * 1e6
            rb_wander_us = (rubidium_error[-1] - rubidium_error[0]) * 1e6
            
            rb_stats = {
                "Metric": [
                    "Start Error",
                    "Middle Error",
                    "End Error",
                    "Mean Error",
                    "Standard Deviation",
                    "Total Wander"
                ],
                "Value (µs)": [
                    f"{rb_start_us:.4f}",
                    f"{rb_mid_us:.4f}",
                    f"{rb_end_us:.4f}",
                    f"{rb_mean_us:.4f}",
                    f"{rb_std_us:.4f}",
                    f"{rb_wander_us:.4f}"
                ]
            }
            st.table(rb_stats)
        
    with col_src2:
        st.markdown("#### GNSS Receiver Time Error Components")
        fig_gnss = plot_gnss_time_components(true_time, sat_clock_error, propagation_error, measurement_noise)
        st.pyplot(fig_gnss)
        plt.close(fig_gnss)
        
        with st.expander("Show Separate Components & Full Error", expanded=True):
            fig_full_gnss = plot_full_gnss_error(true_time, gnss_error, gnss_bias * 1e9)
            st.pyplot(fig_full_gnss)
            plt.close(fig_full_gnss)
            
            fig_breakdown = plot_gnss_component_breakdown(true_time, gnss_error, sat_clock_error, propagation_error, measurement_noise, zoom_limit=500)
            st.pyplot(fig_breakdown)
            plt.close(fig_breakdown)
            
        with st.expander("Show Raw GNSS Statistics", expanded=True):
            gnss_mean_ns = np.mean(gnss_error) * 1e9
            gnss_std_ns = np.std(gnss_error) * 1e9
            
            gnss_stats = {
                "Metric": [
                    "Mean Error",
                    "Standard Deviation (Total Noise)"
                ],
                "Value (ns)": [
                    f"{gnss_mean_ns:.3f}",
                    f"{gnss_std_ns:.3f}"
                ]
            }
            st.table(gnss_stats)

# ----------------------------------------------------
# TAB 4: KALMAN TUNING
# ----------------------------------------------------
with tab_tuning:
    st.markdown("### Kalman Filter Tuning")
    st.markdown("Adjust the process noise covariance (Q) and measurement noise covariance (R) to sweep filter performance in real-time. Find the Goldilocks Region where errors are minimized.")
    
    col_t1, col_t2, col_t3 = st.columns(3)
    with col_t1:
        q_bias_exp = st.slider(
            "Bias Process Noise Exponent: log10(Q_bias)",
            min_value=-24,
            max_value=-10,
            value=-20,
            step=1,
            help="Q_bias represents the uncertainty of the phase state model. Smaller values model a more stable phase standard."
        )
        Q_bias_t = 10.0 ** q_bias_exp
        st.markdown(f"**Q_bias Value:** `{Q_bias_t:.1e} s²`")
        
    with col_t2:
        q_drift_exp = st.slider(
            "Drift Process Noise Exponent: log10(Q_drift)",
            min_value=-26,
            max_value=-16,
            value=-22,
            step=1,
            help="Q_drift represents the uncertainty of the frequency drift model. Smaller values model a more stable frequency drift rate."
        )
        Q_drift_t = 10.0 ** q_drift_exp
        st.markdown(f"**Q_drift Value:** `{Q_drift_t:.1e} s²/s²`")
        
    with col_t3:
        r_sigma_ns = st.slider(
            "Measurement Noise Sigma: σ_gnss (ns)",
            min_value=5.0,
            max_value=300.0,
            value=50.0,
            step=5.0,
            help="R = (σ_gnss * 1e-9)². Smaller values instruct the filter to place more trust in raw GNSS timing."
        )
        R_t = (r_sigma_ns * 1e-9) ** 2
        st.markdown(f"**R Value:** `{R_t:.2e} s²` (σ = {r_sigma_ns} ns)")
        
    st.markdown("---")
    
    # Calculate R_val_t for the tuned filter
    if dynamic_r_enabled and gnss_meas > 0:
        R_val_t = R_profile * ((r_sigma_ns * 1e-9) / gnss_meas) ** 2
        R_plot_t = R_val_t
    else:
        R_val_t = R_t
        R_plot_t = np.full(len(true_time), R_t)
        
    # Run the tuned Kalman filter
    kf_out_t = run_kalman_filter_v2(
        true_time, rubidium_error, gnss_error, outage_enabled, outage_start, outage_end,
        Q_bias=Q_bias_t, Q_drift=Q_drift_t, R_val=R_val_t
    )
    bias_est_t, drift_est_t, var_t, master_error_t, innovations_t = kf_out_t
    kalman_gains_t = kf_out_t.kalman_gains
    
    # Calibrate tuned master error
    calibrated_master_error_t = master_error_t - gnss_bias
    
    # Calculate local metrics
    if outage_enabled:
        t_active_indices = (true_time < outage_start) | (true_time >= outage_end)
        t_outage_indices = (true_time >= outage_start) & (true_time < outage_end)
    else:
        t_active_indices = np.ones(len(true_time), dtype=bool)
        t_outage_indices = np.zeros(len(true_time), dtype=bool)
        
    t_std_active = np.std(calibrated_master_error_t[t_active_indices]) * 1e9
    t_mean_active = np.mean(calibrated_master_error_t[t_active_indices]) * 1e9
    
    t_std_overall = np.std(calibrated_master_error_t) * 1e9
    
    # Error right at the end of the outage (t = OUTAGE_END s)
    idx_outage_end = int(outage_end / dt) if outage_enabled else 0
    t_peak_err = calibrated_master_error_t[idx_outage_end] * 1e9 if outage_enabled else 0.0
    t_final_err = calibrated_master_error_t[-1] * 1e9
    
    if np.any(kf_out_t.diverged_flags):
        st.error("⚠️ **Tuned Kalman Filter Divergence Detected!** Innovations have grown unbounded. Adjust process noise or measurement noise to stabilize.")
        
    # Performance summary metrics cards
    col_tm1, col_tm2, col_tm3 = st.columns(3)
    with col_tm1:
        st.markdown(f'<div class="metric-card"><div class="metric-value value-blue">{t_std_overall:.2f} ns</div><div class="metric-label">Master Error Std (Overall)</div></div>', unsafe_allow_html=True)
    with col_tm2:
        if outage_enabled:
            st.markdown(f'<div class="metric-card"><div class="metric-value value-orange">{t_peak_err:.2f} ns</div><div class="metric-label">Peak Outage Error (t={outage_end}s)</div></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="metric-card"><div class="metric-value value-green">N/A</div><div class="metric-label">Peak Outage Error (Outage Inactive)</div></div>', unsafe_allow_html=True)
    with col_tm3:
        st.markdown(f'<div class="metric-card"><div class="metric-value value-purple">{t_final_err:.2f} ns</div><div class="metric-label">Final Error (t={duration}s)</div></div>', unsafe_allow_html=True)
            
    st.markdown("---")
    st.markdown("### Tuned Filter Diagnostics")
    
    # Renders the live plots
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.markdown("#### Custom Kalman Filter Diagnostics")
        fig_kf_t = plot_kalman_diagnostics_v2(
            true_time, rubidium_error, bias_est_t, var_t, gnss_bias,
            outage_enabled, outage_start, outage_end
        )
        st.pyplot(fig_kf_t)
        plt.close(fig_kf_t)
        
    with col_p2:
        st.markdown("#### Custom Constellation-Driven Covariance Adaptation & Kalman Gain")
        fig_dyn_r_t = plot_dynamic_r_diagnostics(
            true_time,
            np.array([max(0, int(total_count[i % len(total_count)] * sat_scale)) for i in range(len(true_time))]),
            np.sqrt(R_plot_t),
            kalman_gains_t,
            outage_enabled,
            outage_start,
            outage_end
        )
        st.pyplot(fig_dyn_r_t)
        plt.close(fig_dyn_r_t)
        
    st.markdown("---")
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.markdown("#### Custom Estimated Frequency Drift")
        fig_drift_t = plot_estimated_drift(true_time, drift_est_t)
        st.pyplot(fig_drift_t)
        plt.close(fig_drift_t)
        
    with col_d2:
        st.markdown("#### Custom Kalman Filter Measurement Innovation")
        fig_innov_t = plot_kalman_innovation(true_time, innovations_t, outage_enabled, outage_start, outage_end)
        st.pyplot(fig_innov_t)
        plt.close(fig_innov_t)

# ----------------------------------------------------
# TAB 5: ALLAN DEVIATION ANALYSIS
# ----------------------------------------------------
with tab_allan:
    st.markdown("### Allan Deviation Analysis")
    
    # Calculate Allan deviation using current simulation errors
    # Note: rate = 1 / dt, since dt is the time step of the simulation.
    rate_val = 1.0 / dt
    
    with st.spinner("Calculating Allan Deviation..."):
        taus_rb, adev_rb = calculate_allan_deviation(rubidium_error, rate=rate_val)
        taus_gnss, adev_gnss = calculate_allan_deviation(gnss_error, rate=rate_val)
        taus_master, adev_master = calculate_allan_deviation(master_error, rate=rate_val)
        
        # Render the comparison plot
        with allan_plot_lock:
            fig_ad = plot_allan_deviation_comparison(
                taus_rb, adev_rb,
                taus_gnss, adev_gnss,
                taus_master, adev_master
            )
            st.pyplot(fig_ad)
            plt.close(fig_ad)
        
        st.markdown("---")
        st.markdown("### Understanding Allan Deviation & Clock Stability")
        
        col_exp1, col_exp2 = st.columns(2)
        with col_exp1:
            st.markdown("""
            <div style="background-color: #f8fafc; border-left: 5px solid #4f46e5; padding: 15px; border-radius: 8px; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.02); margin-top: 10px;">
                <h4 style="margin-top: 0; color: #1e293b;">Short-Term Stability (Small &tau;)</h4>
                <p style="font-size: 0.9rem; color: #475569; margin-bottom: 8px;">
                    Reflected on the <b>left side</b> of the plot (e.g., &tau; = 1 s, 2 s, 5 s):
                </p>
                <ul style="font-size: 0.85rem; color: #334155; margin-bottom: 0; padding-left: 20px;">
                    <li><b>Normally:</b> <code>Rubidium &lt; GNSS</code> (lower Allan deviation is better).</li>
                    <li>The local Rubidium atomic standard is highly stable in the short term, whereas raw GNSS measurements suffer from atmospheric jitter and receiver noise.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
        with col_exp2:
            st.markdown("""
            <div style="background-color: #f8fafc; border-left: 5px solid #059669; padding: 15px; border-radius: 8px; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.02); margin-top: 10px;">
                <h4 style="margin-top: 0; color: #1e293b;">Long-Term Stability (Large &tau;)</h4>
                <p style="font-size: 0.9rem; color: #475569; margin-bottom: 8px;">
                    Reflected on the <b>right side</b> of the plot (e.g., &tau; = 1000 s, 2000 s):
                </p>
                <ul style="font-size: 0.85rem; color: #334155; margin-bottom: 0; padding-left: 20px;">
                    <li><b>Normally:</b> <code>GNSS &lt; Rubidium</code> (due to oscillator drift).</li>
                    <li>The GNSS reference is tied directly to GPS/Galileo system time (kept stable via ground station atomic clocks), preventing drift, whereas the Rubidium standard will drift continuously if undisciplined.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)


# (Helpers moved to the top of the file)

def plot_live_playback_errors(history_t, history_gnss, history_master, history_rb):
    """
    Plots the last 60 epochs of real-time clock error history.
    Uses twinx to show GNSS/Disciplined error on the left y-axis, and Rubidium error on the right.
    """
    fig, ax = plt.subplots(figsize=(8, 2.7))
    show_points = 60
    t_plot = np.array(history_t[-show_points:])
    t_plot = t_plot - t_plot[0]
    
    gnss_plot = np.array(history_gnss[-show_points:]) * 1e9
    master_plot = np.array(history_master[-show_points:]) * 1e9
    rb_plot = np.array(history_rb[-show_points:]) * 1e9
    
    # Left Axis: True Time Reference, GNSS Jitter, and Disciplined clock
    ax.axhline(0, label="True Time Reference", color="#64748b", linestyle="--", linewidth=1.2)
    ax.plot(t_plot, gnss_plot, label="GNSS Jitter", color="#059669", alpha=0.35, linestyle=":", marker=".")
    ax.plot(t_plot, master_plot, label="Disciplined Master Clock", color="#4f46e5", linewidth=2.0)
    ax.set_ylabel("GNSS / Disciplined Error (ns)", color="#334155", fontsize=9)
    ax.tick_params(colors='#334155', labelsize=8)
    
    # Right Axis: Raw Rubidium Error
    ax2 = ax.twinx()
    ax2.plot(t_plot, rb_plot, label="Raw Rubidium Error", color="#ea580c", linewidth=1.5, alpha=0.8)
    ax2.set_ylabel("Raw Rubidium Error (ns)", color="#ea580c", fontsize=9)
    ax2.tick_params(colors='#ea580c', labelsize=8)
    
    # Combine legends
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper right", framealpha=0.8, labelcolor="#334155", fontsize=8)
    
    ax.set_xlabel("Elapsed Time in Window (s)", color="#334155", fontsize=9)
    ax.grid(True, linestyle=":", color="#cbd5e1")
    
    ax.patch.set_facecolor('#ffffff')
    fig.patch.set_facecolor('#f8fafc')
    plt.tight_layout()
    return fig

def plot_live_playback_times(history_t, history_gnss, history_master, history_rb, times_analysis, k):
    """
    Plots the absolute UTC times of all four sources.
    Uses twinx to show True/GNSS/Disciplined time on the left y-axis, and Rubidium time on the right.
    """
    fig, ax = plt.subplots(figsize=(8, 2.7))
    show_points = 60
    t_plot = np.array(history_t[-show_points:])
    t_plot_relative = t_plot - t_plot[0]
    
    # Get current nominal UTC time baseline
    epoch_dt = times_analysis[k % len(times_analysis)]
    
    # Absolute errors in seconds
    gnss_err = np.array(history_gnss[-show_points:])
    master_err = np.array(history_master[-show_points:])
    rb_err = np.array(history_rb[-show_points:])
    
    # Left Axis: True UTC Time, GNSS Time, and Disciplined clock
    ax.plot(t_plot_relative, np.zeros_like(t_plot), label="True UTC Time", color="#64748b", linestyle="--", linewidth=1.2)
    ax.plot(t_plot_relative, gnss_err, label="GNSS Time", color="#059669", alpha=0.35, linestyle=":", marker=".")
    ax.plot(t_plot_relative, master_err, label="Disciplined Time", color="#4f46e5", linewidth=2.0)
    ax.set_ylabel("True / GNSS / Disciplined UTC Time", color="#334155", fontsize=9)
    ax.tick_params(colors='#334155', labelsize=8)
    
    # Formatting left axis as absolute time stamps
    from matplotlib.ticker import FuncFormatter
    def left_time_formatter(y_val, pos):
        return format_telemetry_time(epoch_dt, y_val, time_only=True)
    ax.yaxis.set_major_formatter(FuncFormatter(left_time_formatter))
    
    # Right Axis: Rubidium Time
    ax2 = ax.twinx()
    ax2.plot(t_plot_relative, rb_err, label="Rubidium Time", color="#ea580c", linewidth=1.5, alpha=0.8)
    ax2.set_ylabel("Rubidium UTC Time (Right Axis)", color="#ea580c", fontsize=9)
    ax2.tick_params(colors='#ea580c', labelsize=8)
    
    # Formatting right axis as absolute time stamps
    def right_time_formatter(y_val, pos):
        return format_telemetry_time(epoch_dt, y_val, time_only=True)
    ax2.yaxis.set_major_formatter(FuncFormatter(right_time_formatter))
    
    # Combine legends
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper right", framealpha=0.8, labelcolor="#334155", fontsize=8)
    
    ax.set_xlabel("Elapsed Time in Window (s)", color="#334155", fontsize=9)
    ax.grid(True, linestyle=":", color="#cbd5e1")
    
    ax.patch.set_facecolor('#ffffff')
    fig.patch.set_facecolor('#f8fafc')
    plt.tight_layout()
    return fig

# ----------------------------------------------------
# TAB 6: HOLDOVER VERIFICATION
# ----------------------------------------------------
with tab_holdover:
    st.subheader("Holdover Verification & Stress Testing")
    st.markdown("""
    This panel performs automated stress testing of the GNSSDO system under various GNSS outage conditions.
    By evaluating clock drift during outages and tracking time recovery post-outage, this suite validates 
    the stability of the Rubidium standard and the reconvergence rate of the Kalman filter.
    
    The test runs three scenarios in parallel:
    * **5-minute Outage** (300 s) — Short transient loss of GNSS signal (e.g. canopy blockage or multi-path interference).
    * **30-minute Outage** (1800 s) — Medium-duration loss of signal.
    * **1-hour Outage** (3600 s) — Long-duration loss of signal (e.g. system maintenance or server downtime).
    """)
    
    # Let the user trigger the simulation
    if st.button("Run Holdover Verification Suite", type="primary", use_container_width=True):
        with st.spinner("Executing holdover stress tests..."):
            # We will run the three profiles
            profiles = [
                ("5-minute Outage", 300.0, "#10b981"),
                ("30-minute Outage", 1800.0, "#f59e0b"),
                ("1-hour Outage", 3600.0, "#ef4444")
            ]
            
            results = []
            plot_data = {}
            
            # Use fixed 7200 s duration and dt=1.0 s for high-resolution recovery timing
            total_dur = 7200
            dt_s = 1.0
            outage_start = 1800.0
            
            # Prepare the satellites list
            scaled_sats = [max(0, int(c * sat_scale)) for c in total_count]
            
            for name, outage_dur_s, color in profiles:
                outage_end = outage_start + outage_dur_s
                
                # 1. Simulate Rubidium clock step (using GLOBAL_RANDOM_SEED for consistency across comparison runs)
                t_p, rb_err_p, rw_p, fw_p, ag_err_p = simulate_rubidium_clock(
                    total_dur, dt_s, rb_bias, rb_rw_step, rb_noise, rb_aging, seed=GLOBAL_RANDOM_SEED
                )
                
                # 2. Simulate GNSS error step
                gnss_err_p, sat_err_p, prop_err_p, meas_ns_p, R_profile_p = simulate_gnss_time(
                    total_dur, dt_s, gnss_bias, gnss_sat, gnss_prop, gnss_meas, use_gauss_markov,
                    sat_counts=scaled_sats, dynamic_r_enabled=dynamic_r_enabled, seed=GLOBAL_RANDOM_SEED
                )
                
                # 3. Kalman Filter
                Q_val = rb_rw_step ** 2
                R_val = R_profile_p if dynamic_r_enabled else gnss_meas ** 2
                
                kf_out_p = run_kalman_filter_v2(
                    t_p, rb_err_p, gnss_err_p,
                    outage_enabled=True, outage_start=outage_start, outage_end=outage_end,
                    Q_bias=Q_val, Q_drift=1e-22, R_val=R_val
                )
                kal_est_p, drift_est_p, kal_var_p, m_err_p, innov_p = kf_out_p
                
                # Calibrate master error by subtracting static receiver bias
                calibrated_master_error_p = m_err_p - gnss_bias
                
                # Calculate metrics
                outage_mask = (t_p >= outage_start) & (t_p < outage_end)
                peak_err_ns = np.max(np.abs(calibrated_master_error_p[outage_mask])) * 1e9
                rms_err_ns = np.sqrt(np.mean(calibrated_master_error_p[outage_mask]**2)) * 1e9
                final_err_ns = np.abs(calibrated_master_error_p[-1]) * 1e9
                
                # Recovery time using calibrated master error
                rec_time = calculate_recovery_time(t_p, calibrated_master_error_p, outage_end, threshold_ns=target_accuracy_ns, dt=dt_s, window_seconds=30.0)
                
                # Mean frequency drift rate (ps/s) during holdover
                idx_start = int(round(outage_start / dt_s))
                idx_end = int(round(outage_end / dt_s))
                phase_start = calibrated_master_error_p[idx_start]
                phase_end = calibrated_master_error_p[idx_end]
                mean_drift_s_s = (phase_end - phase_start) / outage_dur_s
                mean_drift_ps_s = mean_drift_s_s * 1e12
                
                results.append({
                    "Profile": name,
                    "Outage Duration": f"{int(outage_dur_s/60)} min",
                    "Peak Time Error": f"{peak_err_ns/1e3:.2f} µs" if peak_err_ns >= 1000 else f"{peak_err_ns:.1f} ns",
                    "RMS Time Error": f"{rms_err_ns/1e3:.2f} µs" if rms_err_ns >= 1000 else f"{rms_err_ns:.1f} ns",
                    "Final Time Error": f"{final_err_ns/1e3:.2f} µs" if final_err_ns >= 1000 else f"{final_err_ns:.1f} ns",
                    "Target Limit": f"{target_accuracy_ns:.0f} ns",
                    "Recovery Time": f"{rec_time:.1f} s" if rec_time is not None else "Not achieved",
                    "Mean Drift Rate (ps/s)": f"{mean_drift_ps_s:+.2f} ps/s"
                })
                
                plot_data[name] = {
                    "time": t_p,
                    "master_error": calibrated_master_error_p,
                    "color": color,
                    "outage_end": outage_end
                }
                
            # Display summary table
            st.markdown("### Stress Test Performance Summary")
            st.table(pd.DataFrame(results))
            
            # Plot comparison curves
            st.markdown("### Holdover Error Growth & Reconvergence Curves")
            fig_h, ax_h = plt.subplots(figsize=(14, 5.5))
            
            # Zoom in on the outage and post-outage period (t = 1500 s to 6500 s)
            zoom_mask = (t_p >= 1500) & (t_p <= 6500)
            t_zoom = t_p[zoom_mask]
            
            for name in plot_data:
                err_zoom = plot_data[name]["master_error"][zoom_mask] * 1e9
                ax_h.plot(t_zoom, err_zoom, label=name, color=plot_data[name]["color"], linewidth=1.8)
                
                # Draw vertical lines for outage end
                ax_h.axvline(plot_data[name]["outage_end"], color=plot_data[name]["color"], linestyle=":", alpha=0.7)
                
            # Draw common outage start
            ax_h.axvline(outage_start, color="#4b5563", linestyle="--", label="Outage Inject (t=1800s)")
            
            # Draw threshold bounds
            ax_h.axhline(target_accuracy_ns, color="#ef4444", linestyle="-.", alpha=0.5, label=f"Accuracy Target (±{target_accuracy_ns} ns)")
            ax_h.axhline(-target_accuracy_ns, color="#ef4444", linestyle="-.", alpha=0.5)
            
            ax_h.set_ylabel("Calibrated Phase Error (ns)", color="#334155")
            ax_h.set_xlabel("Simulation Elapsed Time (s)", color="#334155")
            ax_h.grid(True, linestyle=":", color="#cbd5e1")
            ax_h.legend(loc="upper left", framealpha=0.9, labelcolor="#334155")
            ax_h.patch.set_facecolor('#ffffff')
            fig_h.patch.set_facecolor('#f8fafc')
            ax_h.tick_params(colors='#334155')
            
            plt.tight_layout()
            st.pyplot(fig_h)
            plt.close(fig_h)

# ----------------------------------------------------
# TAB 7: LIVE GNSSDO PLAYBACK
# ----------------------------------------------------
with tab_playback:
    st.subheader("Live GNSSDO Simulation")
    st.markdown("This panel runs a live epoch-by-epoch simulation of the GNSSDO system. It is driven by the parsed RINEX satellite counts to dynamically adjust the Kalman filter's measurement noise covariance $R$ and operating modes in real-time.")
    
    @st.fragment
    def run_playback_panel():
        def safe_rerun():
            try:
                st.rerun(scope="fragment")
            except Exception:
                st.rerun()

        current_params = (rb_bias, rb_rw_step, rb_noise, rb_aging, gnss_bias, gnss_sat, gnss_prop, gnss_meas, use_gauss_markov, dt, sat_scale)
        if "playback_last_params" not in st.session_state or st.session_state.playback_last_params != current_params:
            st.session_state.playback_last_params = current_params
            st.session_state.playback_epoch = 0
            st.session_state.playback_slider = 0
            st.session_state.playback_should_increment = False
            st.session_state.playback_history_t = []
            st.session_state.playback_history_rb = []
            st.session_state.playback_history_gnss = []
            st.session_state.playback_history_master = []
            st.session_state.playback_history_utc = []
            st.session_state.playback_history_sats = []
            st.session_state.playback_history_mode = []
            st.session_state.playback_history_bias_est = []
            st.session_state.playback_history_drift_est = []
            st.session_state.playback_history_sigma = []
            st.session_state.playback_history_R = []
            st.session_state.playback_history_K = []
            st.session_state.playback_history_true_time = []
            st.session_state.playback_history_gnss_time = []
            st.session_state.playback_history_rubidium_time = []
            st.session_state.playback_history_disciplined_time = []
            st.session_state.playback_history_kf_x = []
            st.session_state.playback_history_kf_P = []
            st.session_state.playback_history_rb_phase_rw = []
            st.session_state.playback_history_rb_freq_offset = []
            st.session_state.playback_history_gnss_prop_err = []
            st.session_state.playback_kf_x = np.array([[rb_bias - gnss_bias], [0.0]])
            st.session_state.playback_kf_P = np.array([[1e-6, 0.0], [0.0, 1e-12]])
            st.session_state.playback_rb_freq_offset = 0.0
            st.session_state.playback_rb_phase_rw = 0.0
            st.session_state.playback_gnss_prop_err = 0.0
            st.session_state.playback_playing = False
            st.session_state.playback_receiver = SimulatedReceiver(
                receiver_bias=gnss_bias,
                sat_clock_std=gnss_sat,
                prop_std=gnss_prop,
                meas_std=gnss_meas,
                use_gauss_markov=use_gauss_markov,
                total_counts=total_count,
                gps_counts=gps_count,
                galileo_counts=galileo_count,
                glonass_counts=glonass_count,
                sbas_counts=sbas_count,
                dynamic_r_enabled=True,
                sat_scale=sat_scale,
                seed=GLOBAL_RANDOM_SEED
            )

        # Initialization fallback
        if "playback_epoch" not in st.session_state:
            st.session_state.playback_epoch = 0
            st.session_state.playback_slider = 0
            st.session_state.playback_should_increment = False
            st.session_state.playback_history_t = []
            st.session_state.playback_history_rb = []
            st.session_state.playback_history_gnss = []
            st.session_state.playback_history_master = []
            st.session_state.playback_history_utc = []
            st.session_state.playback_history_sats = []
            st.session_state.playback_history_mode = []
            st.session_state.playback_history_bias_est = []
            st.session_state.playback_history_drift_est = []
            st.session_state.playback_history_sigma = []
            st.session_state.playback_history_R = []
            st.session_state.playback_history_K = []
            st.session_state.playback_history_true_time = []
            st.session_state.playback_history_gnss_time = []
            st.session_state.playback_history_rubidium_time = []
            st.session_state.playback_history_disciplined_time = []
            st.session_state.playback_history_kf_x = []
            st.session_state.playback_history_kf_P = []
            st.session_state.playback_history_rb_phase_rw = []
            st.session_state.playback_history_rb_freq_offset = []
            st.session_state.playback_history_gnss_prop_err = []
            st.session_state.playback_kf_x = np.array([[rb_bias - gnss_bias], [0.0]])
            st.session_state.playback_kf_P = np.array([[1e-6, 0.0], [0.0, 1e-12]])
            st.session_state.playback_rb_freq_offset = 0.0
            st.session_state.playback_rb_phase_rw = 0.0
            st.session_state.playback_gnss_prop_err = 0.0
            st.session_state.playback_playing = False
            st.session_state.playback_receiver = SimulatedReceiver(
                receiver_bias=gnss_bias,
                sat_clock_std=gnss_sat,
                prop_std=gnss_prop,
                meas_std=gnss_meas,
                use_gauss_markov=use_gauss_markov,
                total_counts=total_count,
                gps_counts=gps_count,
                galileo_counts=galileo_count,
                glonass_counts=glonass_count,
                sbas_counts=sbas_count,
                dynamic_r_enabled=True,
                sat_scale=sat_scale,
                seed=GLOBAL_RANDOM_SEED
            )


        # Ensure all required history keys are present in session state
        new_vars = [
            "playback_history_utc", "playback_history_sats", "playback_history_mode",
            "playback_history_bias_est", "playback_history_drift_est",
            "playback_history_sigma", "playback_history_R", "playback_history_K",
            "playback_history_true_time", "playback_history_gnss_time",
            "playback_history_rubidium_time", "playback_history_disciplined_time",
            "playback_history_kf_x", "playback_history_kf_P",
            "playback_history_rb_phase_rw", "playback_history_rb_freq_offset",
            "playback_history_gnss_prop_err"
        ]
        for var in new_vars:
            if var not in st.session_state:
                st.session_state[var] = []

        # Check for length mismatch between all history lists and reset if mismatched
        history_keys = [
            "playback_history_t", "playback_history_rb", "playback_history_gnss", "playback_history_master",
            "playback_history_utc", "playback_history_sats", "playback_history_mode",
            "playback_history_bias_est", "playback_history_drift_est",
            "playback_history_sigma", "playback_history_R", "playback_history_K",
            "playback_history_true_time", "playback_history_gnss_time",
            "playback_history_rubidium_time", "playback_history_disciplined_time",
            "playback_history_kf_x", "playback_history_kf_P",
            "playback_history_rb_phase_rw", "playback_history_rb_freq_offset",
            "playback_history_gnss_prop_err"
        ]
        lengths = [len(st.session_state[k_key]) for k_key in history_keys if k_key in st.session_state]
        if len(set(lengths)) > 1:
            for k_key in history_keys:
                st.session_state[k_key] = []

        # Safe programmatic increment of the epoch (before widgets are drawn)
        if st.session_state.get("playback_should_increment", False) and st.session_state.playback_playing:
            st.session_state.playback_epoch += 1
            st.session_state.playback_slider = st.session_state.playback_epoch
            st.session_state.playback_should_increment = False

        # Calculate dataset bounds and progress
        total_epochs = len(total_count)
        k = st.session_state.playback_epoch
        if k >= total_epochs:
            k = total_epochs - 1
            st.session_state.playback_epoch = k
            st.session_state.playback_playing = False

        first_time = times_analysis[0]
        last_time = times_analysis[-1]
        current_time = times_analysis[k % len(times_analysis)]
        time_left = last_time - current_time
        total_seconds_left = time_left.total_seconds()
        
        if total_seconds_left >= 0:
            hours = int(total_seconds_left // 3600)
            minutes = int((total_seconds_left % 3600) // 60)
            seconds = int(total_seconds_left % 60)
            time_left_str = f"{hours}h {minutes}m {seconds}s"
        else:
            time_left_str = "0h 0m 0s"

        col_play, col_prev, col_next, col_reset, col_csv, col_speed = st.columns([1, 0.8, 0.8, 1, 1, 2.5])
        with col_play:
            if st.session_state.playback_playing:
                if st.button("Pause", use_container_width=True, type="primary"):
                    st.session_state.playback_playing = False
                    safe_rerun()
            else:
                if st.button("Play", use_container_width=True, type="primary"):
                    st.session_state.playback_playing = True
                    safe_rerun()

        with col_prev:
            if st.button("Step Back", use_container_width=True, disabled=(k == 0), help="Step backward one epoch."):
                st.session_state.playback_playing = False
                history_keys = [
                    "playback_history_t", "playback_history_rb", "playback_history_gnss", "playback_history_master",
                    "playback_history_utc", "playback_history_sats", "playback_history_mode",
                    "playback_history_bias_est", "playback_history_drift_est",
                    "playback_history_sigma", "playback_history_R", "playback_history_K",
                    "playback_history_true_time", "playback_history_gnss_time",
                    "playback_history_rubidium_time", "playback_history_disciplined_time",
                    "playback_history_kf_x", "playback_history_kf_P",
                    "playback_history_rb_phase_rw", "playback_history_rb_freq_offset",
                    "playback_history_gnss_prop_err"
                ]
                if len(st.session_state.playback_history_t) > 1:
                    for h_key in history_keys:
                        if h_key in st.session_state and len(st.session_state[h_key]) > 0:
                            st.session_state[h_key].pop()
                    
                    new_k = max(0, k - 1)
                    st.session_state.playback_epoch = new_k
                    st.session_state.playback_slider = new_k
                    
                    # Reconstruct states & environment for display
                    raw_sats = total_count[new_k]
                    sats = max(0, int(raw_sats * sat_scale))
                    gps_sats = max(0, int(gps_count[new_k] * sat_scale))
                    gal_sats = max(0, int(galileo_count[new_k] * sat_scale))
                    glo_sats = max(0, int(glonass_count[new_k] * sat_scale))
                    sbas_sats = max(0, int(sbas_count[new_k] * sat_scale))
                    
                    st.session_state.playback_kf_x = st.session_state.playback_history_kf_x[-1].copy()
                    st.session_state.playback_kf_P = st.session_state.playback_history_kf_P[-1].copy()
                    st.session_state.playback_rb_phase_rw = st.session_state.playback_history_rb_phase_rw[-1]
                    st.session_state.playback_rb_freq_offset = st.session_state.playback_history_rb_freq_offset[-1]
                    st.session_state.playback_gnss_prop_err = st.session_state.playback_history_gnss_prop_err[-1]
                    
                    dt_step = float(dt)
                    Q_matrix = build_Q_matrix(dt_step)
                    
                    if sats < 4:
                        mode_str = "HOLDOVER"
                        R_k = 1e12
                    else:
                        if sats < 8:
                            mode_str = "DEGRADED"
                        else:
                            mode_str = "TRACKING"
                        unscaled_avg = np.mean(total_count)
                        unscaled_raw_mults = np.sqrt(unscaled_avg / np.maximum(1.0, np.array(total_count)))
                        mean_mult = np.mean(unscaled_raw_mults)
                        raw_mult = np.sqrt(unscaled_avg / max(1.0, sats))
                        multiplier = raw_mult / (mean_mult if mean_mult > 0 else 1.0)
                        R_k = (gnss_meas * multiplier) ** 2
                        
                    st.session_state.last_rendered_vals = {
                        "t": st.session_state.playback_history_t[-1],
                        "rb_err": st.session_state.playback_history_rb[-1],
                        "gnss_err": st.session_state.playback_history_gnss[-1],
                        "master_err": st.session_state.playback_history_master[-1],
                        "sats": sats,
                        "gps_sats": gps_sats,
                        "gal_sats": gal_sats,
                        "glo_sats": glo_sats,
                        "sbas_sats": sbas_sats,
                        "mode_str": mode_str,
                        "bias_est": st.session_state.playback_kf_x[0, 0],
                        "drift_est": st.session_state.playback_kf_x[1, 0],
                        "P_bias": st.session_state.playback_kf_P[0, 0],
                        "P": st.session_state.playback_kf_P,
                        "Q": Q_matrix,
                        "R_k": R_k,
                        "K_bias": st.session_state.playback_history_K[-1]
                    }
                else:
                    st.session_state.playback_epoch = 0
                    st.session_state.playback_slider = 0
                    for h_key in history_keys:
                        st.session_state[h_key] = []
                    st.session_state.playback_kf_x = np.array([[rb_bias - gnss_bias], [0.0]])
                    st.session_state.playback_kf_P = np.array([[1e-6, 0.0], [0.0, 1e-12]])
                    st.session_state.playback_rb_freq_offset = 0.0
                    st.session_state.playback_rb_phase_rw = 0.0
                    st.session_state.playback_gnss_prop_err = 0.0
                    if "playback_receiver" in st.session_state:
                        st.session_state.playback_receiver.reset(GLOBAL_RANDOM_SEED)
                
                safe_rerun()

        with col_next:
            if st.button("Step Forward", use_container_width=True, disabled=(k >= total_epochs - 1), help="Step forward one epoch."):
                st.session_state.playback_playing = False
                new_k = min(total_epochs - 1, k + 1)
                st.session_state.playback_epoch = new_k
                st.session_state.playback_slider = new_k
                st.session_state.playback_manual_step = True
                safe_rerun()

        with col_reset:
            if st.button("Reset", use_container_width=True):
                st.session_state.playback_epoch = 0
                st.session_state.playback_slider = 0
                st.session_state.playback_should_increment = False
                for h_key in history_keys:
                    st.session_state[h_key] = []
                st.session_state.playback_kf_x = np.array([[rb_bias - gnss_bias], [0.0]])
                st.session_state.playback_kf_P = np.array([[1e-6, 0.0], [0.0, 1e-12]])
                st.session_state.playback_rb_freq_offset = 0.0
                st.session_state.playback_rb_phase_rw = 0.0
                st.session_state.playback_gnss_prop_err = 0.0
                st.session_state.playback_playing = False
                if "playback_receiver" in st.session_state:
                    st.session_state.playback_receiver.reset(GLOBAL_RANDOM_SEED)
                safe_rerun()

        with col_csv:
            if len(st.session_state.playback_history_t) > 0:
                # Compile dataframe
                df_telemetry = pd.DataFrame({
                    "Epoch Time (s)": st.session_state.playback_history_t,
                    "UTC Timestamp": st.session_state.playback_history_utc,
                    "True Time": st.session_state.playback_history_true_time,
                    "GNSS Time": st.session_state.playback_history_gnss_time,
                    "Rubidium Time": st.session_state.playback_history_rubidium_time,
                    "Disciplined Time": st.session_state.playback_history_disciplined_time,
                    "Rubidium Error (ns)": np.array(st.session_state.playback_history_rb) * 1e9,
                    "GNSS Error (ns)": np.array(st.session_state.playback_history_gnss) * 1e9,
                    "Disciplined Master Clock Error (ns)": np.array(st.session_state.playback_history_master) * 1e9,
                    "Visible Satellites": st.session_state.playback_history_sats,
                    "Receiver Mode": st.session_state.playback_history_mode,
                    "Bias Estimate (ns)": st.session_state.playback_history_bias_est,
                    "Drift Estimate (ps/s)": st.session_state.playback_history_drift_est,
                    "3σ Uncertainty (ns)": st.session_state.playback_history_sigma,
                    "Current R (ns²)": st.session_state.playback_history_R,
                    "Kalman Gain": st.session_state.playback_history_K
                })
                csv_data = df_telemetry.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name=f"gnssdo_telemetry_epoch_{k}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.button("Download CSV", disabled=True, use_container_width=True, help="Run simulation to generate telemetry data first.")

        with col_speed:
            speed_interval = st.slider(
                "Step Delay (s)",
                min_value=0.1,
                max_value=2.0,
                value=0.5,
                step=0.1,
                help="Time delay between each simulation step."
            )

        # Seek slider (doubles as progress visualization and interactive positioning)
        seek_epoch = st.slider(
            "Playback Position (Seek in RINEX Dataset)",
            min_value=0,
            max_value=total_epochs - 1,
            key="playback_slider",
            format="Epoch %d",
            help="Drag the slider to seek to any epoch in the dataset."
        )
        
        # Check if user dragged the slider
        if seek_epoch != st.session_state.playback_epoch:
            st.session_state.playback_epoch = seek_epoch
            st.session_state.playback_playing = False  # Pause playback on seek to prevent snaps
            # Clear continuous history to prevent jumps on the plot
            for h_key in history_keys:
                st.session_state[h_key] = []
            # Reset running states
            st.session_state.playback_kf_x = np.array([[rb_bias - gnss_bias], [0.0]])
            st.session_state.playback_kf_P = np.array([[1e-6, 0.0], [0.0, 1e-12]])
            st.session_state.playback_rb_freq_offset = 0.0
            st.session_state.playback_rb_phase_rw = 0.0
            st.session_state.playback_gnss_prop_err = 0.0
            if "playback_receiver" in st.session_state:
                st.session_state.playback_receiver.reset(GLOBAL_RANDOM_SEED)
            safe_rerun()

        # Display progress info
        percentage = (k + 1) / total_epochs * 100
        st.markdown(
            f"<div style='margin-top: -15px; margin-bottom: 15px; font-size: 0.9rem; color: #475569; display: flex; justify-content: space-between;'>"
            f"<span><b>Current Epoch:</b> {k + 1} / {total_epochs} ({percentage:.1f}%)</span>"
            f"<span><b>Time Remaining in Data:</b> {time_left_str} (out of {(last_time - first_time).total_seconds() // 3600:.1f} hrs)</span>"
            f"</div>",
            unsafe_allow_html=True
        )

        # Step size dt for playback
        dt_step = float(dt)
        
        # Check if we have hit the end of RINEX data
        if k >= total_epochs - 1:
            st.session_state.playback_playing = False
            st.warning("Reached the end of the RINEX data epochs. Please reset the simulation.")
            k = total_epochs - 1

        # Only run calculation if playing, if manual step was requested, or if we have no history yet
        if st.session_state.playback_playing or st.session_state.get("playback_manual_step", False) or len(st.session_state.playback_history_t) == 0:
            # ── 1. Simulate Rubidium clock step (Reproducible with Seed) ──
            freq_step = 0.0 if k == 0 else np.random.default_rng(GLOBAL_RANDOM_SEED + k).normal(0, rb_rw_step)
            st.session_state.playback_rb_freq_offset += freq_step
            st.session_state.playback_rb_phase_rw += st.session_state.playback_rb_freq_offset * dt_step
            
            aging_k = 0.5 * rb_aging * (k * dt_step)**2
            white_noise_k = np.random.default_rng(GLOBAL_RANDOM_SEED + k + 100000).normal(0, rb_noise)
            rubidium_error_k = rb_bias + st.session_state.playback_rb_phase_rw + aging_k + white_noise_k
            
            # ── 2. Simulate GNSS error step using Receiver Abstraction ──
            res = st.session_state.playback_receiver.measure(k * dt_step, dt_step)
            gnss_error_k = res["gnss_error"]
            sats = res["sat_count"]
            gps_sats = res["gps_count"]
            gal_sats = res["galileo_count"]
            glo_sats = res["glonass_count"]
            sbas_sats = res["sbas_count"]
            R_k = res["R"]
            
            if sats < 4:
                mode_str = "HOLDOVER"
            elif sats < 8:
                mode_str = "DEGRADED"
            else:
                mode_str = "TRACKING"

            
            # ── 5. Kalman Filter Prediction & Propagation ──
            x = st.session_state.playback_kf_x
            P = st.session_state.playback_kf_P
            
            F = np.array([[1.0, dt_step],
                          [0.0, 1.0]])
            Q_matrix = build_Q_matrix(dt_step)
            
            # Predict step
            x_pred = F @ x
            P_pred = F @ P @ F.T + Q_matrix
            
            # Update step (only if not in HOLDOVER)
            if mode_str != "HOLDOVER":
                z = rubidium_error_k - gnss_error_k
                H = np.array([[1.0, 0.0]])
                innov_k = z - x_pred[0, 0]
                S = P_pred[0, 0] + R_k
                K = P_pred[:, 0:1] / S
                
                x = x_pred + K * innov_k
                P = (np.eye(2) - K @ H) @ P_pred
                K_bias = K[0, 0]
            else:
                # Holdover: state propagates with prediction, no measurement update
                x = x_pred
                P = P_pred
                innov_k = np.nan
                K_bias = 0.0
                
            # Disciplined master clock error
            master_error_k = rubidium_error_k - x[0, 0]
            
            # Save updated states
            st.session_state.playback_kf_x = x
            st.session_state.playback_kf_P = P
            
            # Save history (limit to last 1000 steps to conserve memory)
            epoch_dt = times_analysis[k % len(times_analysis)]
            utc_str = epoch_dt.strftime('%Y-%m-%d %H:%M:%S.%f') + " UTC"
            bias_ns = x[0, 0] * 1e9
            drift_ps = x[1, 0] * 1e12
            sigma_ns = 3.0 * np.sqrt(P[0, 0]) * 1e9
            R_ns2 = R_k * 1e18

            st.session_state.playback_history_t.append(k * dt_step)
            st.session_state.playback_history_rb.append(rubidium_error_k)
            st.session_state.playback_history_gnss.append(gnss_error_k)
            st.session_state.playback_history_master.append(master_error_k)
            st.session_state.playback_history_utc.append(utc_str)
            st.session_state.playback_history_sats.append(sats)
            st.session_state.playback_history_mode.append(mode_str)
            st.session_state.playback_history_bias_est.append(bias_ns)
            st.session_state.playback_history_drift_est.append(drift_ps)
            st.session_state.playback_history_sigma.append(sigma_ns)
            st.session_state.playback_history_R.append(R_ns2)
            st.session_state.playback_history_K.append(K_bias)
            st.session_state.playback_history_true_time.append(format_telemetry_time(epoch_dt, time_only=True) + " UTC")
            st.session_state.playback_history_gnss_time.append(format_telemetry_time(epoch_dt, gnss_error_k, time_only=True) + " UTC")
            st.session_state.playback_history_rubidium_time.append(format_telemetry_time(epoch_dt, rubidium_error_k, time_only=True) + " UTC")
            st.session_state.playback_history_disciplined_time.append(format_telemetry_time(epoch_dt, master_error_k, time_only=True) + " UTC")
            st.session_state.playback_history_kf_x.append(x.copy())
            st.session_state.playback_history_kf_P.append(P.copy())
            st.session_state.playback_history_rb_phase_rw.append(st.session_state.playback_rb_phase_rw)
            st.session_state.playback_history_rb_freq_offset.append(st.session_state.playback_rb_freq_offset)
            st.session_state.playback_history_gnss_prop_err.append(st.session_state.playback_gnss_prop_err)
            
            if len(st.session_state.playback_history_t) > 1000:
                st.session_state.playback_history_t.pop(0)
                st.session_state.playback_history_rb.pop(0)
                st.session_state.playback_history_gnss.pop(0)
                st.session_state.playback_history_master.pop(0)
                st.session_state.playback_history_utc.pop(0)
                st.session_state.playback_history_sats.pop(0)
                st.session_state.playback_history_mode.pop(0)
                st.session_state.playback_history_bias_est.pop(0)
                st.session_state.playback_history_drift_est.pop(0)
                st.session_state.playback_history_sigma.pop(0)
                st.session_state.playback_history_R.pop(0)
                st.session_state.playback_history_K.pop(0)
                st.session_state.playback_history_true_time.pop(0)
                st.session_state.playback_history_gnss_time.pop(0)
                st.session_state.playback_history_rubidium_time.pop(0)
                st.session_state.playback_history_disciplined_time.pop(0)
                st.session_state.playback_history_kf_x.pop(0)
                st.session_state.playback_history_kf_P.pop(0)
                st.session_state.playback_history_rb_phase_rw.pop(0)
                st.session_state.playback_history_rb_freq_offset.pop(0)
                st.session_state.playback_history_gnss_prop_err.pop(0)

            # Reset manual step flag
            st.session_state.playback_manual_step = False
                
            # Store current values for rendering (to prevent display updates while paused from recalculating)
            st.session_state.last_rendered_vals = {
                "t": k * dt_step,
                "rb_err": rubidium_error_k,
                "gnss_err": gnss_error_k,
                "master_err": master_error_k,
                "sats": sats,
                "gps_sats": gps_sats,
                "gal_sats": gal_sats,
                "glo_sats": glo_sats,
                "sbas_sats": sbas_sats,
                "mode_str": mode_str,
                "bias_est": x[0, 0],
                "drift_est": x[1, 0],
                "P_bias": P[0, 0],
                "P": P,
                "Q": Q_matrix,
                "R_k": R_k,
                "K_bias": K_bias
            }

        # Fetch last calculated parameters for rendering
        vals = st.session_state.last_rendered_vals
        
        # ── 6. Holdover Uncertainty Propagation ──
        sigma_1m, sigma_10m, sigma_1h = calculate_holdover_uncertainties(vals["P"], dt_step, vals["Q"])
        
        # ── 7. Epoch date calculations for displays ──
        epoch_dt = times_analysis[k % len(times_analysis)]
        true_time_str = format_telemetry_time(epoch_dt)
        gnss_time_str = format_telemetry_time(epoch_dt, vals["gnss_err"], time_only=True)
        rubidium_time_str = format_telemetry_time(epoch_dt, vals["rb_err"], time_only=True)
        disciplined_time_str = format_telemetry_time(epoch_dt, vals["master_err"], time_only=True)
        
        # ── 8. Render UI Panels ──
        
        # Top Row: Large UTC clock card (No Status Badge, light styled)
        st.markdown(f"""
        <div class="telemetry-card" style="text-align: center; border-color: #e2e8f0; background-color: #f8fafc;">
            <div class="telemetry-header" style="color: #4f46e5; font-size: 1.1rem; border-bottom: none; margin-bottom: 0;">GNSSDO MASTER CLOCK</div>
            <div class="telemetry-value-large" style="font-size: 2.8rem; color: #1e1b4b; margin: 10px 0;">{true_time_str}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Middle Rows
        col_row1_left, col_row1_right = st.columns(2)
        
        with col_row1_left:
            # Time Sources Panel
            st.markdown(f"""
            <div class="telemetry-card">
                <div class="telemetry-header">Time Sources</div>
                <div class="telemetry-row">
                    <span class="telemetry-label">TRUE TIME</span>
                    <span class="telemetry-value" style="color: #64748b;">{format_telemetry_time(epoch_dt, time_only=True)}</span>
                </div>
                <div class="telemetry-row">
                    <span class="telemetry-label">GNSS TIME</span>
                    <span class="telemetry-value">{gnss_time_str} ({format_offset(vals["gnss_err"])})</span>
                </div>
                <div class="telemetry-row">
                    <span class="telemetry-label">RUBIDIUM TIME</span>
                    <span class="telemetry-value">{rubidium_time_str} ({format_offset(vals["rb_err"])})</span>
                </div>
                <div class="telemetry-row">
                    <span class="telemetry-label">DISCIPLINED TIME</span>
                    <span class="telemetry-value" style="color: #4f46e5;">{disciplined_time_str} ({format_offset(vals["master_err"])})</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # PPS Panel
            pps_expected_str = format_telemetry_time(epoch_dt, time_only=True).split('.')[0] + ".000000000"
            pps_received_str = format_telemetry_time(epoch_dt, vals["master_err"], time_only=True)
            st.markdown(f"""
            <div class="telemetry-card">
                <div class="telemetry-header">PPS Hardware Emulation</div>
                <div class="telemetry-row">
                    <span class="telemetry-label">Expected PPS</span>
                    <span class="telemetry-value" style="color: #64748b;">{pps_expected_str}</span>
                </div>
                <div class="telemetry-row">
                    <span class="telemetry-label">Received PPS</span>
                    <span class="telemetry-value">{pps_received_str}</span>
                </div>
                <div class="telemetry-row">
                    <span class="telemetry-label">Hardware Offset</span>
                    <span class="telemetry-value" style="color: #059669;">{format_offset(vals["master_err"])}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_row1_right:
            # Constellation Panel
            st.markdown(f"""
            <div class="telemetry-card">
                <div class="telemetry-header">Constellation (RINEX Station AB04)</div>
                <div class="telemetry-row">
                    <span class="telemetry-label">GPS</span>
                    <span class="telemetry-value" style="color: #059669;">{vals["gps_sats"]}</span>
                </div>
                <div class="telemetry-row">
                    <span class="telemetry-label">Galileo (GAL)</span>
                    <span class="telemetry-value" style="color: #d97706;">{vals["gal_sats"]}</span>
                </div>
                <div class="telemetry-row">
                    <span class="telemetry-label">GLONASS (GLO)</span>
                    <span class="telemetry-value" style="color: #dc2626;">{vals["glo_sats"]}</span>
                </div>
                <div class="telemetry-row">
                    <span class="telemetry-label">SBAS</span>
                    <span class="telemetry-value" style="color: #7c3aed;">{vals["sbas_sats"]}</span>
                </div>
                <div class="telemetry-row" style="border-top: 1px solid #e2e8f0; padding-top: 8px; font-weight: bold;">
                    <span class="telemetry-label" style="color: #0f172a;">TOTAL</span>
                    <span class="telemetry-value" style="color: #4f46e5;">{vals["sats"]}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Kalman Filter Panel
            bias_ns = vals["bias_est"] * 1e9
            drift_ps = vals["drift_est"] * 1e12
            sigma_ns = 3.0 * np.sqrt(vals["P_bias"]) * 1e9
            
            R_k = vals.get("R_k", 0.0)
            if R_k >= 1e10:
                R_str = "Lock Loss (∞)"
            else:
                R_str = f"{np.sqrt(R_k)*1e9:.1f} ns ({R_k*1e18:.0f} ns²)"
                
            K_bias = vals.get("K_bias", 0.0)
            
            if vals["mode_str"] == "TRACKING":
                mode_color = "#059669" # green
            elif vals["mode_str"] == "DEGRADED":
                mode_color = "#d97706" # orange
            else:
                mode_color = "#dc2626" # red
                
            st.markdown(f"""
            <div class="telemetry-card">
                <div class="telemetry-header">Kalman Filter State</div>
                <div class="telemetry-row">
                    <span class="telemetry-label">Bias Estimate</span>
                    <span class="telemetry-value">{bias_ns:.1f} ns</span>
                </div>
                <div class="telemetry-row">
                    <span class="telemetry-label">Drift Estimate</span>
                    <span class="telemetry-value">{drift_ps:.2f} ps/s</span>
                </div>
                <div class="telemetry-row">
                    <span class="telemetry-label">3σ Uncertainty</span>
                    <span class="telemetry-value">±{sigma_ns:.1f} ns</span>
                </div>
                <div class="telemetry-row">
                    <span class="telemetry-label">Current R</span>
                    <span class="telemetry-value">{R_str}</span>
                </div>
                <div class="telemetry-row">
                    <span class="telemetry-label">Kalman Gain</span>
                    <span class="telemetry-value">{K_bias:.4f}</span>
                </div>
                <div class="telemetry-row">
                    <span class="telemetry-label">Mode</span>
                    <span class="telemetry-value" style="color: {mode_color};">{vals["mode_str"]}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Bottom Row: Holdover & Chart
        col_row2_left, col_row2_right = st.columns([1.2, 2])
        
        with col_row2_left:
            # Holdover Prediction Panel
            st.markdown(f"""
            <div class="telemetry-card" style="height: 94%;">
                <div class="telemetry-header">Holdover Prediction (If GNSS Lost NOW)</div>
                <div class="telemetry-row" style="margin: 15px 0;">
                    <span class="telemetry-label">1 minute</span>
                    <span class="telemetry-value" style="color: #d97706;">{format_uncertainty(sigma_1m)}</span>
                </div>
                <div class="telemetry-row" style="margin: 15px 0;">
                    <span class="telemetry-label">10 minutes</span>
                    <span class="telemetry-value" style="color: #ea580c;">{format_uncertainty(sigma_10m)}</span>
                </div>
                <div class="telemetry-row" style="margin: 15px 0;">
                    <span class="telemetry-label">1 hour</span>
                    <span class="telemetry-value" style="color: #dc2626;">{format_uncertainty(sigma_1h)}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_row2_right:
            # Live Charts
            if len(st.session_state.playback_history_t) > 1:
                st.markdown("##### Live Clock Errors (relative to True Time)")
                fig_playback = plot_live_playback_errors(
                    st.session_state.playback_history_t,
                    st.session_state.playback_history_gnss,
                    st.session_state.playback_history_master,
                    st.session_state.playback_history_rb
                )
                st.pyplot(fig_playback)
                plt.close(fig_playback)
                
                st.markdown("##### Live UTC Clock Times")
                fig_times = plot_live_playback_times(
                    st.session_state.playback_history_t,
                    st.session_state.playback_history_gnss,
                    st.session_state.playback_history_master,
                    st.session_state.playback_history_rb,
                    times_analysis,
                    k
                )
                st.pyplot(fig_times)
                plt.close(fig_times)
            else:
                st.info("Simulation started. Live telemetry charts will render once data accumulates.")

        # ── 9. Handle Loop Sleep and Rerun ──
        if st.session_state.playback_playing:
            time.sleep(speed_interval)
            st.session_state.playback_should_increment = True
            safe_rerun()

    run_playback_panel()
