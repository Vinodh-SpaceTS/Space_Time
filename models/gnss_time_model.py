# models/gnss_time_model.py
import numpy as np
import matplotlib.pyplot as plt

def simulate_gnss_time(duration, dt, receiver_bias, sat_clock_std, prop_std, meas_std, 
                       use_gauss_markov=False, correlation_time=3600.0,
                       sat_counts=None, dynamic_r_enabled=False, seed=None):
    """
    Simulates GNSS receiver timing errors, optionally using a time-varying measurement noise 
    profile based on satellite visibility (constellation-driven covariance adaptation).
    
    Under the hood, this uses the SimulatedReceiver class.
    
    Parameters:
      duration (float): Total simulation time in seconds.
      dt (float): Sample interval in seconds.
      receiver_bias (float): Constant receiver clock bias in seconds.
      sat_clock_std (float): Standard deviation of satellite clock error (seconds).
      prop_std (float): Standard deviation of atmospheric propagation delay (seconds).
      meas_std (float): Standard deviation of receiver measurement noise (seconds).
      use_gauss_markov (bool): If True, models propagation error as a first-order Gauss-Markov process.
      correlation_time (float): Correlation time constant for the Gauss-Markov process (seconds).
      sat_counts (list/ndarray): Time series of visible satellite counts mapped to epochs.
      dynamic_r_enabled (bool): Enable satellite-visibility-driven scaling of measurement noise.
      seed (int): Optional random seed for reproducibility.
      
    Returns:
      gnss_error (ndarray): Array of total GNSS timing errors (seconds).
      sat_clock_error (ndarray): Array of satellite clock errors (seconds).
      propagation_error (ndarray): Array of atmospheric propagation delays (seconds).
      measurement_noise (ndarray): Array of receiver measurement noises (seconds).
      R_profile (ndarray): Time-varying measurement noise variance profile (seconds^2).
    """
    from receivers.simulated_receiver import SimulatedReceiver
    
    rx = SimulatedReceiver(
        receiver_bias=receiver_bias,
        sat_clock_std=sat_clock_std,
        prop_std=prop_std,
        meas_std=meas_std,
        use_gauss_markov=use_gauss_markov,
        correlation_time=correlation_time,
        total_counts=sat_counts,
        dynamic_r_enabled=dynamic_r_enabled,
        seed=seed
    )
    
    true_time = np.arange(0, duration, dt)
    N = len(true_time)
    
    gnss_error = np.zeros(N)
    sat_clock_error = np.zeros(N)
    propagation_error = np.zeros(N)
    measurement_noise = np.zeros(N)
    R_profile = np.zeros(N)
    
    for i, t in enumerate(true_time):
        res = rx.measure(t, dt)
        gnss_error[i] = res["gnss_error"]
        sat_clock_error[i] = res["sat_clock_error"]
        propagation_error[i] = res["propagation_error"]
        measurement_noise[i] = res["measurement_noise"]
        R_profile[i] = res["R"]
        
    return gnss_error, sat_clock_error, propagation_error, measurement_noise, R_profile


def plot_gnss_time_components(true_time, sat_clock_error, propagation_error, measurement_noise):
    """
    Plots the GNSS receiver timing error components (Satellite clock, propagation delay, measurement noise).
    Uses the exact styling from the Streamlit Web UI.
    """
    fig_gnss, ax_gnss = plt.subplots(figsize=(7, 3.8))
    limit = min(200, len(true_time))
    ax_gnss.plot(true_time[:limit], sat_clock_error[:limit] * 1e9, label="Sat Clock Error", color="#ea580c", linewidth=1.5)
    ax_gnss.plot(true_time[:limit], propagation_error[:limit] * 1e9, label="Propagation Delay", color="#7c3aed", linewidth=1.5)
    ax_gnss.plot(true_time[:limit], measurement_noise[:limit] * 1e9, label="Measurement Noise", color="#059669", linewidth=1.2, alpha=0.7)
    ax_gnss.set_ylabel("Error (ns)", color="#334155")
    ax_gnss.set_xlabel("Time (s) [First 200s shown for clarity]", color="#334155")
    ax_gnss.grid(True, linestyle=":", color="#cbd5e1")
    ax_gnss.legend(loc="upper right", framealpha=0.8, labelcolor="#334155")
    ax_gnss.patch.set_facecolor('#ffffff')
    fig_gnss.patch.set_facecolor('#f8fafc')
    ax_gnss.tick_params(colors='#334155')
    plt.tight_layout()
    return fig_gnss

def plot_full_gnss_error(t, error, receiver_bias_ns):
    """
    Plots the total GNSS error over the full duration.
    """
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(t, error * 1e9, color='#059669', label='Total GNSS Error', alpha=0.8, linewidth=1.0)
    ax.axhline(receiver_bias_ns, color='#e11d48', linestyle='--', label=f'Receiver Bias ({receiver_bias_ns:.1f} ns)', linewidth=1.5)
    ax.set_ylabel('Error (ns)', color="#334155")
    ax.set_xlabel('Time (s)', color="#334155")
    ax.set_title('Total GNSS Timing Error', color="#0f172a", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle=":", color="#cbd5e1")
    ax.legend(loc='upper right', framealpha=0.8, labelcolor="#334155")
    ax.patch.set_facecolor('#ffffff')
    fig.patch.set_facecolor('#f8fafc')
    ax.tick_params(colors='#334155')
    plt.tight_layout()
    return fig

def plot_gnss_component_breakdown(t, error, sat, prop, meas, zoom_limit=500):
    """
    Plots a zoomed-in component breakdown of the GNSS timing error.
    """
    fig, ax = plt.subplots(figsize=(7, 3.8))
    limit = min(zoom_limit, len(t))
    t_zoom = t[:limit]
    ax.plot(t_zoom, error[:limit] * 1e9, color='#059669', label='Total GNSS Error', linewidth=1.5, alpha=0.6)
    ax.plot(t_zoom, sat[:limit] * 1e9, color='#ea580c', label='Sat Clock Error', linewidth=1.2)
    ax.plot(t_zoom, prop[:limit] * 1e9, color='#7c3aed', label='Propagation Delay', linewidth=1.5)
    ax.plot(t_zoom, meas[:limit] * 1e9, color='#64748b', label='Measurement Noise', linewidth=0.8, alpha=0.7)
    
    ax.set_xlabel('Time (s)', color="#334155")
    ax.set_ylabel('Error Components (ns)', color="#334155")
    ax.set_title(f'Component Breakdown (First {limit}s Zoom)', color="#0f172a", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle=":", color="#cbd5e1")
    ax.legend(loc='upper right', framealpha=0.8, labelcolor="#334155")
    ax.patch.set_facecolor('#ffffff')
    fig.patch.set_facecolor('#f8fafc')
    ax.tick_params(colors='#334155')
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    # Test script standalone
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from config import SIM_DURATION, SIM_DT, GNSS_BIAS, GNSS_SAT_CLOCK_STD, GNSS_PROP_STD, GNSS_MEAS_STD
    
    print("Testing models/gnss_time_model.py...")
    error, sat, prop, meas, R_profile = simulate_gnss_time(
        SIM_DURATION, SIM_DT, GNSS_BIAS, GNSS_SAT_CLOCK_STD, GNSS_PROP_STD, GNSS_MEAS_STD, use_gauss_markov=True
    )
    print(f"Simulation completed. Length: {len(error)}")
    print(f"Mean Error: {np.mean(error)*1e9:.3f} ns")
    print(f"Std Error:  {np.std(error)*1e9:.3f} ns")
    
    t = np.arange(0, SIM_DURATION, SIM_DT)
    
    print("Displaying standalone plots (same as Web UI)...")
    fig1 = plot_gnss_time_components(t, sat, prop, meas)
    fig2 = plot_full_gnss_error(t, error, GNSS_BIAS * 1e9)
    fig3 = plot_gnss_component_breakdown(t, error, sat, prop, meas, zoom_limit=500)
    plt.show()
