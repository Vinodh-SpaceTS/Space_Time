# config.py
# Centralized simulation configuration parameters

# ====================================================
# SIMULATION SETTINGS
# ====================================================
SIM_DURATION = 10000      # Total simulation duration in seconds
SIM_DT = 1                # Time step in seconds

# ====================================================
# RUBIDIUM CLOCK DEFAULT PARAMETERS
# ====================================================
RB_BIAS = 1e-3                  # 1 ms initial phase offset
RB_RANDOM_WALK_STEP = 1e-10     # Random walk step size (seconds)
RB_WHITE_NOISE_STD = 5e-9       # Realistic short-term white noise (5 ns)
RB_AGING_RATE = 1e-13           # Linear frequency aging rate (seconds/sec^2)

# ====================================================
# GNSS RECEIVER DEFAULT ERRORS
# ====================================================
GNSS_BIAS = 100e-9              # Constant receiver timing bias (100 ns)
GNSS_SAT_CLOCK_STD = 20e-9      # Satellite clock error std dev (20 ns)
GNSS_PROP_STD = 30e-9           # Propagation error std dev (30 ns)
GNSS_MEAS_STD = 50e-9           # GNSS measurement noise std dev (50 ns)

# ====================================================
# OUTAGE SETTINGS
# ====================================================
OUTAGE_START = 3000             # GNSS outage start time (seconds)
OUTAGE_END = 5000               # GNSS outage end time (seconds)

# ====================================================
# DATA PATHS
# ====================================================
RINEX_FILEPATH = r"data/ab041000.25o"
