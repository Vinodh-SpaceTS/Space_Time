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
VALIDATION_RINEX_FILEPATH = r"GNSS_DATA/ALGO00CAN_R_20251000000_01D_30S_MO.crx.gz"
VALIDATION_CLOCK_FILEPATH = r"GNSS_DATA/ESA0OPSFIN_20251000000_01D_30S_CLK.CLK.gz"
VALIDATION_CLOCK_TRUTH_CSV = r"data/algo_receiver_clock_truth_2025100.csv"

# ====================================================
# REPRODUCIBILITY
# ====================================================
GLOBAL_RANDOM_SEED = 42

def validate_config():
    """
    Validates that the configuration parameters are within realistic and physical bounds.
    Raises ValueError if any parameter is invalid.
    """
    if SIM_DURATION <= 0:
        raise ValueError(f"SIM_DURATION must be positive, got {SIM_DURATION}")
    if SIM_DT <= 0:
        raise ValueError(f"SIM_DT must be positive, got {SIM_DT}")
    if SIM_DT > SIM_DURATION:
        raise ValueError(f"SIM_DT ({SIM_DT}) cannot be larger than SIM_DURATION ({SIM_DURATION})")
    
    if RB_BIAS < 0:
        raise ValueError(f"RB_BIAS cannot be negative, got {RB_BIAS}")
    if RB_RANDOM_WALK_STEP < 0 or RB_RANDOM_WALK_STEP > 1e-3:
        raise ValueError(f"RB_RANDOM_WALK_STEP must be in [0, 1e-3], got {RB_RANDOM_WALK_STEP}")
    if RB_WHITE_NOISE_STD < 0 or RB_WHITE_NOISE_STD > 1e-3:
        raise ValueError(f"RB_WHITE_NOISE_STD must be in [0, 1e-3], got {RB_WHITE_NOISE_STD}")
    if RB_AGING_RATE < 0 or RB_AGING_RATE > 1e-3:
        raise ValueError(f"RB_AGING_RATE must be in [0, 1e-3], got {RB_AGING_RATE}")
        
    if GNSS_BIAS < 0 or GNSS_BIAS > 1e-3:
        raise ValueError(f"GNSS_BIAS must be in [0, 1e-3], got {GNSS_BIAS}")
    if GNSS_SAT_CLOCK_STD < 0 or GNSS_SAT_CLOCK_STD > 1e-3:
        raise ValueError(f"GNSS_SAT_CLOCK_STD must be in [0, 1e-3], got {GNSS_SAT_CLOCK_STD}")
    if GNSS_PROP_STD < 0 or GNSS_PROP_STD > 1e-3:
        raise ValueError(f"GNSS_PROP_STD must be in [0, 1e-3], got {GNSS_PROP_STD}")
    if GNSS_MEAS_STD < 0 or GNSS_MEAS_STD > 1e-3:
        raise ValueError(f"GNSS_MEAS_STD must be in [0, 1e-3], got {GNSS_MEAS_STD}")
        
    if OUTAGE_START < 0:
        raise ValueError(f"OUTAGE_START cannot be negative, got {OUTAGE_START}")
    if OUTAGE_END < OUTAGE_START:
        raise ValueError(f"OUTAGE_END ({OUTAGE_END}) cannot be less than OUTAGE_START ({OUTAGE_START})")

