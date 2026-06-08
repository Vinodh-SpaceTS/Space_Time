# analysis/gnss_analysis.py
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from hifitime import Epoch, TimeScale

def parse_rinex_file(filepath):
    """
    Parses a RINEX 2 observation file to extract visible satellite counts and constellation breakdowns.
    
    Parameters:
      filepath (str): Absolute or relative path to the RINEX file.
      
    Returns:
      times (list): List of datetime objects for each epoch.
      total_count (list): List of total visible satellites per epoch.
      gps_count (list): List of visible GPS satellites per epoch.
      galileo_count (list): List of visible Galileo satellites per epoch.
      glonass_count (list): List of visible GLONASS satellites per epoch.
      sbas_count (list): List of visible SBAS satellites per epoch.
    """
    try:
        with open(filepath, "r") as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error opening RINEX file: {e}")
        return [], [], [], [], [], []

    # Find the header end
    header_end = None
    for i, line in enumerate(lines):
        if "END OF HEADER" in line:
            header_end = i
            break
            
    if header_end is None:
        print("Invalid RINEX file: 'END OF HEADER' not found.")
        return [], [], [], [], [], []
        
    times = []
    total_count = []
    gps_count = []
    galileo_count = []
    glonass_count = []
    sbas_count = []

    idx = header_end + 1
    N_lines = len(lines)

    while idx < N_lines:
        line = lines[idx]
        
        if len(line) < 32:
            idx += 1
            continue

        try:
            # Parse date and time from the epoch header
            year = int(line[0:3])
            month = int(line[3:6])
            day = int(line[6:9])
            hour = int(line[9:12])
            minute = int(line[12:15])
            second = float(line[15:26])

            sat_count = int(line[29:32])

            # Simple RINEX Y2K handling
            yr = 2000 + year if year < 80 else 1900 + year
            sec_whole = int(second)
            nsec = int(round((second - sec_whole) * 1e9))
            
            # Construct Epoch in GPST (GPS Time scale - standard for RINEX observations)
            epoch_gps = Epoch.from_gregorian(yr, month, day, hour, minute, sec_whole, nsec, TimeScale.GPST)
            
            # Convert to UTC time scale (automatically handling leap seconds offset)
            epoch_utc = epoch_gps.to_time_scale(TimeScale.UTC)
            
            # Convert back to standard datetime object for plotting compatibility
            dt = epoch_utc.to_datetime()

            # Collect satellite IDs (some epochs span multiple lines)
            sat_text = line[32:]
            satellites = []

            for j in range(0, len(sat_text), 3):
                sat = sat_text[j:j+3].strip()
                if sat:
                    satellites.append(sat)

            extra_line = idx + 1

            while len(satellites) < sat_count and extra_line < N_lines:
                extra_sat_text = lines[extra_line][32:]
                for j in range(0, len(extra_sat_text), 3):
                    sat = extra_sat_text[j:j+3].strip()
                    if sat:
                        satellites.append(sat)
                extra_line += 1

            # Count satellites by constellation
            gps = 0
            gal = 0
            glo = 0
            sbas = 0

            for sat in satellites:
                if sat.startswith("G"):
                    gps += 1
                elif sat.startswith("E"):
                    gal += 1
                elif sat.startswith("R"):
                    glo += 1
                elif sat.startswith("S"):
                    sbas += 1

            times.append(dt)
            total_count.append(sat_count)
            gps_count.append(gps)
            galileo_count.append(gal)
            glonass_count.append(glo)
            sbas_count.append(sbas)

            idx = extra_line

        except Exception:
            # Skip invalid epoch lines
            idx += 1

    return times, total_count, gps_count, galileo_count, glonass_count, sbas_count

def parse_rinex_metadata(filepath):
    """
    Parses a RINEX 2 observation file header to extract key metadata.
    """
    metadata = {
        "version": "Unknown",
        "type": "Unknown",
        "marker_name": "Unknown",
        "observer_agency": "Unknown",
        "receiver_info": "Unknown",
        "antenna_info": "Unknown",
        "approx_position": "Unknown",
        "first_obs_time": "Unknown",
        "leap_seconds": "Unknown",
        "obs_types": []
    }
    
    try:
        with open(filepath, "r") as f:
            for line in f:
                if "END OF HEADER" in line:
                    break
                
                label = line[60:].strip()
                content = line[0:60].strip()
                
                if "RINEX VERSION / TYPE" in label:
                    parts = line[0:60].split()
                    if len(parts) >= 1:
                        metadata["version"] = parts[0]
                    if len(parts) >= 2:
                        metadata["type"] = parts[1]
                elif "MARKER NAME" in label:
                    metadata["marker_name"] = content
                elif "OBSERVER / AGENCY" in label:
                    metadata["observer_agency"] = content
                elif "REC # / TYPE / VERS" in label:
                    metadata["receiver_info"] = content
                elif "ANT # / TYPE" in label:
                    metadata["antenna_info"] = content
                elif "APPROX POSITION XYZ" in label:
                    metadata["approx_position"] = content
                elif "TIME OF FIRST OBS" in label:
                    # Parse into a nicer format
                    parts = content.split()
                    if len(parts) >= 6:
                        metadata["first_obs_time"] = f"{parts[0]}-{parts[1]:>02}-{parts[2]:>02} {parts[3]:>02}:{parts[4]:>02}:{float(parts[5]):09.6f} {parts[6]}"
                    else:
                        metadata["first_obs_time"] = content
                elif "LEAP SECONDS" in label:
                    metadata["leap_seconds"] = content
                elif "# / TYPES OF OBSERV" in label:
                    parts = content.split()
                    for p in parts:
                        if not p.isdigit() and len(p) <= 3:
                            metadata["obs_types"].append(p)
    except Exception as e:
        print(f"Error parsing RINEX metadata: {e}")
        
    return metadata

def plot_satellite_visibility(times_analysis, total_count, gps_count, galileo_count, glonass_count, sbas_count):
    """
    Plots the visible satellite counts by constellation over time.
    Uses the exact styling from the Streamlit Web UI.
    """
    fig_sat, ax_sat = plt.subplots(figsize=(14, 3.8))
    ax_sat.plot(times_analysis, total_count, label="Total Satellites", color="#2563eb", linewidth=2.0)
    ax_sat.plot(times_analysis, gps_count, label="GPS", color="#059669", linewidth=1.2, alpha=0.8)
    ax_sat.plot(times_analysis, galileo_count, label="Galileo", color="#ea580c", linewidth=1.2, alpha=0.8)
    ax_sat.plot(times_analysis, glonass_count, label="GLONASS", color="#dc2626", linewidth=1.2, alpha=0.8)
    ax_sat.plot(times_analysis, sbas_count, label="SBAS", color="#7c3aed", linewidth=1.2, alpha=0.8)
    ax_sat.set_ylabel("Satellites Count", color="#334155")
    ax_sat.set_xlabel("Time (UTC on 10 April 2025)", color="#334155")
    ax_sat.grid(True, linestyle=":", color="#cbd5e1")
    ax_sat.legend(loc="upper right", ncol=5, framealpha=0.8, labelcolor="#334155")
    ax_sat.patch.set_facecolor('#ffffff')
    fig_sat.patch.set_facecolor('#f8fafc')
    ax_sat.tick_params(colors='#334155')
    plt.tight_layout()
    return fig_sat

if __name__ == "__main__":
    # Test script standalone
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from config import RINEX_FILEPATH
    
    print(f"Testing analysis/gnss_analysis.py with file: {RINEX_FILEPATH}")
    
    t, tot, gps, gal, glo, sbas = [], [], [], [], [], []
    if os.path.exists(RINEX_FILEPATH):
        t, tot, gps, gal, glo, sbas = parse_rinex_file(RINEX_FILEPATH)
        print(f"Parsed {len(t)} epochs successfully from RINEX file.")
        
        # Parse and print metadata
        meta = parse_rinex_metadata(RINEX_FILEPATH)
        print("\n========== RINEX HEADER METADATA ==========")
        for key, val in meta.items():
            print(f"{key.replace('_', ' ').title()}: {val}")
        print("===========================================\n")
    else:
        print(f"Warning: RINEX data file not found at '{RINEX_FILEPATH}'. Generating simulated fallback dataset...")
        t = [datetime(2025, 4, 10, h, m, 0) for h in range(24) for m in range(0, 60, 30)]
        np.random.seed(42)
        tot = list(np.random.randint(25, 40, len(t)))
        gps = list(np.random.randint(10, 15, len(t)))
        gal = list(np.random.randint(8, 12, len(t)))
        glo = list(np.random.randint(5, 10, len(t)))
        sbas = list(np.random.randint(1, 4, len(t)))
        print(f"Generated {len(t)} simulated epochs.")
        
    if len(t) > 0:
        print(f"First Epoch: {t[0]} -> {tot[0]} satellites (GPS={gps[0]}, GAL={gal[0]}, GLO={glo[0]}, SBAS={sbas[0]})")
        
        # Generate and show the exact Web UI plot
        print("Displaying standalone plot (same as Web UI)...")
        fig = plot_satellite_visibility(t, tot, gps, gal, glo, sbas)
        plt.show()
    else:
        print("Error: No data available for plotting.")
