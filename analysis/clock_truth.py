# analysis/clock_truth.py
from __future__ import annotations

import csv
import gzip
from datetime import datetime, timezone
from pathlib import Path


def parse_receiver_clock_offsets(clk_gz_path, station):
    """
    Extract receiver clock offset records from a RINEX CLK file.

    RINEX CLK `AR` rows give station receiver clock offsets relative to the
    product reference timescale. The offset is reported in seconds.
    """
    station = station.upper()
    rows = []

    with gzip.open(clk_gz_path, "rt", errors="replace") as handle:
        for line in handle:
            if not line.startswith("AR "):
                continue

            parts = line.split()
            if len(parts) < 10 or parts[1].upper() != station:
                continue

            year = int(parts[2])
            month = int(parts[3])
            day = int(parts[4])
            hour = int(parts[5])
            minute = int(parts[6])
            second = float(parts[7])
            sec_whole = int(second)
            
            # Simple RINEX Y2K handling
            yr = 2000 + year if year < 80 else year
            if yr < 100:
                yr += 1900
            
            nsec = int(round((second - sec_whole) * 1e9))
            if nsec < 0:
                nsec = 0

            # Construct Epoch in GPST (GPS Time scale - standard for RINEX clock files)
            from hifitime import Epoch, TimeScale
            epoch_gps = Epoch.from_gregorian(yr, month, day, hour, minute, sec_whole, nsec, TimeScale.GPST)
            # Convert to UTC
            epoch_utc = epoch_gps.to_time_scale(TimeScale.UTC)
            epoch = epoch_utc.to_datetime()


            offset_s = float(parts[9])
            sigma_s = float(parts[10]) if len(parts) > 10 else None
            rows.append(
                {
                    "epoch_utc": epoch.isoformat().replace("+00:00", "Z"),
                    "station": station,
                    "receiver_clock_offset_s": offset_s,
                    "receiver_clock_offset_ns": offset_s * 1e9,
                    "sigma_s": sigma_s,
                    "sigma_ns": sigma_s * 1e9 if sigma_s is not None else "",
                }
            )

    return rows


def write_receiver_clock_csv(clk_gz_path, station, output_csv_path):
    rows = parse_receiver_clock_offsets(clk_gz_path, station)
    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "epoch_utc",
        "station",
        "receiver_clock_offset_s",
        "receiver_clock_offset_ns",
        "sigma_s",
        "sigma_ns",
    ]

    with output_csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return rows


if __name__ == "__main__":
    rows = write_receiver_clock_csv(
        "GNSS_DATA/ESA0OPSFIN_20251000000_01D_30S_CLK.CLK.gz",
        "ALGO",
        "data/algo_receiver_clock_truth_2025100.csv",
    )
    if not rows:
        raise SystemExit("No receiver clock rows found.")

    print(f"Extracted {len(rows)} receiver-clock truth rows.")
    print(f"First epoch: {rows[0]['epoch_utc']} ({rows[0]['receiver_clock_offset_ns']:.3f} ns)")
    print(f"Last epoch:  {rows[-1]['epoch_utc']} ({rows[-1]['receiver_clock_offset_ns']:.3f} ns)")
