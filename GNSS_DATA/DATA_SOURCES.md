# GNSS Data Sources and Validation

This project currently uses the historical RINEX observation file:

- Local file: `data/ab041000.25o`
- Station: `AB04`
- Observation day: `2025-04-10`
- Format: RINEX `2.11`, mixed GNSS observation data
- Time scale in header: GPS time, converted to UTC by `analysis/gnss_analysis.py`
- Primary use in the app: constellation history and live GNSSDO playback validation

## Historical GNSS Data

Use archived RINEX observation files for repeatable validation runs. Good sources:

- International GNSS Service data portal: https://www.igs.org/data/
- IGS data access overview: https://www.igs.org/data-access/
- BKG GNSS Data Center: https://igs.bkg.bund.de/
- NASA CDDIS GNSS archive: https://cddis.nasa.gov/archive/gnss/data/

Recommended historical validation approach:

1. Select one station and one UTC day.
2. Download the daily observation RINEX file for that station/day.
3. Store the raw file under `GNSS_DATA/`.
4. Copy or point `config.RINEX_FILEPATH` to the file used by the app.
5. Run the app and verify epoch count, observation start time, satellite counts, and playback behavior.

## Real-Time GNSS Data

Use NTRIP/RTCM streams for real-time validation when receiver/network credentials are available.

Good references:

- IGS Real-Time Service: https://www.igs.org/rts
- BKG NTRIP information: https://igs.bkg.bund.de/ntrip/index
- GAGE real-time GNSS streams: https://www.unavco.org/data/gps-gnss/real-time/real-time.html

Real-time validation path:

1. Connect to a real-time NTRIP mountpoint for a station or nearby GNSS receiver.
2. Decode RTCM/BINEX to epochs or convert rolling data to RINEX.
3. Feed epoch visibility and timing quality into the same validation logic used by the historical RINEX playback.
4. Compare the disciplined master clock error, Kalman innovation, holdover behavior, and Allan deviation against the archived replay.

Without NTRIP credentials, the app's "Live GNSSDO Playback" tab is still a valid real-time emulator: it replays historical RINEX epochs at a controlled rate and validates the GNSSDO logic against real observed constellation history.
