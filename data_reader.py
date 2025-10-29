#data_reader.py
"""
Module for reading and parsing aircraft.json from dump1090.
"""

import json
import time
import os
from config_loader import CONFIG

def _to_float(v):
    """Safely convert a value to a float, returning None on failure."""
    try:
        return float(v)
    except (ValueError, TypeError):
        return None

def _in_range(v, lo, hi):
    """Check if a value is a number and within a given range."""
    return v is not None and lo <= v <= hi

def read_aircraft_data() -> dict:
    """
    Reads aircraft.json, validates timestamp and per-aircraft freshness,
    coerces numeric fields, and returns a sanitized dict keyed by lowercased ICAO.
    Relies *only* on timestamps within the JSON file.
    """
    file_path = CONFIG['adsb']['json_file_path']

    # Tunables
    MAX_AIRCRAFT_STALENESS_S = 10 # Keep this for filtering aircraft
    RETRY_ON_DECODE_MS = 50

    def _load_once():
        with open(file_path, 'r', encoding='utf-8') as f: # Specify encoding
            return json.load(f)

    try:
        try:
            data = _load_once()
        except json.JSONDecodeError:
            print(f"Warning: JSONDecodeError reading {os.path.basename(file_path)}. Retrying...")
            time.sleep(RETRY_ON_DECODE_MS / 1000.0)
            data = _load_once() # Retry load

        # Get the 'now' timestamp from *within* the file.
        file_time = _to_float(data.get('now'))
        if file_time is None:
             print(f"Warning: Missing or invalid 'now' timestamp in {os.path.basename(file_path)}. Skipping.")
             return {}
        # We no longer compare file_time to time.time(). We trust the file's 'now'.

        aircraft_dict = {}
        processed_count = 0
        skipped_stale = 0
        skipped_invalid = 0
        skipped_alt = 0

        for ac in data.get('aircraft', []):
            icao_raw = ac.get('hex')
            icao = (str(icao_raw).strip().lower() if icao_raw else None)
            if not icao:
                skipped_invalid += 1
                continue

            # Calculate age relative to the file's 'now' timestamp
            seen_pos = _to_float(ac.get('seen_pos'))
            seen = _to_float(ac.get('seen'))
            age = None
            if seen_pos is not None and seen_pos >= 0:
                 age = seen_pos
            elif seen is not None and seen >= 0:
                 age = seen

            # Check age validity against MAX_AIRCRAFT_STALENESS_S
            if age is None or age > MAX_AIRCRAFT_STALENESS_S:
                skipped_stale += 1
                continue

            # Extract and validate fields (using only alt_geom as per previous fix)
            lat = _to_float(ac.get('lat'))
            lon = _to_float(ac.get('lon'))
            gs = _to_float(ac.get('gs'))
            track = _to_float(ac.get('track'))
            if track == 360.0: track = 0.0
            alt = _to_float(ac.get('alt_geom')) # Only geometric altitude
            vr = _to_float(ac.get('baro_rate')) or _to_float(ac.get('vert_rate')) # Fallback for vertical rate
            flight = str(ac.get('flight') or '').strip()

            # --- REVERT: Remove NUCp Reading ---
            # nucp = int(ac.get('nucp', 0) or 0) # <- REMOVED
            # --- END REVERT ---

            # Combined validation check
            is_valid = True
            if not _in_range(lat, -90, 90): is_valid = False
            if not _in_range(lon, -180, 180): is_valid = False
            if not _in_range(track, 0, 359.999): is_valid = False # Allow 0-359.999
            if alt is None: # Check if alt_geom was missing/invalid
                 is_valid = False
                 skipped_alt += 1 # Count specifically why it was skipped
            elif not _in_range(alt, -2000, 80000): # Allow higher altitude? Check reasonable max.
                 is_valid = False
            if not _in_range(gs, 0, 1200): is_valid = False

            if not is_valid:
                skipped_invalid += 1
                continue

            # Calculate the absolute sample time based on the file's timestamp and the age
            sample_time = file_time - age

            aircraft_dict[icao] = {
                'lat': lat,
                'lon': lon,
                'gs': gs,              # knots
                'track': track,        # degrees
                'alt': alt,            # feet (geometric)
                'vert_rate': vr,       # ft/min (can be None)
                'timestamp': sample_time, # Absolute epoch time of measurement
                'flight': flight,
                'age_s': age,          # Age relative to file_time
                # --- REVERT: Remove NUCp Field ---
                # 'nucp': nucp # <- REMOVED
                # --- END REVERT ---
            }
            processed_count += 1

        return aircraft_dict

    except FileNotFoundError:
        print(f"Warning: ADS-B file not found: {file_path}")
        return {}
    except json.JSONDecodeError as e:
         print(f"ERROR: Failed to decode JSON from {file_path}: {e}")
         return {} # Return empty on decode error after retry
    except Exception as e:
        print(f"ERROR: Unexpected error in read_aircraft_data: {e}")
        return {}