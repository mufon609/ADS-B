# data_reader.py
"""
Module for reading and parsing aircraft.json from dump1090.
"""

import json
import logging
import os
import time

from config_loader import CONFIG

logger = logging.getLogger(__name__)

def _to_float(v):
    """Safely convert a value to a float, returning None on failure."""
    try:
        # Handle special case "ground" for alt_baro
        if isinstance(v, str) and v.lower() == 'ground':
            return 0.0
        return float(v)
    except (ValueError, TypeError):
        return None

def _in_range(v, lo, hi):
    """Check if a value is a number and within a given range."""
    # Allow comparison with None implicitly handled by comparison operators
    try:
        return lo <= v <= hi
    except TypeError: # Catches comparison with None
        return False

def read_aircraft_data() -> dict:
    """
    Reads aircraft.json, validates timestamp and per-aircraft freshness,
    coerces numeric fields, and returns a sanitized dict keyed by lowercased ICAO.
    Relies *only* on timestamps within the JSON file.
    Uses alt_geom primarily, falls back to alt_baro if alt_geom is invalid/missing.
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
            logger.warning(f"Warning: JSONDecodeError reading {os.path.basename(file_path)}. Retrying...")
            time.sleep(RETRY_ON_DECODE_MS / 1000.0)
            data = _load_once() # Retry load

        # Get the 'now' timestamp from *within* the file.
        file_time = _to_float(data.get('now'))
        if file_time is None:
            logger.warning(f"Warning: Missing or invalid 'now' timestamp in {os.path.basename(file_path)}. Skipping.")
            return {}
        # We no longer compare file_time to time.time(). We trust the file's 'now'.

        aircraft_dict = {}

        for ac in data.get('aircraft', []):
            icao_raw = ac.get('hex')
            icao = (str(icao_raw).strip().lower() if icao_raw else None)
            if not icao:
                continue

            # Calculate age relative to the file's 'now' timestamp
            seen_pos = _to_float(ac.get('seen_pos'))
            seen = _to_float(ac.get('seen'))
            age = None
            if (seen_pos is not None) and (seen_pos >= 0):
                 age = seen_pos
            elif (seen is not None) and (seen >= 0):
                 age = seen

            # Check age validity against MAX_AIRCRAFT_STALENESS_S
            if age is None or age > MAX_AIRCRAFT_STALENESS_S:
                continue

            # Extract and validate fields
            lat = _to_float(ac.get('lat'))
            lon = _to_float(ac.get('lon'))
            gs = _to_float(ac.get('gs'))
            track = _to_float(ac.get('track'))
            if track == 360.0:
                track = 0.0

            # --- ALTITUDE FALLBACK LOGIC ---
            alt_geom_val = _to_float(ac.get('alt_geom'))
            alt_baro_val = _to_float(ac.get('alt_baro')) # Handles "ground" -> 0.0

            alt = None
            if alt_geom_val is not None:
                alt = alt_geom_val # Prefer geometric
            elif alt_baro_val is not None:
                alt = alt_baro_val # Fallback to barometric
            # If both are None, 'alt' remains None
            # --- END ALTITUDE FALLBACK ---

            vr = (_to_float(ac.get('baro_rate'))
                  or _to_float(ac.get('vert_rate')))  # Fallback for vertical rate
            flight = str(ac.get('flight') or '').strip()


            # Combined validation check
            if not _in_range(lat, -90, 90):
                continue
            if not _in_range(lon, -180, 180):
                continue
            if not _in_range(track, 0, 359.999):
                continue
            if alt is None:
                continue
            if not _in_range(alt, -2000, 80000):
                continue
            if not _in_range(gs, 0, 1200):
                continue

            # Calculate the absolute sample time based on the file's timestamp and the age
            sample_time = file_time - age

            aircraft_dict[icao] = {
                'lat': lat,
                'lon': lon,
                'gs': gs,              # knots
                'track': track,        # degrees
                'alt': alt,            # feet (geometric preferred, baro fallback)
                'vert_rate': vr,       # ft/min (can be None)
                'timestamp': sample_time, # Absolute epoch time of measurement
                'flight': flight,
                'age_s': age,          # Age relative to file_time
            }

        return aircraft_dict

    except FileNotFoundError:
        logger.warning(f"Warning: ADS-B file not found: {file_path}")
        return {}
    except json.JSONDecodeError as e:
        logger.error(f"ERROR: Failed to decode JSON from {file_path}: {e}")
        return {}  # Return empty on decode error after retry
    except Exception as e:
        logger.error(f"ERROR: Unexpected error in read_aircraft_data: {e}")
        # Optionally re-raise or log traceback here
        # import traceback
        # traceback.print_exc()
        return {}
