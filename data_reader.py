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
    """
    file_path = CONFIG['adsb']['json_file_path']

    # Tunables
    MAX_FILE_STALENESS_S = 60
    MAX_AIRCRAFT_STALENESS_S = 10
    RETRY_ON_DECODE_MS = 50

    def _load_once():
        with open(file_path, 'r') as f:
            return json.load(f)

    try:
        try:
            data = _load_once()
        except json.JSONDecodeError:
            time.sleep(RETRY_ON_DECODE_MS / 1000.0)
            data = _load_once()

        now_epoch = time.time()
        file_time = _to_float(data.get('now', now_epoch))
        if file_time is None or abs(now_epoch - file_time) > MAX_FILE_STALENESS_S:
            print(f"Warning: Stale or invalid timestamp in {os.path.basename(file_path)}. Skipping.")
            return {}

        aircraft_dict = {}
        for ac in data.get('aircraft', []):
            icao_raw = ac.get('hex')
            icao = (str(icao_raw).strip().lower() if icao_raw else None)
            if not icao:
                continue

            seen_pos = _to_float(ac.get('seen_pos'))
            age = seen_pos if seen_pos is not None else _to_float(ac.get('seen'))
            if age is None or age < 0 or age > MAX_AIRCRAFT_STALENESS_S:
                continue

            lat = _to_float(ac.get('lat'))
            lon = _to_float(ac.get('lon'))
            gs = _to_float(ac.get('gs'))
            track = _to_float(ac.get('track'))
            if track == 360.0:
                track = 0.0

            alt = None
            for k in ('altitude', 'alt_geom', 'alt_baro'):
                alt_val = _to_float(ac.get(k))
                if alt_val is not None:
                    alt = alt_val
                    break

            if not (_in_range(lat, -90, 90) and
                    _in_range(lon, -180, 180) and
                    _in_range(track, 0, 360) and
                    _in_range(alt, -2000, 60000) and
                    _in_range(gs, 0, 1200)):
                continue

            vr = _to_float(ac.get('baro_rate'))
            if vr is None:
                vr = _to_float(ac.get('vert_rate'))

            flight = str(ac.get('flight') or '').strip()

            # The actual time the aircraft's state was measured
            sample_time = file_time - age

            aircraft_dict[icao] = {
                'lat': lat,
                'lon': lon,
                'gs': gs,                  # knots
                'track': track,             # degrees
                'alt': alt,                 # feet
                'vert_rate': vr,            # ft/min
                'timestamp': sample_time,   # Corrected sample time
                'flight': flight,
                'age_s': age
            }

        return aircraft_dict

    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    except Exception as e:
        print(f"ERROR: Unexpected error in read_aircraft_data: {e}")
        return {}
