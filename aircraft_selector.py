#aircraft_selector.py
"""
Module for selecting the best aircraft to track using an Expected Value model.

Enhancements:
- Adds evaluate_manual_target_viability() which explains *why* a manual target
  is "not currently viable" (stale/no position, below elevation limits,
  too close to sun, beyond range, etc.). This does NOT change automatic EV
  selection behavior; it only provides visibility for manual overrides.
- Fixes distance unit handling: distance_km returns kilometers. We now
  convert to nautical miles (nm) only when comparing to nm thresholds or when
  presenting nm in details; internally the EV quality uses kilometers.
- Adds temporary debug visibility for the Bonus flag from config.yaml.
"""

from __future__ import annotations

import time
from functools import lru_cache
from typing import Dict, Any, Tuple, List, Optional

from astropy.coordinates import EarthLocation, AltAz
from astropy.time import Time
import astropy.units as u
import numpy as np # Import numpy

from coord_utils import (
    get_altaz_frame,
    latlonalt_to_azel,
    distance_km,              # Use the new name
    get_sun_azel,
    solve_intercept_time,
    angular_sep_deg,
    angular_speed_deg_s,
)
from config_loader import CONFIG
from dead_reckoning import estimate_positions_at_times


# -------------------------------
# Existing EV model (unit-correct)
# -------------------------------

def calculate_quality(state: dict) -> float:
    """Calculates a quality score (0-1) for a given aircraft state."""
    sel_cfg = CONFIG['selection']

    min_el = sel_cfg.get('min_elevation_deg', 10.0)
    # Ensure elevation is valid before comparison
    current_el = state.get('el')
    if current_el is None or current_el < min_el:
        return 0.0
    # Normalize elevation quality (0 at min_el, approaches 1 at min_el + 20)
    q_el = min(1.0, max(0.0, (current_el - min_el) / 20.0))

    min_sun_sep = sel_cfg.get('min_sun_separation_deg', 15.0) # Get default from config if needed
    # Ensure sun separation is valid
    current_sun_sep = state.get('sun_sep')
    if current_sun_sep is None or current_sun_sep < min_sun_sep:
        return 0.0
    # Normalize sun separation quality (0 at min_sun_sep, approaches 1 at min_sun_sep + 15)
    q_sun = min(1.0, max(0.0, (current_sun_sep - min_sun_sep) / 15.0))

    # Normalize distance quality (higher score for closer targets)
    # state['range_km'] should be kilometers
    current_range_km = state.get('range_km', 1000.0) # Default far if missing
    q_dist = 1.0 / max(1.0, current_range_km / 10.0) # Score drops significantly beyond 10km? Tune divisor.

    # Normalize angular speed quality (higher score for slower targets)
    max_ang_speed = sel_cfg.get('max_angular_speed_deg_s', 1.0)
    current_ang_speed = state.get('ang_speed', float('inf')) # Default fast if missing
    if max_ang_speed <= 0: max_ang_speed = 1.0 # Avoid division by zero
    q_speed = max(0.0, 1.0 - (current_ang_speed / max_ang_speed))

    # Weighted combination of quality factors
    w_el = 0.4
    w_sun = 0.3
    w_dist = 0.1
    w_speed = 0.2
    return (w_el * q_el + w_sun * q_sun + w_dist * q_dist + w_speed * q_speed)


def calculate_expected_value(current_az_el: tuple, icao: str, aircraft_data: dict) -> dict:
    """Calculates the Expected Value (EV) of tracking an aircraft."""
    obs_cfg = CONFIG['observer']
    hw_cfg = CONFIG['hardware']
    ev_cfg = CONFIG.get('ev', {})
    observer_loc = EarthLocation(
        lat=obs_cfg['latitude_deg'] * u.deg,
        lon=obs_cfg['longitude_deg'] * u.deg,
        height=obs_cfg['altitude_m'] * u.m
    )
    frame = get_altaz_frame(observer_loc) # Gets frame for Time.now()
    now = time.time()

    @lru_cache(maxsize=None)
    def predictor(t: float) -> Optional[dict]:
        """Predicts aircraft state (Az/El, Sun Az/El, Lat/Lon, Time) at time now + t."""
        pred_time = now + t
        pos_list = estimate_positions_at_times(aircraft_data, [pred_time])
        if not pos_list:
            return None
        pos = pos_list[0]
        # Ensure alt is valid before passing to latlonalt_to_azel
        alt_ft = pos.get('est_alt')
        if alt_ft is None:
            return None # Cannot proceed without predicted altitude
        try:
            # latlonalt_to_azel expects geometric altitude
            az, el = latlonalt_to_azel(pos['est_lat'], pos['est_lon'], alt_ft, pred_time, observer_loc)
            if not (np.isfinite(az) and np.isfinite(el)): # Check for NaN/Inf from conversion
                 print(f"Warning: Non-finite az/el ({az},{el}) from latlonalt_to_azel for {icao} at t={t:.1f}")
                 return None
        except Exception as e:
             print(f"Warning: latlonalt_to_azel failed in predictor for {icao} at t={t:.1f}: {e}")
             return None # Failed coordinate conversion
        sun_az, sun_el = get_sun_azel(pred_time, observer_loc)
        return {
            "az": az, "el": el, "sun_az": sun_az, "sun_el": sun_el,
            "time": pred_time, "lat": pos['est_lat'], "lon": pos['est_lon']
        }

    def target_azel_func(t):
        """Helper function for solve_intercept_time, returns (az, el) tuple."""
        s = predictor(t)
        return None if s is None else (s['az'], s['el'])

    # Solve for the time it takes for the slew to intercept the target's predicted path
    max_slew_rate = hw_cfg.get('max_slew_deg_s', 6.0)
    if max_slew_rate <= 0: max_slew_rate = 6.0 # Ensure positive rate
    intercept_time = solve_intercept_time(current_az_el, target_azel_func, max_slew_rate, frame) # Pass frame

    if intercept_time is None:
        return {'icao': icao, 'ev': 0, 'reason': 'no_intercept'}

    # Define parameters for EV integration window
    start_margin = float(ev_cfg.get('start_margin_s', 5.0)) # Time after intercept to start tracking
    t_horizon = float(ev_cfg.get('horizon_s', 180.0))    # How far into the future to evaluate
    dt = float(ev_cfg.get('dt_s', 2.0))             # Time step for integration
    min_q = float(ev_cfg.get('min_quality', 0.1))       # Minimum quality threshold to continue tracking

    # Check if the intercept happens too late
    track_start_time_rel = intercept_time + start_margin
    if track_start_time_rel >= t_horizon:
        return {'icao': icao, 'ev': 0, 'reason': 'late_intercept'}

    # Integrate quality over the tracking window
    t = track_start_time_rel
    ev = 0.0
    start_state = None # Store the state at the beginning of the valid track

    while t < t_horizon:
        state = predictor(t)
        if state is None:
            if start_state is None:
                 return {'icao': icao, 'ev': 0, 'reason': 'prediction_failed'}
            else:
                 break # Exit loop, return EV accumulated so far

        next_state = predictor(t + 1.0) # Predict 1 second ahead for angular speed calc
        if next_state is None:
             if start_state is None:
                  return {'icao': icao, 'ev': 0, 'reason': 'prediction_failed'}
             else:
                  break # Exit loop, return EV accumulated so far

        # Calculate additional state variables needed for quality function
        dist_km = distance_km(
            obs_cfg['latitude_deg'],
            obs_cfg['longitude_deg'],
            state['lat'], state['lon']
        )
        state['range_km'] = dist_km

        # Use frame valid for this specific time 't' for separation calcs
        frame_t = AltAz(obstime=Time(state['time'], format='unix'), location=observer_loc)
        state['sun_sep'] = angular_sep_deg(
            (state['az'], state['el']),
            (state['sun_az'], state['sun_el']),
            frame_t # Use frame_t
        )
        try:
            ang_speed = angular_speed_deg_s(
                 (state['az'], state['el']),
                 (next_state['az'], next_state['el']),
                 1.0, # time delta is 1.0 second
                 frame_t # Use frame_t
             )
            if not np.isfinite(ang_speed): ang_speed = float('inf') # Handle non-finite results
        except Exception as e:
             print(f"Warning: Angular speed calculation failed for {icao} at t={t:.1f}: {e}")
             ang_speed = float('inf') # Treat as very fast if calc fails
        state['ang_speed'] = ang_speed


        quality = calculate_quality(state)
        if quality < min_q: # Use '<' instead of '<=' to allow tracking at exactly min_q?
            if start_state is None:
                 return {'icao': icao, 'ev': 0, 'reason': 'low_quality'}
            else:
                 break # Exit loop

        if start_state is None:
            start_state = state

        ev += quality * dt
        t += dt

    if start_state:
        return {
            'icao': icao,
            'ev': ev,
            'intercept_time': intercept_time,
            'slew_time': intercept_time, # Legacy alias
            'start_state': start_state   # State at the beginning of the track
        }
    else:
        final_reason = 'low_quality' if 'quality' in locals() and quality < min_q else 'unknown_ev_failure'
        return {'icao': icao, 'ev': 0, 'reason': final_reason}


def select_aircraft(aircraft_dict: dict, current_mount_az_el: tuple) -> list:
    """
    Evaluates all aircraft and returns a list sorted by Expected Value.
    Applies quick pre-filters on altitude and range. (NUCp filter removed)
    """
    candidates = []
    sel_cfg = CONFIG['selection'] # Get selection config once
    max_range_nm = float(sel_cfg.get('max_range_nm', 120.0))
    max_range_km = max_range_nm * 1.852 # Convert NM to KM
    min_alt_ft   = float(sel_cfg.get('min_altitude_ft', 1000.0))
    # --- REVERT: Remove NUCp Threshold ---
    # min_nucp     = int(sel_cfg.get('min_nucp', 7)) # <- REMOVED
    # --- END REVERT ---


    for icao, data in aircraft_dict.items():
        # Pre-filter altitude (using alt_geom from data_reader fix)
        alt = data.get('alt')
        if alt is None or alt < min_alt_ft:
            continue

        # --- REVERT: Remove NUCp Filter ---
        # nucp = data.get('nucp', 0) # <- REMOVED (data_reader no longer adds it)
        # if nucp < min_nucp:       # <- REMOVED
        #     continue              # <- REMOVED
        # --- END REVERT ---

        # Pre-filter range (if current position available)
        lat = data.get('lat')
        lon = data.get('lon')
        if lat is not None and lon is not None:
            dist_km = distance_km(CONFIG['observer']['latitude_deg'],
                                  CONFIG['observer']['longitude_deg'],
                                  lat, lon)
            if dist_km > max_range_km:
                continue

        # Calculate Expected Value
        try:
            result = calculate_expected_value(current_mount_az_el, icao, data)
            # Only add candidates with positive EV score
            if result.get('ev', 0) > 0:
                candidates.append(result)
        except Exception as e:
            # Catch errors during EV calculation for a single aircraft
            print(f"Warning: EV calculation failed unexpectedly for {icao}: {e}")
            continue # Skip this aircraft if EV calculation fails

    # Sort candidates by EV score in descending order
    candidates.sort(key=lambda x: x.get('ev', 0), reverse=True)
    return candidates


# ---------------------------------------------------------
# Manual-target viability evaluation with reasons
# ---------------------------------------------------------

def _observer_location_from_config() -> EarthLocation:
    """Helper to get observer location from config."""
    obs = CONFIG['observer']
    return EarthLocation(
        lat=obs['latitude_deg'] * u.deg,
        lon=obs['longitude_deg'] * u.deg,
        height=obs['altitude_m'] * u.m
    )


def evaluate_manual_target_viability(
    icao: str,
    aircraft_dict: Dict[str, Dict[str, Any]],
    observer_loc: Optional[EarthLocation] = None,
    now: Optional[float] = None
) -> Tuple[bool, List[str], Dict[str, Any]]:
    """
    Evaluate if a manually selected target is viable *right now*, and explain why not.
    """
    now = now or time.time()
    if observer_loc is None:
        observer_loc = _observer_location_from_config()

    reasons: List[str] = []
    details: Dict[str, Any] = {"icao": icao, "viable": False} # Default to not viable

    # Load configuration thresholds
    sel_cfg = CONFIG.get('selection', {})
    max_age_s        = float(sel_cfg.get('manual_max_age_s', 15.0)) # Stricter age for manual check?
    min_el_sel       = float(sel_cfg.get('min_elevation_deg', 10.0))
    min_sun_sep_deg  = float(sel_cfg.get('min_sun_separation_deg', 15.0))
    max_range_nm_cfg = float(sel_cfg.get('max_range_nm', 120.0))
    # --- REVERT: Remove NUCp Threshold ---
    # min_nucp_sel     = int(sel_cfg.get('min_nucp', 7)) # <- REMOVED
    # --- END REVERT ---

    hw_cfg = CONFIG.get('hardware', {})
    min_el_hw  = float(hw_cfg.get('min_el_deg', min_el_sel)) # Use hardware min if stricter
    max_el_hw  = float(hw_cfg.get('max_el_deg', 90.0))    # Hardware max elevation

    # Check if aircraft exists in the latest data
    ac = aircraft_dict.get(icao)
    if not ac:
        reasons.append("no ADS-B contact (not in latest feed)")
        details["present"] = False
        details["reasons"] = reasons
        return False, reasons, details

    details["present"] = True

    # Age Check (using pre-sanitized 'age_s')
    age_s = ac.get('age_s')
    if age_s is not None:
         details["age_s"] = round(age_s, 1)
         if age_s > max_age_s:
             reasons.append(f"position too old ({age_s:.0f}s > {max_age_s:.0f}s)")
    else:
         reasons.append("no valid age ('age_s' missing or invalid)")

    # --- REVERT: Remove NUCp Check ---
    # nucp = ac.get('nucp', 0) # <- REMOVED (data_reader no longer provides it)
    # details["nucp"] = nucp  # <- REMOVED
    # if nucp < min_nucp_sel: # <- REMOVED
    #     reasons.append(f"low GPS quality (NUCp {nucp} < {min_nucp_sel})") # <- REMOVED
    # --- END REVERT ---

    # Check for Position Data
    lat = ac.get("lat")
    lon = ac.get("lon")
    if lat is None or lon is None:
        reasons.append("no position (lat/lon missing)")
        details["reasons"] = reasons
        return False, reasons, details # Cannot proceed without position

    # Check Range
    dist_km = distance_km(CONFIG['observer']['latitude_deg'],
                          CONFIG['observer']['longitude_deg'],
                          lat, lon)
    if not np.isfinite(dist_km):
        reasons.append("distance calculation failed")
        dist_nm = None
    else:
        dist_nm = dist_km / 1.852
        details["range_nm"] = round(dist_nm, 1)
        if dist_nm > max_range_nm_cfg:
            reasons.append(f"outside range limit ({dist_nm:.1f}nm > {max_range_nm_cfg:.1f}nm)")

    # Check Altitude Data (should be alt_geom)
    alt_ft = ac.get("alt")
    if alt_ft is None:
        reasons.append("no valid altitude ('alt' missing or invalid)")
        details["reasons"] = reasons
        return False, reasons, details

    # Calculate Current Az/El and Sun Separation
    try:
        alt_ft_float = float(alt_ft) # Ensure float
        az, el = latlonalt_to_azel(lat, lon, alt_ft_float, now, observer_loc)
        if not (np.isfinite(az) and np.isfinite(el)):
             raise ValueError(f"Non-finite Az/El result ({az},{el})")
        details["az_el"] = (round(az, 2), round(el, 2))

        # Elevation constraints (check against effective min and hardware max)
        min_el_req = max(min_el_sel, min_el_hw)
        if el < min_el_req:
            reasons.append(f"below min elevation ({el:.1f}° < {min_el_req:.1f}°)")
        if el > max_el_hw:
            reasons.append(f"above max elevation ({el:.1f}° > {max_el_hw:.1f}°)")

        # Sun avoidance
        try:
            sun_az, sun_el = get_sun_azel(now, observer_loc)
            # Use frame valid for the 'now' timestamp
            frame_now = AltAz(obstime=Time(now, format='unix'), location=observer_loc)
            sun_sep = angular_sep_deg((az, el), (sun_az, sun_el), frame_now)
            details["sun_sep_deg"] = round(sun_sep, 2)
            if sun_sep < min_sun_sep_deg:
                reasons.append(f"too close to Sun ({sun_sep:.1f}° < {min_sun_sep_deg:.1f}°)")
        except Exception as sun_e:
            print(f"Warning: Sun separation calculation failed during manual check: {sun_e}") # Log error
            details["sun_sep_deg"] = None # Indicate failure
            pass

    except Exception as coord_e:
        reasons.append(f"coordinate conversion failed: {coord_e}")
        details["az_el"] = None
        details["reasons"] = reasons
        return False, reasons, details

    # Determine final viability
    ok = len(reasons) == 0
    details["viable"] = ok
    details["reasons"] = reasons
    # Include thresholds used for clarity
    details["thresholds"] = {
        "stale_after_s": max_age_s,
        "min_elevation_deg_effective": max(min_el_sel, min_el_hw),
        "max_elevation_deg_hw": max_el_hw,
        "min_sun_separation_deg": min_sun_sep_deg,
        "max_range_nm": max_range_nm_cfg,
        # --- REVERT: Remove NUCp from thresholds ---
        # "min_nucp": min_nucp_sel, # <- REMOVED
        # --- END REVERT ---
    }
    return ok, reasons, details