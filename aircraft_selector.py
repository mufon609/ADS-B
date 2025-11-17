# aircraft_selector.py
"""
Module for selecting the best aircraft to track using an Expected Value model.

This version implements a distance‑only scoring mode: elevation, Sun separation and
angular speed act purely as pass/fail filters; the quality score is derived from
distance alone.  We also define a constant USE_DISTANCE_ONLY to make it clear
that the distance-only logic is hard-coded.
"""

from __future__ import annotations

import logging
import time
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import astropy.units as u
import numpy as np
from astropy.coordinates import AltAz, EarthLocation
from astropy.time import Time

from config_loader import CONFIG
from coord_utils import (
    angular_sep_deg,
    angular_speed_deg_s,
    distance_km,
    get_altaz_frame,
    get_sun_azel,
    latlonalt_to_azel,
    solve_intercept_time,
)
from dead_reckoning import estimate_positions_at_times

logger = logging.getLogger(__name__)

# When USE_DISTANCE_ONLY is true, quality is based solely on distance.
# Other factors serve as pass/fail checks.
USE_DISTANCE_ONLY = True


def calculate_quality(state: dict) -> float:
    """
    Calculates a quality score for a given aircraft state.  Elevation, sun separation
    and angular speed act as pass/fail filters based on thresholds in
    CONFIG['selection'].  If USE_DISTANCE_ONLY is True, only distance contributes
    to the score; otherwise the score is a weighted sum using weights defined
    in CONFIG['selection']['weights'] (if present).
    """
    sel_cfg = CONFIG['selection']

    # Pass/fail on minimum elevation
    min_el = float(sel_cfg.get('min_elevation_deg', 10.0))
    current_el = state.get('el')
    if current_el is None or current_el < min_el:
        return 0.0

    # Pass/fail on minimum sun separation
    min_sun_sep = float(sel_cfg.get('min_sun_separation_deg', 15.0))
    current_sun_sep = state.get('sun_sep')
    if current_sun_sep is None or current_sun_sep < min_sun_sep:
        return 0.0

    # Pass/fail on maximum angular speed
    max_ang_speed = float(sel_cfg.get('max_angular_speed_deg_s', 1.0))
    current_ang_speed = state.get('ang_speed', float('inf'))
    if max_ang_speed > 0 and current_ang_speed > max_ang_speed:
        return 0.0

    # Distance quality (higher score for closer targets)
    current_range_km = state.get('range_km', 1000.0)
    q_dist = 1.0 / max(1.0, current_range_km / 10.0)

    if USE_DISTANCE_ONLY:
        # Only distance influences the ranking
        return q_dist

    # If not distance-only, use weights from config (with fallbacks)
    weights = sel_cfg.get('weights', {})
    w_el = float(weights.get('elevation', 0.4))
    w_sun = float(weights.get('sun_separation', 0.3))
    w_dist = float(weights.get('distance', 0.1))
    w_speed = float(weights.get('angular_speed', 0.2))

    # Normalized elevation quality (0 at min_el, approaches 1 at min_el + 20)
    q_el = min(1.0, max(0.0, (current_el - min_el) / 20.0))
    # Normalized sun separation quality (0 at min_sun_sep, approaches 1 at min_sun_sep + 15)
    q_sun = min(1.0, max(0.0, (current_sun_sep - min_sun_sep) / 15.0))
    # Normalized angular speed quality (0 when > max, approaches 1 when near 0)
    # Avoid division by zero; if max_ang_speed <= 0, treat as 1.0
    effective_max_speed = max_ang_speed if max_ang_speed > 0 else 1.0
    q_speed = max(0.0, 1.0 - (current_ang_speed / effective_max_speed))

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
    frame = get_altaz_frame(observer_loc)  # Gets frame for Time.now()
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
            return None  # Cannot proceed without predicted altitude
        try:
            # latlonalt_to_azel expects geometric altitude
            az, el = latlonalt_to_azel(
                pos['est_lat'], pos['est_lon'], alt_ft, pred_time, observer_loc)
            if not (np.isfinite(az) and np.isfinite(el)):  # Check for NaN/Inf from conversion
                logger.warning(
                    f"Warning: Non-finite az/el ({az},{el}) from latlonalt_to_azel for {icao} at t={t:.1f}")
                return None
        except Exception as e:
            logger.warning(
                f"Warning: latlonalt_to_azel failed in predictor for {icao} at t={t:.1f}: {e}")
            return None  # Failed coordinate conversion
        sun_az, sun_el = get_sun_azel(pred_time, observer_loc)
        return {
            "az": az,
            "el": el,
            "sun_az": sun_az,
            "sun_el": sun_el,
            "time": pred_time,
            "lat": pos['est_lat'],
            "lon": pos['est_lon']
        }

    def target_azel_func(t):
        """Helper function for solve_intercept_time, returns (az, el) tuple."""
        s = predictor(t)
        return None if s is None else (s['az'], s['el'])

    # Solve for the time it takes for the slew to intercept the target's predicted path
    max_slew_rate = hw_cfg.get('max_slew_deg_s', 6.0)
    if max_slew_rate <= 0:
        max_slew_rate = 6.0  # Ensure positive rate
    intercept_time = solve_intercept_time(
        current_az_el, target_azel_func, max_slew_rate, frame)  # Pass frame

    if intercept_time is None:
        return {'icao': icao, 'ev': 0, 'reason': 'no_intercept'}

    # Define parameters for EV integration window
    # Time after intercept to start tracking
    start_margin = float(ev_cfg.get('start_margin_s', 5.0))
    # How far into the future to evaluate
    t_horizon = float(ev_cfg.get('horizon_s', 180.0))
    # Time step for integration
    dt = float(ev_cfg.get('dt_s', 2.0))
    # Minimum quality threshold to continue tracking
    min_q = float(ev_cfg.get('min_quality', 0.1))

    # Check if the intercept happens too late
    track_start_time_rel = intercept_time + start_margin
    if track_start_time_rel >= t_horizon:
        return {'icao': icao, 'ev': 0, 'reason': 'late_intercept'}

    # Compute expected value based solely on distance at the start of tracking
    # Predict the state at the track start time
    state = predictor(track_start_time_rel)
    if state is None:
        return {'icao': icao, 'ev': 0, 'reason': 'prediction_failed'}

    # Calculate additional state variables needed for quality function
    dist_km = distance_km(
        obs_cfg['latitude_deg'],
        obs_cfg['longitude_deg'],
        state['lat'], state['lon']
    )
    state['range_km'] = dist_km

    # Use frame valid for this specific time for separation calcs
    frame_t = AltAz(obstime=Time(
        state['time'], format='unix'), location=observer_loc)
    state['sun_sep'] = angular_sep_deg(
        (state['az'], state['el']),
        (state['sun_az'], state['sun_el']),
        frame_t  # Use frame_t
    )
    try:
        # Estimate angular speed over 1-second interval
        next_state = predictor(track_start_time_rel + 1.0)
        if next_state is not None:
            ang_speed = angular_speed_deg_s(
                (state['az'], state['el']),
                (next_state['az'], next_state['el']),
                1.0,
                frame_t
            )
            if not np.isfinite(ang_speed):
                ang_speed = float('inf')
        else:
            ang_speed = float('inf')
    except Exception as e:
        logger.warning(
            f"Warning: Angular speed calculation failed for {icao} at track start: {e}")
        ang_speed = float('inf')
    state['ang_speed'] = ang_speed

    # Compute quality for pass/fail checks and distance-based score
    quality = calculate_quality(state)
    if quality < min_q:
        return {'icao': icao, 'ev': 0, 'reason': 'low_quality'}

    # The EV is simply the quality (distance-based score)
    return {
        'icao': icao,
        'ev': quality,
        'intercept_time': intercept_time,
        'slew_time': intercept_time,  # Legacy alias
        'start_state': state
    }


def select_aircraft(aircraft_dict: dict, current_mount_az_el: tuple) -> list:
    """
    Evaluates all aircraft and returns a list sorted by Expected Value.
    Applies quick pre-filters on altitude and range. (NUCp filter removed)
    """
    candidates = []
    sel_cfg = CONFIG['selection']  # Get selection config once
    max_range_nm = float(sel_cfg.get('max_range_nm', 120.0))
    max_range_km = max_range_nm * 1.852  # Convert NM to KM
    min_alt_ft = float(sel_cfg.get('min_altitude_ft', 1000.0))

    for icao, data in aircraft_dict.items():
        # Pre-filter altitude (using alt_geom from data_reader fix)
        alt = data.get('alt')
        if alt is None or alt < min_alt_ft:
            continue

        # Pre-filter range (if current position available)
        lat = data.get('lat')
        lon = data.get('lon')
        if lat is not None and lon is not None:
            dist_km = distance_km(
                CONFIG['observer']['latitude_deg'],
                CONFIG['observer']['longitude_deg'],
                lat, lon
            )
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
            logger.warning(
                f"Warning: EV calculation failed unexpectedly for {icao}: {e}")
            continue  # Skip this aircraft if EV calculation fails

    # Sort candidates by EV score in descending order
    candidates.sort(key=lambda x: x.get('ev', 0), reverse=True)
    return candidates


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
    details: Dict[str, Any] = {"icao": icao,
                               "viable": False}  # Default to not viable

    # Load configuration thresholds
    sel_cfg = CONFIG.get('selection', {})
    max_age_s = float(sel_cfg.get('manual_max_age_s', 15.0)
                      )  # Stricter age for manual check?
    min_el_sel = float(sel_cfg.get('min_elevation_deg', 10.0))
    min_sun_sep_deg = float(sel_cfg.get('min_sun_separation_deg', 15.0))
    max_range_nm_cfg = float(sel_cfg.get('max_range_nm', 120.0))

    hw_cfg = CONFIG.get('hardware', {})
    min_el_hw = float(hw_cfg.get('min_el_deg', min_el_sel)
                      )  # Use hardware min if stricter
    max_el_hw = float(hw_cfg.get('max_el_deg', 90.0)
                      )      # Hardware max elevation

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
            reasons.append(
                f"position too old ({age_s:.0f}s > {max_age_s:.0f}s)")
    else:
        reasons.append("no valid age ('age_s' missing or invalid)")

    # Check for Position Data
    lat = ac.get("lat")
    lon = ac.get("lon")
    if lat is None or lon is None:
        reasons.append("no position (lat/lon missing)")
        details["reasons"] = reasons
        return False, reasons, details  # Cannot proceed without position

    # Check Range
    dist_km = distance_km(
        CONFIG['observer']['latitude_deg'],
        CONFIG['observer']['longitude_deg'],
        lat, lon
    )
    if not np.isfinite(dist_km):
        reasons.append("distance calculation failed")
        dist_nm = None
    else:
        dist_nm = dist_km / 1.852
        details["range_nm"] = round(dist_nm, 1)
        if dist_nm > max_range_nm_cfg:
            reasons.append(
                f"outside range limit ({dist_nm:.1f}nm > {max_range_nm_cfg:.1f}nm)")

    # Check Altitude Data (should be alt_geom)
    alt_ft = ac.get("alt")
    if alt_ft is None:
        reasons.append("no valid altitude ('alt' missing or invalid)")
        details["reasons"] = reasons
        return False, reasons, details

    # Calculate Current Az/El and Sun Separation
    try:
        alt_ft_float = float(alt_ft)  # Ensure float
        az, el = latlonalt_to_azel(lat, lon, alt_ft_float, now, observer_loc)
        if not (np.isfinite(az) and np.isfinite(el)):
            raise ValueError(f"Non-finite Az/El result ({az},{el})")
        details["az_el"] = (round(az, 2), round(el, 2))

        # Elevation constraints (check against effective min and hardware max)
        min_el_req = max(min_el_sel, min_el_hw)
        if el < min_el_req:
            reasons.append(
                f"below min elevation ({el:.1f}° < {min_el_req:.1f}°)")
        if el > max_el_hw:
            reasons.append(
                f"above max elevation ({el:.1f}° > {max_el_hw:.1f}°)")

        # Sun avoidance
        try:
            sun_az, sun_el = get_sun_azel(now, observer_loc)
            # Use frame valid for the 'now' timestamp
            frame_now = AltAz(obstime=Time(
                now, format='unix'), location=observer_loc)
            sun_sep = angular_sep_deg((az, el), (sun_az, sun_el), frame_now)
            details["sun_sep_deg"] = round(sun_sep, 2)
            if sun_sep < min_sun_sep_deg:
                reasons.append(
                    f"too close to Sun ({sun_sep:.1f}° < {min_sun_sep_deg:.1f}°)")
        except Exception as sun_e:
            logger.warning(
                # Log error
                f"Warning: Sun separation calculation failed during manual check: {sun_e}")
            details["sun_sep_deg"] = None  # Indicate failure
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
    }
    return ok, reasons, details
