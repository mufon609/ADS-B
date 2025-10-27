#aircraft_selector.py
"""
Module for selecting the best aircraft to track using an Expected Value model.

Enhancements:
- Adds evaluate_manual_target_viability() which explains *why* a manual target
  is "not currently viable" (stale/no position, below elevation limits,
  too close to sun, beyond range, etc.). This does NOT change automatic EV
  selection behavior; it only provides visibility for manual overrides.
- Fixes distance unit handling: haversine_distance returns kilometers. We now
  convert to nautical miles (nm) only when comparing to nm thresholds or when
  presenting nm in details; internally the EV quality uses kilometers.
- Adds temporary debug visibility for the Bonus flag from config.yaml.
"""

from __future__ import annotations

import time
from functools import lru_cache
from typing import Dict, Any, Tuple, List, Optional

from astropy.coordinates import EarthLocation
import astropy.units as u

from coord_utils import (
    get_altaz_frame,
    latlonalt_to_azel,
    haversine_distance,       # returns kilometers
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
    if state['el'] < min_el:
        return 0.0
    q_el = min(1.0, max(0.0, (state['el'] - min_el) / 20.0))

    min_sun_sep = sel_cfg['min_sun_separation_deg']
    if state['sun_sep'] < min_sun_sep:
        return 0.0
    q_sun = min(1.0, max(0.0, (state['sun_sep'] - min_sun_sep) / 15.0))

    # NOTE: state['range_km'] should be kilometers
    q_dist = 1.0 / max(1.0, state.get('range_km', 50) / 10.0)
    q_speed = max(0.0, 1.0 - (state['ang_speed'] / sel_cfg['max_angular_speed_deg_s']))

    return (0.4 * q_el + 0.3 * q_sun + 0.1 * q_dist + 0.2 * q_speed)


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
    frame = get_altaz_frame(observer_loc)
    now = time.time()

    @lru_cache(maxsize=None)
    def predictor(t: float) -> Optional[dict]:
        pred_time = now + t
        pos_list = estimate_positions_at_times(aircraft_data, [pred_time])
        if not pos_list:
            return None
        pos = pos_list[0]
        az, el = latlonalt_to_azel(pos['est_lat'], pos['est_lon'], pos['est_alt'], pred_time, observer_loc)
        sun_az, sun_el = get_sun_azel(pred_time, observer_loc)
        return {
            "az": az, "el": el, "sun_az": sun_az, "sun_el": sun_el,
            "time": pred_time, "lat": pos['est_lat'], "lon": pos['est_lon']
        }

    def target_azel_func(t):
        s = predictor(t)
        return None if s is None else (s['az'], s['el'])

    intercept_time = solve_intercept_time(current_az_el, target_azel_func, hw_cfg['max_slew_deg_s'], frame)
    if intercept_time is None:
        return {'icao': icao, 'ev': 0, 'reason': 'no_intercept'}

    start_margin = ev_cfg.get('start_margin_s', 5.0)
    t_max = ev_cfg.get('horizon_s', 180.0)
    dt = ev_cfg.get('dt_s', 2.0)
    min_q = ev_cfg.get('min_quality', 0.1)

    if intercept_time + start_margin >= t_max:
        return {'icao': icao, 'ev': 0, 'reason': 'late_intercept'}

    t = intercept_time + start_margin
    ev = 0.0
    start_state = None
    
    while t < t_max:
        state = predictor(t)
        if state is None:
            break

        next_state = predictor(t + 1.0)
        if next_state is None:
            break

        # haversine_distance returns kilometers (km)
        dist_km = haversine_distance(
            obs_cfg['latitude_deg'],
            obs_cfg['longitude_deg'],
            state['lat'], state['lon']
        )
        state['range_km'] = dist_km
        state['sun_sep']  = angular_sep_deg(
            (state['az'], state['el']),
            (state['sun_az'], state['sun_el']),
            frame
        )
        state['ang_speed'] = angular_speed_deg_s(
            (state['az'], state['el']),
            (next_state['az'], next_state['el']),
            1.0, frame
        )
        
        quality = calculate_quality(state)
        if quality <= min_q:
            break
        
        if start_state is None:
            start_state = state
            
        ev += quality * dt
        t += dt

    if not start_state:
        return {'icao': icao, 'ev': 0, 'reason': 'low_quality'}

    return {
        'icao': icao,
        'ev': ev,
        'intercept_time': intercept_time,
        'slew_time': intercept_time, # Legacy, same as intercept_time
        'start_state': start_state
    }


def select_aircraft(aircraft_dict: dict, current_mount_az_el: tuple) -> list:
    """
    Evaluates all aircraft and returns a list sorted by Expected Value.
    Applies quick pre-filters on altitude and range.
    """
    candidates = []
    max_range_nm = float(CONFIG['selection']['max_range_nm'])
    max_range_km = max_range_nm * 1.852 # <-- FIX: Calculate km limit once
    min_alt_ft   = float(CONFIG['selection']['min_altitude_ft'])

    for icao, data in aircraft_dict.items():
        if data.get('alt', 0.0) < min_alt_ft:
            continue

        lat = data.get('lat')
        lon = data.get('lon')
        if lat is None or lon is None:
            # Can't prefilter by range without a position; still evaluate EV which
            # may use dead-reckoning if available.
            pass
        else:
            dist_km = haversine_distance(CONFIG['observer']['latitude_deg'],
                                         CONFIG['observer']['longitude_deg'],
                                         lat, lon)
            # dist_nm = dist_km / 1.852 # <-- FIX: DELETE this incorrect conversion
            if dist_km > max_range_km: # <-- FIX: Compare km to km limit
                continue

        result = calculate_expected_value(current_mount_az_el, icao, data)
        if result['ev'] > 0:
            candidates.append(result)
            
    candidates.sort(key=lambda x: x['ev'], reverse=True)
    return candidates


# ---------------------------------------------------------
# NEW: Manual-target viability evaluation with reasons
# ---------------------------------------------------------

def _observer_location_from_config() -> EarthLocation:
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

    Returns:
      ok: True if viable
      reasons: list of human-readable reasons when not viable
      details: metrics (age, range, az/el, sun sep, thresholds used)
    """
    now = now or time.time()
    if observer_loc is None:
        observer_loc = _observer_location_from_config()

    reasons: List[str] = []
    details: Dict[str, Any] = {"icao": icao, "viable": False}

    sel_cfg = CONFIG.get('selection', {})
    max_age_s        = float(sel_cfg.get('stale_after_s', 15.0))
    min_el_sel       = float(sel_cfg.get('min_elevation_deg', 10.0))
    min_sun_sep_deg  = float(sel_cfg.get('min_sun_separation_deg', 15.0))
    max_range_nm_cfg = float(sel_cfg.get('max_range_nm', 120.0))

    hw_cfg = CONFIG.get('hardware', {})
    min_el_hw  = float(hw_cfg.get('min_el_deg', min_el_sel))
    max_el_hw  = float(hw_cfg.get('max_el_deg', 90.0))

    ac = aircraft_dict.get(icao)
    if not ac:
        reasons.append("no ADS-B contact (not in latest feed)")
        details["present"] = False
        details["reasons"] = reasons
        return False, reasons, details

    details["present"] = True

    # Staleness (prefer explicit 'seen_pos' if present; otherwise try 'last_seen' or 'seen')
    age_s: Optional[float] = None
    if ac.get("seen_pos") is not None:
        try:
            age_s = float(ac["seen_pos"])
        except Exception:
            age_s = None
    if age_s is None and ac.get("last_seen") is not None:
        try:
            age_s = float(now - float(ac["last_seen"]))
        except Exception:
            age_s = None
    if age_s is None and ac.get("seen") is not None:
        try:
            age_s = float(ac["seen"])
        except Exception:
            age_s = None

    if age_s is not None:
        details["age_s"] = age_s
        if age_s > max_age_s:
            reasons.append(f"position too old ({age_s:.0f}s > {max_age_s:.0f}s)")
    else:
        reasons.append("no last_seen/seen_pos timestamp")

    lat = ac.get("lat")
    lon = ac.get("lon")
    if lat is None or lon is None:
        reasons.append("no position (lat/lon missing)")
        details["reasons"] = reasons
        return False, reasons, details

    # Range: convert km -> nm for comparison/output
    dist_km = haversine_distance(CONFIG['observer']['latitude_deg'],
                                 CONFIG['observer']['longitude_deg'],
                                 lat, lon)
    dist_nm = dist_km / 1.852 # This conversion is correct now
    details["range_nm"] = round(dist_nm, 1)
    if dist_nm > max_range_nm_cfg:
        reasons.append(f"outside range limit ({dist_nm:.1f}nm > {max_range_nm_cfg:.1f}nm)")

    # Elevation / Sun separation now
    alt_ft = float(ac.get("alt", 0.0))
    alt_m  = alt_ft * 0.3048
    az, el = latlonalt_to_azel(lat, lon, alt_m, now, observer_loc)
    details["az_el"] = (round(az, 2), round(el, 2))

    # Elevation constraints (selection + hardware)
    min_el_req = max(min_el_sel, min_el_hw)
    if el < min_el_req:
        reasons.append(f"below min elevation ({el:.1f}° < {min_el_req:.1f}°)")
    if el > max_el_hw:
        reasons.append(f"above max elevation ({el:.1f}° > {max_el_hw:.1f}°)")

    # Sun avoidance
    try:
        sun_az, sun_el = get_sun_azel(now, observer_loc)
        frame = get_altaz_frame(observer_loc)
        sun_sep = angular_sep_deg((az, el), (sun_az, sun_el), frame)
        details["sun_sep_deg"] = round(sun_sep, 2)
        if sun_sep < min_sun_sep_deg:
            reasons.append(f"too close to Sun ({sun_sep:.1f}° < {min_sun_sep_deg:.1f}°)")
    except Exception:
        # Don't block if sun calc fails
        pass

    ok = len(reasons) == 0
    details["viable"] = ok
    details["reasons"] = reasons
    details["thresholds"] = {
        "stale_after_s": max_age_s,
        "min_elevation_deg_effective": min_el_req,
        "max_elevation_deg_hw": max_el_hw,
        "min_sun_separation_deg": min_sun_sep_deg,
        "max_range_nm": max_range_nm_cfg,
    }
    return ok, reasons, details