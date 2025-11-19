# aircraft_selector.py
"""
Module for selecting the best aircraft to track using a Hybrid Scoring Model.
Prioritizes Distance and Closure Rate, with strict hard filters for Range, Elevation, and Sun Safety.
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


def _observer_location_from_config() -> EarthLocation:
    """Helper to get observer location from config."""
    obs = CONFIG['observer']
    return EarthLocation(
        lat=obs['latitude_deg'] * u.deg,
        lon=obs['longitude_deg'] * u.deg,
        height=obs['altitude_m'] * u.m
    )


def calculate_expected_value(current_az_el: tuple, icao: str, aircraft_data: dict) -> dict:
    """
    Calculates the Expected Value (EV) of tracking an aircraft based on:
    1. Hard Filters: Elevation, Max Range, Intercept Time, Future Sun Separation.
    2. Weighted Score: Distance (with receding penalty), Closure Rate, Intercept Time.
    """
    # -------------------------------------------------------------------------
    # 1. Load Configuration
    # -------------------------------------------------------------------------
    obs_cfg = CONFIG['observer']
    hw_cfg = CONFIG['hardware']
    sel_cfg = CONFIG['selection']
    ev_cfg = CONFIG.get('ev', {})

    # Hard Filters
    min_el = float(sel_cfg.get('min_elevation_deg', 5.0))
    max_range_km = float(sel_cfg.get('max_range_km', 100.0))
    min_sun_sep = float(sel_cfg.get('min_sun_separation_deg', 15.0))
    sun_lookahead = float(sel_cfg.get('sun_check_lookahead_s', 30.0))
    
    # Scoring Parameters
    receding_penalty_pct = float(sel_cfg.get('receding_penalty_percent', 20.0))
    max_closure_rate = float(sel_cfg.get('max_closure_rate_m_s', 300.0))
    horizon_s = float(ev_cfg.get('horizon_s', 180.0))
    weights = sel_cfg.get('weights', {})

    observer_loc = EarthLocation(
        lat=obs_cfg['latitude_deg'] * u.deg,
        lon=obs_cfg['longitude_deg'] * u.deg,
        height=obs_cfg['altitude_m'] * u.m
    )
    now = time.time()
    frame_now = get_altaz_frame(observer_loc)

    # -------------------------------------------------------------------------
    # 2. Trajectory Prediction Helper
    # -------------------------------------------------------------------------
    @lru_cache(maxsize=None)
    def predictor(t: float) -> Optional[dict]:
        """Predicts aircraft state at relative time t (seconds from now)."""
        pred_time = now + t
        pos_list = estimate_positions_at_times(aircraft_data, [pred_time])
        if not pos_list: return None
        pos = pos_list[0]
        
        alt_ft = pos.get('est_alt')
        if alt_ft is None: return None

        try:
            az, el = latlonalt_to_azel(
                pos['est_lat'], pos['est_lon'], alt_ft, pred_time, observer_loc)
            if not (np.isfinite(az) and np.isfinite(el)): return None
        except Exception:
            return None

        # Calculate distance for scoring
        dist_k = distance_km(
            obs_cfg['latitude_deg'], obs_cfg['longitude_deg'],
            pos['est_lat'], pos['est_lon']
        )
        
        # Sun pos at this specific time
        sun_az, sun_el = get_sun_azel(pred_time, observer_loc)
        
        return {
            "az": az, "el": el,
            "sun_az": sun_az, "sun_el": sun_el,
            "time": pred_time,
            "lat": pos['est_lat'], "lon": pos['est_lon'],
            "range_km": dist_k
        }

    # -------------------------------------------------------------------------
    # 3. Solve Intercept
    # -------------------------------------------------------------------------
    def target_azel_func(t):
        s = predictor(t)
        return None if s is None else (s['az'], s['el'])

    max_slew_rate = hw_cfg.get('max_slew_deg_s', 6.0)
    intercept_time = solve_intercept_time(
        current_az_el, target_azel_func, max_slew_rate, frame_now
    )

    # FILTER 1: Intercept Time (Pass/Fail)
    if intercept_time is None or intercept_time > horizon_s:
        return {'icao': icao, 'ev': 0.0, 'reason': 'intercept_impossible_or_too_late'}

    # -------------------------------------------------------------------------
    # 4. Calculate State at Intercept
    # -------------------------------------------------------------------------
    # Add a small margin (e.g. 5s) to ensure the mount has settled
    track_start_rel = intercept_time + float(ev_cfg.get('start_margin_s', 5.0))
    
    state_start = predictor(track_start_rel)
    if state_start is None:
        return {'icao': icao, 'ev': 0.0, 'reason': 'prediction_failed'}

    # -------------------------------------------------------------------------
    # 5. Apply Hard Filters
    # -------------------------------------------------------------------------
    
    # FILTER 2: Elevation (Pass/Fail)
    if state_start['el'] < min_el:
        return {'icao': icao, 'ev': 0.0, 'reason': f"low_elevation ({state_start['el']:.1f} < {min_el})"}

    # FILTER 3: Range (Pass/Fail)
    if state_start['range_km'] > max_range_km:
        return {'icao': icao, 'ev': 0.0, 'reason': f"out_of_range ({state_start['range_km']:.1f} > {max_range_km})"}

    # FILTER 4: Future Sun Separation (Pass/Fail)
    # Check sun separation 'sun_lookahead' seconds into the future
    state_future = predictor(track_start_rel + sun_lookahead)
    if state_future:
        frame_future = AltAz(obstime=Time(state_future['time'], format='unix'), location=observer_loc)
        sep_future = angular_sep_deg(
            (state_future['az'], state_future['el']),
            (state_future['sun_az'], state_future['sun_el']),
            frame_future
        )
        if sep_future < min_sun_sep:
             return {'icao': icao, 'ev': 0.0, 'reason': f"future_sun_conflict (<{min_sun_sep} in {sun_lookahead}s)"}
    
    # Also check current sun separation just to be safe
    frame_start = AltAz(obstime=Time(state_start['time'], format='unix'), location=observer_loc)
    sep_start = angular_sep_deg(
        (state_start['az'], state_start['el']),
        (state_start['sun_az'], state_start['sun_el']),
        frame_start
    )
    if sep_start < min_sun_sep:
         return {'icao': icao, 'ev': 0.0, 'reason': "current_sun_conflict"}


    # -------------------------------------------------------------------------
    # 6. Calculate Metrics (Closure, Ang Speed)
    # -------------------------------------------------------------------------
    # Calculate change over 1 second to get rates
    state_plus_1s = predictor(track_start_rel + 1.0)
    
    closure_rate_m_s = 0.0
    ang_speed = 0.0
    
    if state_plus_1s:
        # Closure Rate: (Distance Start - Distance End) / dt
        # Positive = Approaching, Negative = Leaving
        dist_diff_km = state_start['range_km'] - state_plus_1s['range_km']
        closure_rate_m_s = (dist_diff_km * 1000.0) # Convert km to m

        # Angular Speed (Metric only, not filtered)
        ang_speed = angular_speed_deg_s(
            (state_start['az'], state_start['el']),
            (state_plus_1s['az'], state_plus_1s['el']),
            1.0, frame_start
        )

    # Store metrics in state for debugging/logging
    state_start['closure_rate'] = closure_rate_m_s
    state_start['sun_sep'] = sep_start
    state_start['ang_speed'] = ang_speed

    # -------------------------------------------------------------------------
    # 7. Scoring Logic
    # -------------------------------------------------------------------------
    
    # A. Distance Score with Receding Penalty
    eff_dist = state_start['range_km']
    if closure_rate_m_s < 0:
        # Apply penalty if receding: add X% to effective distance
        eff_dist *= (1.0 + (receding_penalty_pct / 100.0))
    
    # Normalize: 1.0 at 0km, 0.0 at max_range_km.
    score_dist = max(0.0, 1.0 - (eff_dist / max_range_km))

    # B. Closure Rate Score (Bonus)
    # Normalize against max_closure_rate.
    # Clamp between 0 (receding/stationary) and 1 (max speed approach).
    score_closure = max(0.0, min(1.0, closure_rate_m_s / max_closure_rate))

    # C. Intercept Time Score (Weighted)
    # Shorter wait is better. 1.0 at 0s wait, 0.0 at horizon.
    score_int = max(0.0, 1.0 - (intercept_time / horizon_s))

    # -------------------------------------------------------------------------
    # 8. Weighted Sum & Detail Packaging
    # -------------------------------------------------------------------------
    w_dist = float(weights.get('distance', 0.5))
    w_close = float(weights.get('closure_rate', 0.3))
    w_int = float(weights.get('intercept_time', 0.2))

    contrib_dist = score_dist * w_dist
    contrib_close = score_closure * w_close
    contrib_int = score_int * w_int

    final_ev = contrib_dist + contrib_close + contrib_int

    return {
        'icao': icao,
        'ev': final_ev,
        'intercept_time': intercept_time,
        'start_state': state_start,
        'score_details': {
            'contrib_dist': contrib_dist,
            'contrib_close': contrib_close,
            'contrib_int': contrib_int,
            'raw_range_km': state_start['range_km'],
            'raw_closure_ms': closure_rate_m_s,
            'raw_int_s': intercept_time,
            'raw_el': state_start['el']
        }
    }


def select_aircraft(aircraft_dict: dict, current_mount_az_el: tuple) -> list:
    """
    Evaluates all aircraft and returns a list sorted by Expected Value.
    Applies a loose pre-filter to save CPU cycles before full prediction.
    """
    candidates = []
    
    sel_cfg = CONFIG['selection']
    min_alt_ft = float(sel_cfg.get('min_altitude_ft', 1000.0))
    # Use a loose range filter (e.g., 1.5x limit) for initial culling
    max_range_loose_km = float(sel_cfg.get('max_range_km', 100.0)) * 1.5 

    for icao, data in aircraft_dict.items():
        # 1. Altitude Pre-filter
        alt = data.get('alt')
        if alt is None or alt < min_alt_ft:
            continue

        # 2. Loose Range Pre-filter
        lat = data.get('lat')
        lon = data.get('lon')
        if lat is None or lon is None:
            continue
            
        dist_km = distance_km(
            CONFIG['observer']['latitude_deg'],
            CONFIG['observer']['longitude_deg'],
            lat, lon
        )
        if dist_km > max_range_loose_km:
            continue

        # 3. Full EV Calculation
        try:
            result = calculate_expected_value(current_mount_az_el, icao, data)
            if result.get('ev', 0) > 0:
                candidates.append(result)
        except Exception as e:
            logger.warning(f"EV calc error for {icao}: {e}")
            continue

    # Sort by EV descending
    candidates.sort(key=lambda x: x.get('ev', 0), reverse=True)
    return candidates


def evaluate_manual_target_viability(
    icao: str,
    aircraft_dict: Dict[str, Dict[str, Any]],
    observer_loc: Optional[EarthLocation] = None,
    now: Optional[float] = None
) -> Tuple[bool, List[str], Dict[str, Any]]:
    """
    Evaluate if a manually selected target is viable *right now* based on the new 
    configuration limits (km range, sun separation, etc.).
    """
    now = now or time.time()
    if observer_loc is None:
        observer_loc = _observer_location_from_config()

    reasons: List[str] = []
    details: Dict[str, Any] = {"icao": icao, "viable": False}

    # Load new config thresholds
    sel_cfg = CONFIG.get('selection', {})
    max_age_s = float(sel_cfg.get('manual_max_age_s', 15.0))
    min_el_sel = float(sel_cfg.get('min_elevation_deg', 5.0))
    min_sun_sep_deg = float(sel_cfg.get('min_sun_separation_deg', 15.0))
    max_range_km = float(sel_cfg.get('max_range_km', 100.0))

    hw_cfg = CONFIG.get('hardware', {})
    min_el_hw = float(hw_cfg.get('min_el_deg', min_el_sel))
    max_el_hw = float(hw_cfg.get('max_el_deg', 90.0))

    # Check existence
    ac = aircraft_dict.get(icao)
    if not ac:
        reasons.append("no ADS-B contact (not in latest feed)")
        details["present"] = False
        details["reasons"] = reasons
        return False, reasons, details

    details["present"] = True

    # Check Age
    age_s = ac.get('age_s')
    if age_s is not None:
        details["age_s"] = round(age_s, 1)
        if age_s > max_age_s:
            reasons.append(f"position too old ({age_s:.0f}s > {max_age_s:.0f}s)")
    else:
        reasons.append("no valid age")

    # Check Position & Range (KM)
    lat, lon = ac.get("lat"), ac.get("lon")
    if lat is None or lon is None:
        reasons.append("no position")
        return False, reasons, details

    dist_km = distance_km(
        CONFIG['observer']['latitude_deg'],
        CONFIG['observer']['longitude_deg'],
        lat, lon
    )
    
    if not np.isfinite(dist_km):
        reasons.append("distance calculation failed")
    else:
        details["range_km"] = round(dist_km, 1)
        if dist_km > max_range_km:
            reasons.append(f"outside range limit ({dist_km:.1f}km > {max_range_km:.1f}km)")

    # Check Altitude & Elevation
    alt_ft = ac.get("alt")
    if alt_ft is None:
        reasons.append("no altitude")
        return False, reasons, details

    try:
        az, el = latlonalt_to_azel(lat, lon, float(alt_ft), now, observer_loc)
        details["az_el"] = (round(az, 2), round(el, 2))

        min_el_req = max(min_el_sel, min_el_hw)
        if el < min_el_req:
            reasons.append(f"below min elevation ({el:.1f}° < {min_el_req:.1f}°)")
        if el > max_el_hw:
            reasons.append(f"above max elevation ({el:.1f}° > {max_el_hw:.1f}°)")

        # Sun Avoidance
        sun_az, sun_el = get_sun_azel(now, observer_loc)
        frame_now = AltAz(obstime=Time(now, format='unix'), location=observer_loc)
        sun_sep = angular_sep_deg((az, el), (sun_az, sun_el), frame_now)
        details["sun_sep_deg"] = round(sun_sep, 2)
        
        if sun_sep < min_sun_sep_deg:
            reasons.append(f"too close to Sun ({sun_sep:.1f}° < {min_sun_sep_deg:.1f}°)")

    except Exception as e:
        reasons.append(f"calculation failed: {e}")
        return False, reasons, details

    ok = len(reasons) == 0
    details["viable"] = ok
    details["reasons"] = reasons
    return ok, reasons, details