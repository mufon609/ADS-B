# dead_reckoning.py
"""
Module for predicting future aircraft positions using dead reckoning.

This module provides functionality to estimate an aircraft's future latitude,
longitude, and altitude based on its current state (position, speed, track,
and vertical rate) and a given time delta. It leverages the `geopy` library
for accurate geodesic calculations.
"""
import math
from typing import Any, Dict, List, Optional

from geopy.distance import geodesic
from geopy.point import Point

from config.loader import CONFIG


def _finite_or(default: Optional[float], *vals: Any) -> Optional[float]:
    """
    Returns the first finite float value from the provided arguments, else a default.

    This helper function iterates through `vals` and attempts to convert each
    to a float. If a finite float is found, it is returned immediately. If no
    finite float is found after checking all `vals`, the `default` value is returned.

    Args:
        default: The value to return if no finite float is found in `vals`.
        *vals: Variable arguments to check for a finite float value.

    Returns:
        The first finite float found, or the default value.
    """
    for v in vals:
        try:
            if v is not None and math.isfinite(float(v)):
                return float(v)
        except (TypeError, ValueError):
            pass
    return default


def _predict(start_lat: float, start_lon: float, start_alt: float, gs_kts: float, 
             track_deg: float, vert_rate_fpm: Optional[float], delta_seconds: float) -> Optional[Dict[str, float]]:
    """
    Predicts a single future position of an aircraft using dead reckoning.

    This function calculates the estimated latitude, longitude, and altitude
    of an aircraft after a specified time duration, given its current state.

    Args:
        start_lat: Initial latitude in degrees.
        start_lon: Initial longitude in degrees.
        start_alt: Initial altitude in feet.
        gs_kts: Ground speed in knots.
        track_deg: True track in degrees (0-360).
        vert_rate_fpm: Vertical rate in feet per minute (can be None).
        delta_seconds: Time difference in seconds for the prediction.

    Returns:
        A dictionary containing the predicted position (`est_lat`, `est_lon`,
        `est_alt`) if all necessary inputs are finite and provided, otherwise None.
    """
    if None in (start_lat, start_lon, start_alt, gs_kts, track_deg, delta_seconds):
        return None

    # Ensure all inputs are finite and castable to float, using None as initial default
    # if _finite_or should not provide a default here.
    start_lat = _finite_or(None, start_lat)
    start_lon = _finite_or(None, start_lon)
    start_alt = _finite_or(None, start_alt)
    gs_kts = _finite_or(None, gs_kts)
    track_deg = _finite_or(None, track_deg)
    delta_seconds = _finite_or(None, delta_seconds)

    # Re-check for None after conversion, as _finite_or could return None if no finite value and default is None
    if None in (start_lat, start_lon, start_alt, gs_kts, track_deg, delta_seconds):
        return None

    # Constants for unit conversion
    KTS_TO_KMH = 1.852
    FPM_TO_FTPS = 1.0 / 60.0

    # Calculate horizontal distance traveled (km)
    distance_km = (gs_kts * KTS_TO_KMH) * \
                  (delta_seconds / 3600.0)

    # Calculate new latitude and longitude using geodesic projection
    start_point = Point(latitude=start_lat, longitude=start_lon)
    bearing = track_deg % 360.0
    destination = geodesic(kilometers=distance_km).destination(
        start_point, bearing=bearing)

    # Calculate altitude change; treat non-finite vertical rate as 0
    vr_fpm_finite = _finite_or(0.0, vert_rate_fpm) # Provides a default of 0.0 if vert_rate_fpm is None or non-finite
    altitude_change_ft = (vr_fpm_finite * FPM_TO_FTPS) * \
                           delta_seconds
    new_altitude_ft = start_alt + altitude_change_ft

    return {
        'est_lat': destination.latitude,
        'est_lon': destination.longitude,
        'est_alt': new_altitude_ft,
    }


def estimate_positions_at_times(aircraft_data: Dict[str, Any], timestamps: List[float]) -> List[Dict[str, float]]:
    """
    Estimates future aircraft positions at a series of absolute timestamps.

    This function takes a single aircraft's current data and a list of future
    timestamps, then uses dead reckoning to predict its position at each timestamp.

    Args:
        aircraft_data: A dictionary containing a single aircraft's current state,
                       as returned by `read_aircraft_data`. Expected keys include
                       `lat`, `lon`, `alt`, `gs`, `track`, `vert_rate`, and `timestamp`.
        timestamps: A list of Unix timestamps (in seconds) for which to predict positions.

    Returns:
        A list of dictionaries, where each dictionary represents a predicted position
        at a given timestamp. Each prediction dictionary contains:
        - `est_lat`: Estimated latitude in degrees.
        - `est_lon`: Estimated longitude in degrees.
        - `est_alt`: Estimated altitude in feet.
        - `est_time`: The absolute epoch time of the prediction.

        Timestamps that are earlier than the `aircraft_data['timestamp']` are skipped.
    """
    predictions: List[Dict[str, float]] = []
    start_time = aircraft_data['timestamp']
    for ts in timestamps:
        delta_seconds = ts - start_time
        if delta_seconds < 0:
            continue

        # Predict position using the internal helper function
        pred = _predict(
            aircraft_data['lat'], aircraft_data['lon'], aircraft_data['alt'],
            aircraft_data['gs'], aircraft_data['track'],
            aircraft_data.get('vert_rate'),  # vert_rate can be optional
            delta_seconds
        )
        if pred is None:
            continue

        pred['est_time'] = ts
        predictions.append(pred)
    return predictions