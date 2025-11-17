#dead_reckoning.py
"""
Module for predicting future aircraft positions using dead reckoning.
"""
import math
from typing import Dict, List

from geopy.distance import geodesic
from geopy.point import Point

from config_loader import CONFIG

def _finite_or(default: float, *vals) -> float:
    """Return the first finite value from vals, else default."""
    for v in vals:
        try:
                    if math.isfinite(float(v)):
                        return float(v)
        except (TypeError, ValueError):
            pass
    return float(default) if default is not None else None

def _predict(start_lat, start_lon, start_alt, gs_kts, track_deg, vert_rate_fpm, delta_seconds):
    """Internal helper to predict a single future position."""
    if None in (start_lat, start_lon, start_alt, gs_kts, track_deg, delta_seconds):
        return None

    start_lat = _finite_or(None, start_lat)
    start_lon = _finite_or(None, start_lon)
    start_alt = _finite_or(None, start_alt)
    gs_kts = _finite_or(None, gs_kts)
    track_deg = _finite_or(None, track_deg)
    delta_seconds = _finite_or(None, delta_seconds)

    # Constants
    KTS_TO_KMH = 1.852
    FPM_TO_FTPS = 1.0 / 60.0

    # Distance traveled (km)
    distance_km = (gs_kts * KTS_TO_KMH) * (delta_seconds / 3600.0)

    # Great-circle destination
    start_point = Point(latitude=start_lat, longitude=start_lon)
    bearing = track_deg % 360.0
    destination = geodesic(kilometers=distance_km).destination(start_point, bearing=bearing)

    # Altitude (ft); treat non-finite vert_rate as 0
    vr_fpm = _finite_or(0.0, vert_rate_fpm)
    altitude_change_ft = (vr_fpm * FPM_TO_FTPS) * delta_seconds
    new_altitude_ft = start_alt + altitude_change_ft

    return {
        'est_lat': destination.latitude,
        'est_lon': destination.longitude,
        'est_alt': new_altitude_ft,
    }

def estimate_positions_at_times(aircraft_data: dict, timestamps: List[float]) -> List[dict]:
    """
    Estimates aircraft positions at a specific list of (absolute, epoch) timestamps.
    """
    predictions: List[dict] = []
    start_time = aircraft_data['timestamp']
    for ts in timestamps:
        delta_seconds = ts - start_time
        if delta_seconds < 0:
            continue
        pred = _predict(
            aircraft_data['lat'], aircraft_data['lon'], aircraft_data['alt'],
            aircraft_data['gs'], aircraft_data['track'],
            aircraft_data.get('vert_rate'),
            delta_seconds
        )
        if pred is None:
            continue
        pred['est_time'] = ts
        predictions.append(pred)
    return predictions
        