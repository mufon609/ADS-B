import logging
import math
import numpy as np

import astropy.units as u
from astropy.coordinates import AltAz, EarthLocation
from astropy.time import Time

from config.loader import CONFIG
from astro.coords import angular_sep_deg

logger = logging.getLogger(__name__)

def is_target_viable(
    az: float,
    el: float,
    range_km: float,
    sun_az: float,
    sun_el: float,
    frame: AltAz,
) -> tuple[bool, list[str]]:
    """
    Checks if a target is viable for tracking based on common safety constraints
    (elevation, range, sun separation).

    Args:
        az: Target azimuth in degrees.
        el: Target elevation in degrees.
        range_km: Target range in kilometers.
        sun_az: Sun azimuth in degrees.
        sun_el: Sun elevation in degrees.
        frame: Astropy AltAz frame object for the time of evaluation.

    Returns:
        A tuple: (is_viable: bool, reasons: list[str]).
        is_viable is True if all constraints are met, False otherwise.
        reasons is a list of strings explaining any violated constraints.
    """
    reasons = []

    # Load configuration thresholds
    sel_cfg = CONFIG.get('selection', {})
    min_el = float(sel_cfg.get('min_elevation_deg', 5.0))
    max_range_km = float(sel_cfg.get('max_range_km', 100.0))
    min_sun_sep = float(sel_cfg.get('min_sun_separation_deg', 15.0))

    # Elevation Check
    if el < min_el:
        reasons.append(f"low_elevation ({el:.1f}° < {min_el}°)")

    # Range Check
    if range_km > max_range_km:
        reasons.append(f"out_of_range ({range_km:.1f}km > {max_range_km:.1f}km)")

    # Sun Separation Check
    try:
        sun_sep = angular_sep_deg((az, el), (sun_az, sun_el), frame)
        if sun_sep < min_sun_sep:
            reasons.append(f"too_close_to_sun ({sun_sep:.1f}° < {min_sun_sep}°)")
    except Exception as e:
        reasons.append(f"sun_separation_check_failed: {e}")
        logger.error(f"Error calculating sun separation for viability check: {e}")

    return len(reasons) == 0, reasons
