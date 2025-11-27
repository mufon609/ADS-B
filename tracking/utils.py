import logging
import math
import time
from typing import Optional, Tuple

from adsb.dead_reckoning import estimate_positions_at_times
from astro.coords import latlonalt_to_azel

logger = logging.getLogger(__name__)


def predict_target_az_el(aircraft_data: dict, observer_loc, when: Optional[float] = None) -> Optional[Tuple[float, float]]:
    """
    Predict the Az/El coordinates of an aircraft at a specific time.

    Args:
        aircraft_data: Dictionary containing the aircraft's state vectors.
        observer_loc: Astropy EarthLocation for the observer.
        when: Unix timestamp at which to predict the position (defaults to now).

    Returns:
        Tuple of (azimuth, elevation) in degrees, or None if prediction fails.
    """
    t = when or time.time()
    try:
        pos_list = estimate_positions_at_times(aircraft_data, [t])
        if not pos_list:
            return None
        pos = pos_list[0]
        if not all(k in pos for k in ['est_lat', 'est_lon', 'est_alt']):
            logger.warning("Warning: Prediction missing required keys (lat/lon/alt).")
            return None
        az, el = latlonalt_to_azel(pos['est_lat'], pos['est_lon'], pos['est_alt'], t, observer_loc)
        if not (math.isfinite(az) and math.isfinite(el)):
            logger.warning(f"Warning: Non-finite az/el prediction ({az}, {el}).")
            return None
        return (float(az), float(el))
    except Exception as e:
        logger.error(f"Error predicting target az/el: {e}")
        return None
