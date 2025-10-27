#coord_utils.py
"""
Module for coordinate transformations and astronomical calculations.
"""
import math
from astropy.coordinates import EarthLocation, SkyCoord, AltAz, get_sun
from astropy import units as u
from astropy.time import Time
from config_loader import CONFIG

def get_altaz_frame(observer_loc: EarthLocation) -> AltAz:
    """Creates an AltAz frame for the current time and observer location."""
    return AltAz(obstime=Time.now(), location=observer_loc)

def latlonalt_to_azel(lat: float, lon: float, alt_ft: float, timestamp: float, observer_loc: EarthLocation) -> tuple:
    """Converts geodetic coordinates to local Azimuth/Elevation."""
    time_obj = Time(timestamp, format='unix')
    # FIX: Use u.imperial.foot instead of the deprecated u.ft
    target_loc = EarthLocation.from_geodetic(lon*u.deg, lat*u.deg, height=alt_ft*u.imperial.foot)
    itrs_coord = SkyCoord(target_loc.get_itrs(obstime=time_obj))
    altaz_frame = AltAz(obstime=time_obj, location=observer_loc)
    altaz_coord = itrs_coord.transform_to(altaz_frame)
    return (altaz_coord.az.deg, altaz_coord.alt.deg)

def angular_sep_deg(p1: tuple, p2: tuple, frame: AltAz) -> float:
    """Calculates the angular separation between two Az/El points."""
    c1 = SkyCoord(az=p1[0]*u.deg, alt=p1[1]*u.deg, frame=frame)
    c2 = SkyCoord(az=p2[0]*u.deg, alt=p2[1]*u.deg, frame=frame)
    return c1.separation(c2).deg

def slew_time_needed(current: tuple, target: tuple, max_slew_rate_deg_s: float, frame: AltAz) -> float:
    """Calculates the time required to slew between two points."""
    if max_slew_rate_deg_s <= 0:
        return float('inf')
    angular_distance = angular_sep_deg(current, target, frame)
    return angular_distance / max_slew_rate_deg_s

def angular_speed_deg_s(start_azel: tuple, end_azel: tuple, time_delta_s: float, frame: AltAz) -> float:
    """Calculates the angular speed between two points in degrees per second."""
    if time_delta_s <= 0: return float('inf')
    angular_distance = angular_sep_deg(start_azel, end_azel, frame)
    return angular_distance / time_delta_s

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculates the Haversine distance between two points in kilometers."""
    R_km = 6371.0 # Radius of Earth in kilometers
    lat1_rad, lon1_rad = math.radians(lat1), math.radians(lon1)
    lat2_rad, lon2_rad = math.radians(lat2), math.radians(lon2)
    
    dlon, dlat = lon2_rad - lon1_rad, lat2_rad - lat1_rad
    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R_km * c

def get_sun_azel(timestamp: float, observer_loc: EarthLocation) -> tuple:
    """Gets the Sun's current Azimuth and Elevation."""
    time = Time(timestamp, format='unix')
    sun = get_sun(time)
    altaz_frame = AltAz(obstime=time, location=observer_loc)
    sun_altaz = sun.transform_to(altaz_frame)
    return (sun_altaz.az.deg, sun_altaz.alt.deg)

def calculate_plate_scale() -> float:
    """Calculates the plate scale in arcsec/pixel from config."""
    cam_specs = CONFIG['camera_specs']
    if 'plate_scale_arcsec_px' in CONFIG['pointing_calibration']:
        return CONFIG['pointing_calibration']['plate_scale_arcsec_px']
    
    focal_length_mm = cam_specs['focal_length_mm']
    pixel_size_um = cam_specs['pixel_size_um']
    return (206.265 * pixel_size_um) / focal_length_mm

def solve_intercept_time(current_az_el: tuple, target_azel_func, max_rate_deg_s: float, frame: AltAz, lo: float = 0.0, hi: float = 120.0) -> float:
    """
    Solves for the time 't' where t = slew_time(t).
    Uses a bisection method to find the intercept time.
    """
    def f(t):
        if t <= 0: return -1.0
        try:
            target_az, target_el = target_azel_func(t)
            return t - (angular_sep_deg(current_az_el, (target_az, target_el), frame) / max_rate_deg_s)
        except (TypeError, ValueError):
            return -1.0

    try:
        f_lo, f_hi = f(lo), f(hi)
        if f_lo * f_hi > 0:
            return None
    except TypeError:
        return None

    for _ in range(24):
        mid = 0.5 * (lo + hi)
        if mid == lo or mid == hi: break
        f_mid = f(mid)
        if f_mid > 0:
            hi = mid
        else:
            lo = mid
            
    return 0.5 * (lo + hi)