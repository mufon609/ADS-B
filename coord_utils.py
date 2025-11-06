#coord_utils.py
"""
Module for coordinate transformations and astronomical calculations.
"""
import math
from typing import Optional
from astropy.coordinates import EarthLocation, SkyCoord, AltAz, get_sun
from astropy import units as u
from astropy.time import Time
from geopy.distance import geodesic
from config_loader import CONFIG

def get_altaz_frame(observer_loc: EarthLocation) -> AltAz:
    """Creates an AltAz frame for the current time and observer location."""
    # Ensure observer location uses WGS84 ellipsoid for consistency with geopy
    # Astropy's default EarthLocation is already WGS84, so this is mainly for clarity.
    # We could explicitly set observer_loc = EarthLocation(..., ellipsoid='WGS84') if needed.
    return AltAz(obstime=Time.now(), location=observer_loc)

def latlonalt_to_azel(lat: float, lon: float, alt_ft: float, timestamp: float, observer_loc: EarthLocation) -> tuple:
    """
    Converts geodetic coordinates (WGS84) to local Azimuth/Elevation.
    Altitude is expected to be geometric height above the WGS84 ellipsoid.
    """
    time_obj = Time(timestamp, format='unix')
    # Use u.imperial.foot for altitude conversion
    # EarthLocation uses WGS84 ellipsoid by default.
    target_loc = EarthLocation.from_geodetic(lon*u.deg, lat*u.deg, height=alt_ft*u.imperial.foot)
    # Convert geodetic location to ITRS (Earth-centered) coordinates at the given time
    itrs_coord = SkyCoord(target_loc.get_itrs(obstime=time_obj))
    # Define the AltAz frame for the observer at the specified time
    altaz_frame = AltAz(obstime=time_obj, location=observer_loc)
    # Transform the target's ITRS coordinates to the observer's AltAz frame
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

def distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculates the geodesic distance between two points using the WGS84 ellipsoid.
    Returns distance in kilometers.
    """
    # geopy handles potential errors internally, returns distance object
    try:
        dist = geodesic((lat1, lon1), (lat2, lon2)).km
        return float(dist) # Ensure return type is float
    except Exception as e:
        # Log error or handle cases where distance calculation might fail
        print(f"Warning: Geodesic distance calculation failed between ({lat1},{lon1}) and ({lat2},{lon2}): {e}")
        return float('inf') # Return infinity or another indicator of failure

# Keep haversine for reference or specific use cases if needed, but rename
def _haversine_distance_km_spherical(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculates the Haversine distance assuming a spherical Earth."""
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
    # Removed the check for hard-coded value. Always calculate.
    focal_length_mm = cam_specs['focal_length_mm']
    pixel_size_um = cam_specs['pixel_size_um']

    # Apply binning factor if present
    bin_x = cam_specs.get('binning', {}).get('x', 1)
    bin_y = cam_specs.get('binning', {}).get('y', 1)
    # Assuming square pixels after binning, use average or primary bin factor
    # Or calculate separate X/Y scales if needed. Using bin_x for now.
    effective_pixel_size_um = pixel_size_um * bin_x

    if focal_length_mm <= 0:
         raise ValueError("Focal length must be positive.")
    if effective_pixel_size_um <= 0:
         raise ValueError("Pixel size and binning must result in positive effective size.")

    # Plate Scale (arcsec/pixel) = (Pixel Size (microns) / Focal Length (mm)) * 206.265
    return (206.265 * effective_pixel_size_um) / focal_length_mm

def solve_intercept_time(current_az_el: tuple, target_azel_func, max_rate_deg_s: float, frame: AltAz, lo: float = 0.0, hi: float = 120.0) -> Optional[float]:
    """
    Solves for the time 't' where t = slew_time(t).
    Uses a bisection method to find the intercept time. Returns None if no solution.
    """
    # Ensure max rate is positive
    if max_rate_deg_s <= 0:
        return None

    def f(t):
        """ Target function: t - slew_time(t). We want to find where f(t) = 0. """
        if t <= 0: return -1.0 # Slew time must be positive
        try:
            target_pos = target_azel_func(t)
            # Handle case where target function returns None (e.g., prediction failed)
            if target_pos is None or target_pos[0] is None or target_pos[1] is None:
                # Treat prediction failure as if the function value is very large (no solution in range)
                return float('inf')

            target_az, target_el = target_pos
            slew_time = angular_sep_deg(current_az_el, (target_az, target_el), frame) / max_rate_deg_s
            return t - slew_time
        except (TypeError, ValueError, AttributeError):
            # Handle potential errors from target_azel_func or angular_sep_deg
            # Treat errors as if no solution is likely
             print(f"Warning: Exception in f(t) for t={t:.2f}. Treating as no solution.")
             return float('inf') # Return a value unlikely to cross zero correctly

    try:
        f_lo, f_hi = f(lo), f(hi)
        # Check if f_lo or f_hi indicate immediate failure (inf)
        if not (math.isfinite(f_lo) and math.isfinite(f_hi)):
             return None
        # Bisection requires the function to cross zero within the interval [lo, hi]
        if f_lo * f_hi > 0:
            # If both have the same sign, no root guaranteed in this interval
            # Could mean no intercept, or intercept is outside [lo, hi]
            return None
    except Exception as e:
         print(f"Warning: Error evaluating initial bounds for intercept solve: {e}")
         return None # Error during initial evaluation

    # Bisection loop
    # Increase iterations slightly for better precision? 24 is likely fine.
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        # Check if interval is too small to continue
        if mid == lo or mid == hi: break
        try:
            f_mid = f(mid)
            if not math.isfinite(f_mid): # Handle prediction failures mid-solve
                 print(f"Warning: Non-finite result in intercept solve at t={mid:.2f}")
                 # Cannot reliably continue bisection if function fails
                 return None # Abort solve
        except Exception as e:
             print(f"Warning: Error during intercept solve iteration at t={mid:.2f}: {e}")
             return None # Abort solve

        # Bisection logic
        if f_mid == 0: # Found exact root (unlikely with floats)
             return mid
        elif f_lo * f_mid < 0: # Root is in [lo, mid]
            hi = mid
            f_hi = f_mid # Update f_hi for next iteration's sign check
        else: # Root is in [mid, hi]
            lo = mid
            f_lo = f_mid # Update f_lo

    # Return the midpoint of the final interval
    final_t = 0.5 * (lo + hi)

    # Final check: does the solution make sense? f(final_t) should be close to 0
    f_final = f(final_t)
    if not math.isfinite(f_final) or abs(f_final) > 0.5: # Allow tolerance (e.g., 0.5 seconds)
        print(f"Warning: Intercept solution {final_t:.2f}s did not converge well (f={f_final:.2f}). No intercept likely.")
        return None

    return final_t