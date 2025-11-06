#test_math_and_coords.py
"""
Unit tests for mathematical and coordinate utility functions.
"""

import unittest
import math
import time
from astropy.coordinates import EarthLocation
import astropy.units as u

from dead_reckoning import estimate_positions_at_times
from coord_utils import latlonalt_to_azel, distance_km
from config_loader import CONFIG


class TestMathAndCoords(unittest.TestCase):

    def test_distance_km(self):
        """Tests geodesic distance between NYC and LA."""
        lat1, lon1 = 40.64, -73.78
        lat2, lon2 = 33.94, -118.40
        known_dist_km = 3982 # <-- FIX: Updated to kilometers (was 2150 nm)
        calc_dist = distance_km(lat1, lon1, lat2, lon2)
        # Check if the calculated distance is close to the known distance in kilometers
        self.assertAlmostEqual(calc_dist, known_dist_km, delta=45)

    def test_dead_reckoning(self):
        """Tests a simple dead reckoning projection due East using the correct function."""
        start_time = time.time()
        start_data = {
            'lat': 34.0, 'lon': -118.0, 'gs': 500, 'track': 90,
            'alt': 35000, 'vert_rate': 0, 'timestamp': start_time
        }
        future_time = start_time + 30.0
        predictions = estimate_positions_at_times(start_data, [future_time])
        self.assertEqual(len(predictions), 1, "Should return exactly one prediction")
        est = predictions[0]
        
        self.assertAlmostEqual(est['est_time'], future_time, delta=1e-6)
        # Moving East from -118.0 longitude increases longitude (moves towards 0)
        # Approximate distance: 500 kts * (1.852 km/kt) * (30 / 3600 hr) ~= 7.7 km
        # At latitude 34 deg, 1 deg longitude is approx 111.32 * cos(34) ~= 92.2 km
        # Expected longitude change: 7.7 km / 92.2 km/deg ~= 0.08 degrees
        self.assertAlmostEqual(est['est_lat'], 34.0, delta=0.01)
        self.assertAlmostEqual(est['est_lon'], -118.0 + 0.08, delta=0.01)
        self.assertAlmostEqual(est['est_alt'], 35000, delta=1)

    def test_dead_reckoning_edge_cases(self):
        """Tests edge cases like stationary and North/South tracks using the correct function."""
        start_time = time.time()
        start_data = {'lat': 40.0, 'lon': -74.0, 'gs': 0, 'track': 123, 'alt': 10000, 'vert_rate': 0, 'timestamp': start_time}
        future_time = start_time + 30.0
        predictions = estimate_positions_at_times(start_data, [future_time])
        self.assertEqual(len(predictions), 1)
        est = predictions[0]
        self.assertAlmostEqual(est['est_lat'], 40.0, delta=1e-6)
        self.assertAlmostEqual(est['est_lon'], -74.0, delta=1e-6)
        
        # Test North track
        start_data['gs'] = 500
        start_data['track'] = 0 # North
        start_data['timestamp'] = start_time # Reset timestamp for consistency
        predictions = estimate_positions_at_times(start_data, [future_time])
        self.assertEqual(len(predictions), 1)
        est = predictions[0]
        self.assertTrue(est['est_lat'] > 40.0, "Latitude should increase when tracking North")
        self.assertAlmostEqual(est['est_lon'], -74.0, delta=1e-5) # Allow slightly larger delta for geodesic calc

        # Test South track
        start_data['track'] = 180 # South
        start_data['timestamp'] = start_time # Reset timestamp
        predictions = estimate_positions_at_times(start_data, [future_time])
        self.assertEqual(len(predictions), 1)
        est = predictions[0]
        self.assertTrue(est['est_lat'] < 40.0, "Latitude should decrease when tracking South")
        self.assertAlmostEqual(est['est_lon'], -74.0, delta=1e-5)

    def test_azel_conversion(self):
        """Tests a known lat/lon/alt to Az/El conversion with a fixed time."""
        # Observer directly below the target point at LAX
        observer_loc = EarthLocation(lat=33.94*u.deg, lon=-118.40*u.deg, height=38*u.m) # LAX approx location & elevation
        fixed_timestamp = 1664640000.0 # Arbitrary fixed time: 2022-10-01 16:00:00 UTC
        
        # Target directly overhead
        lat, lon, alt_ft = 33.94, -118.40, 40000 
        az, el = latlonalt_to_azel(lat, lon, alt_ft, fixed_timestamp, observer_loc)
        # Should be very close to zenith (El=90), Azimuth is arbitrary at zenith
        self.assertAlmostEqual(el, 90.0, delta=0.5, msg="Target directly overhead should be near El=90")

        # Target roughly East
        lat_east, lon_east = 33.94, -117.40 # Approx 1 degree East
        az_east, el_east = latlonalt_to_azel(lat_east, lon_east, alt_ft, fixed_timestamp, observer_loc)
        self.assertTrue(80 < az_east < 100, f"Target East should have Az near 90 (got {az_east:.1f})")
        self.assertTrue(0 < el_east < 80, f"Target East should have El < 90 (got {el_east:.1f})")

        # Target roughly North
        lat_north, lon_north = 34.94, -118.40 # Approx 1 degree North
        az_north, el_north = latlonalt_to_azel(lat_north, lon_north, alt_ft, fixed_timestamp, observer_loc)
        # Note: Azimuth can be ~0 or ~360 depending on exact position relative to pole
        self.assertTrue(az_north < 10 or az_north > 350, f"Target North should have Az near 0/360 (got {az_north:.1f})")
        self.assertTrue(0 < el_north < 80, f"Target North should have El < 90 (got {el_north:.1f})")


if __name__ == '__main__':
    unittest.main()