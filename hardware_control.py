# hardware_control.py
"""
Module for controlling hardware via INDI.
"""

import PyIndi
import time
import os
import numpy as np
import cv2
import threading
from typing import List, Optional, Callable
from storage import append_to_json
from config_loader import CONFIG, LOG_DIR
# --- FIX: Import in-memory helpers ---
from image_analyzer import measure_sharpness, save_png_preview, _load_fits_data, _measure_sharpness_from_data
# --- End FIX ---
from coord_utils import get_sun_azel, slew_time_needed, get_altaz_frame, angular_sep_deg
from astropy.coordinates import SkyCoord, AltAz, EarthLocation
import astropy.units as u
from astropy.time import Time
from astropy.io import fits
from stack_orchestrator import schedule_stack_and_publish

class IndiController(PyIndi.BaseClient):
    """INDI client for mount, camera, and focuser."""
    def __init__(self):
        super().__init__()
        self.shutdown_event: Optional[threading.Event] = threading.Event()
        if CONFIG['development']['dry_run']:
            print("[DRY RUN] IndiController initialized in dry-run mode.")
            self.simulated_az_el = (180.0, 45.0) # Start somewhere reasonable
            obs_cfg = CONFIG['observer']
            self.observer_loc = EarthLocation(
                lat=obs_cfg['latitude_deg']*u.deg,
                lon=obs_cfg['longitude_deg']*u.deg,
                height=obs_cfg['altitude_m']*u.m
            )
            return

        print("Connecting to INDI server...")
        self.setServer(CONFIG['hardware']['indi_host'], CONFIG['hardware']['indi_port'])
        if not self.connectServer():
            raise ConnectionError("Failed to connect to INDI server")
        # Allow time for properties to arrive
        time.sleep(3) # Increased from 2

        # Fetch devices, check existence immediately
        self.mount = self.getDevice(CONFIG['hardware']['mount_device_name'])
        self.camera = self.getDevice(CONFIG['hardware']['camera_device_name'])
        self.focuser = self.getDevice(CONFIG['hardware']['focuser_device_name'])
        if not self.mount: raise ValueError(f"Mount device '{CONFIG['hardware']['mount_device_name']}' not found")
        if not self.camera: raise ValueError(f"Camera device '{CONFIG['hardware']['camera_device_name']}' not found")
        if not self.focuser: raise ValueError(f"Focuser device '{CONFIG['hardware']['focuser_device_name']}' not found")

        # Configure BLOB mode robustly
        self.setBLOBMode(PyIndi.B_CLIENT, self.camera.getDeviceName(), None) # Subscribe initially
        time.sleep(0.5) # Give time for BLOB properties to appear
        self._blob_name = self._detect_ccd_blob_name()
        # If a specific BLOB was found, subscribe only to that one
        # Use a distinct fallback name check
        if self._blob_name != "FALLBACK_CCD1":
            self.setBLOBMode(PyIndi.B_CLIENT, self.camera.getDeviceName(), self._blob_name)
            print(f"  - Set Camera: BLOB mode to '{self._blob_name}'.")
        else:
             print(f"  - Warning: Using fallback BLOB subscription ('CCD1'). Ensure only one image source.")
             # Explicitly set to CCD1 if fallback needed, though None might have worked too
             self.setBLOBMode(PyIndi.B_CLIENT, self.camera.getDeviceName(), "CCD1")


        print("--- Hardware Connected & Configured ---")
        for device, name in [(self.mount, "Mount"), (self.camera, "Camera"), (self.focuser, "Focuser")]:
            # Added more robust driver info fetching with fallback message
            driver_info_text = "Driver info not available."
            try:
                di = device.getText("DRIVER_INFO")
                # Check if property exists, has elements, and the element has text attribute
                if di and len(di) > 0 and hasattr(di[0], 'text') and di[0].text:
                     driver_info_text = di[0].text
            except Exception as e:
                print(f"  - Error fetching driver info for {name}: {e}") # Log error if fetching fails
            print(f"  - {name} ({device.getDeviceName()}): {driver_info_text}")

        cam_specs = CONFIG['camera_specs']

        # Configure Camera Upload Mode
        upload_mode = self._wait_switch(self.camera, "UPLOAD_MODE")
        if upload_mode:
            # Check if Client mode exists before setting it
            client_widget = upload_mode.findWidgetByName("UPLOAD_CLIENT")
            if client_widget:
                client_widget.s = PyIndi.ISS_ON
                self.sendNewSwitch(upload_mode)
                print("  - Set Camera: Upload mode to 'Client'.")
            else:
                print("  - Warning: Camera does not support 'UPLOAD_CLIENT' mode.")

        # Configure Camera Cooling
        if cam_specs.get('cooling', {}).get('enabled', False):
            cooler_switch = self._wait_switch(self.camera, "CCD_COOLER")
            if cooler_switch:
                # Check if COOLER_ON exists before trying to set it
                on_widget = cooler_switch.findWidgetByName("COOLER_ON")
                if on_widget:
                    on_widget.s = PyIndi.ISS_ON
                    self.sendNewSwitch(cooler_switch)
                else:
                    print("  - Warning: CCD_COOLER does not have COOLER_ON switch.")

            temp_prop = self._wait_number(self.camera, "CCD_TEMPERATURE")
            if temp_prop:
                widget = temp_prop.findWidgetByName("CCD_TEMPERATURE_VALUE")
                if widget:
                    setpoint = cam_specs['cooling']['setpoint_c']
                    widget.value = setpoint
                    self.sendNewNumber(temp_prop)
                    print(f"  - Set Camera: Cooling enabled, setpoint {setpoint}°C.")
                else:
                    print("  - Warning: CCD_TEMPERATURE does not have VALUE widget.")

        # Configure Camera Binning
        if 'binning' in cam_specs:
            bin_prop = self._wait_number(self.camera, "CCD_BINNING")
            if bin_prop:
                hor_widget = bin_prop.findWidgetByName("HOR_BIN")
                ver_widget = bin_prop.findWidgetByName("VER_BIN")
                if hor_widget and ver_widget:
                    hor_widget.value = cam_specs['binning']['x']
                    ver_widget.value = cam_specs['binning']['y']
                    self.sendNewNumber(bin_prop)
                    print(f"  - Set Camera: Binning to {cam_specs['binning']['x']}x{cam_specs['binning']['y']}.")
                else:
                    print("  - Warning: CCD_BINNING missing HOR_BIN or VER_BIN.")

        # Configure Camera Gain
        if 'gain' in cam_specs:
            gain_prop = self._wait_number(self.camera, "CCD_GAIN")
            if gain_prop:
                widget = gain_prop.findWidgetByName("GAIN")
                if widget:
                    widget.value = cam_specs['gain']
                    self.sendNewNumber(gain_prop)
                    print(f"  - Set Camera: Gain to {cam_specs['gain']}.")
                else:
                    # Some drivers use CCD_GAIN_VALUE instead of GAIN
                    widget = gain_prop.findWidgetByName("CCD_GAIN_VALUE")
                    if widget:
                        widget.value = cam_specs['gain']
                        self.sendNewNumber(gain_prop)
                        print(f"  - Set Camera: Gain to {cam_specs['gain']} (using CCD_GAIN_VALUE).")
                    else:
                         print("  - Warning: CCD_GAIN property missing GAIN widget.")


        # Configure Camera Offset
        if 'offset' in cam_specs:
            offset_prop = self._wait_number(self.camera, "CCD_OFFSET")
            if offset_prop:
                widget = offset_prop.findWidgetByName("OFFSET")
                if widget:
                    widget.value = cam_specs['offset']
                    self.sendNewNumber(offset_prop)
                    print(f"  - Set Camera: Offset to {cam_specs['offset']}.")
                else:
                     # Some drivers might use different names, e.g., CCD_OFFSET_VALUE
                    widget = offset_prop.findWidgetByName("CCD_OFFSET_VALUE")
                    if widget:
                        widget.value = cam_specs['offset']
                        self.sendNewNumber(offset_prop)
                        print(f"  - Set Camera: Offset to {cam_specs['offset']} (using CCD_OFFSET_VALUE).")
                    else:
                        print("  - Warning: CCD_OFFSET property missing OFFSET widget.")

        print("---------------------------------------")

        # Store observer location from config for internal use
        obs_cfg = CONFIG['observer']
        self.observer_loc = EarthLocation(
            lat=obs_cfg['latitude_deg']*u.deg,
            lon=obs_cfg['longitude_deg']*u.deg,
            height=obs_cfg['altitude_m']*u.m
        )

    # Helper to wait for INDI properties with timeout
    def _wait_prop(self, get_func, dev, name, timeout=5.0): # Increased default timeout
        t0 = time.time()
        prop = None
        while time.time() - t0 < timeout:
            # Added check for device validity before calling get_func
            if not dev: return None
            try:
                prop = get_func(dev, name)
                if prop:
                    return prop
            except Exception as e:
                # Catch potential errors if device disconnects during wait
                print(f"Warning: Error accessing property '{name}' on device: {e}")
                return None
            time.sleep(0.1)
        # Provide more context in warning message
        dev_name = "UNKNOWN_DEVICE"
        try:
             # Safer way to get device name
             if dev and hasattr(dev, 'getDeviceName'):
                dev_name = dev.getDeviceName() or "EMPTY_NAME"
        except Exception:
             pass # Ignore errors getting name if device is invalid
        print(f"Warning: Property '{name}' not available on device '{dev_name}' after {timeout}s.")
        return None


    def _wait_number(self, dev, name, timeout=5.0):
        return self._wait_prop(PyIndi.BaseDevice.getNumber, dev, name, timeout)

    def _wait_switch(self, dev, name, timeout=5.0):
        return self._wait_prop(PyIndi.BaseDevice.getSwitch, dev, name, timeout)

    # Helper to clear any pending BLOBs (images) from the camera buffer
    def _drain_blobs(self, prop_name: str, max_loops: int = 5) -> None: # Increased loops slightly
        """Clears any pending BLOBs for the given property name."""
        if CONFIG['development']['dry_run']: return # Skip in dry run
        try:
            # Check camera device validity at the start
            if not self.camera:
                 print("Warning: Cannot drain BLOBs, camera device is not valid.")
                 return
            for _ in range(max_loops):
                blob_prop = self.camera.getBLOB(prop_name)
                # Check property state and blob length before getting data
                if (blob_prop and
                    len(blob_prop) > 0 and
                    blob_prop.s != PyIndi.IPS_IDLE and # Don't drain if idle
                    hasattr(blob_prop[0], 'bloblen') and
                    blob_prop[0].bloblen > 0):

                    _ = blob_prop[0].getblobdata() # Read data to clear buffer
                    # Short sleep to allow INDI server to process
                    time.sleep(0.1)
                else:
                    # No more data or property is idle, exit loop
                    break
        except Exception as e:
            # Catch potential errors during BLOB access (e.g., device disconnects)
            print(f"Warning: Error draining BLOB '{prop_name}': {e}")


    # Helper to detect the correct BLOB property name for the camera
    def _detect_ccd_blob_name(self) -> str:
        """Tries common BLOB property names and returns the first one found."""
        if CONFIG['development']['dry_run']: return "DRY_RUN_CCD1" # Return dummy for dry run

        # Check camera device validity first
        if not self.camera:
             print("  - Error: Camera device is not valid for BLOB detection.")
             return "FALLBACK_CCD1" # Use fallback if camera is invalid

        # Common INDI BLOB property names
        common_names = ["CCD1", "CCD_IMAGE", "PRIMARY_CCD", "MAIN_IMAGE",
                        "VIDEO_STREAM", "PREVIEW_IMAGE", "CCD_EXPOSURE_FRAME"]
        for name in common_names:
            try:
                # Check if the BLOB property exists for the camera device
                if self.camera.getBLOB(name) is not None:
                    print(f"  - Detected camera BLOB property: {name}")
                    return name
            except Exception as e:
                 # Ignore errors if a property name doesn't exist or device disconnects
                 # print(f"    - Debug: Error checking BLOB name '{name}': {e}") # Optional debug log
                 pass
        print("  - Warning: Could not detect a common BLOB property, falling back to 'CCD1'.")
        # Use a distinct fallback name
        return "FALLBACK_CCD1"

    def get_hardware_status(self) -> dict:
        """Fetches current status from all connected hardware."""
        if CONFIG['development']['dry_run']:
            # Return realistic simulated status
            return {
                "mount_az_el": self.simulated_az_el,
                "camera_temp": CONFIG['camera_specs'].get('cooling', {}).get('setpoint_c', -10.0) + 0.1, # Simulate near setpoint
                "camera_cooler_power": 45.5,
                "focuser_pos": 12345,
                "focuser_temp": 22.8,
                "mount_pier_side": "East", # Or West, simulate reasonably
                "camera_binning": CONFIG['camera_specs'].get('binning', {"x": 1, "y": 1}),
                "camera_gain": CONFIG['camera_specs'].get('gain', 100),
                "camera_offset": CONFIG['camera_specs'].get('offset', 50),
            }

        status = {}

        # --- Mount Status ---
        try:
            # Use internal helper which handles coordinate systems
            status["mount_az_el"] = self._get_current_az_el_internal()
        except Exception as e:
            status["mount_az_el"] = (0.0, 0.0) # Sensible default on error
            print(f"Warning: could not get mount az/el: {e}")

        # Get Pier Side if available
        # Added check for self.mount validity
        pier_side_prop = self.mount.getSwitch("TELESCOPE_PIER_SIDE") if self.mount else None
        if pier_side_prop:
            # Iterate through switches to find the active one (ON state)
            status["mount_pier_side"] = next((el.label for el in pier_side_prop if el.s == PyIndi.ISS_ON), "Unknown")

        # --- Camera Status ---
        if self.camera:
            # Temperature
            temp_prop = self.camera.getNumber("CCD_TEMPERATURE")
            if temp_prop:
                widget = temp_prop.findWidgetByName("CCD_TEMPERATURE_VALUE")
                if widget: status["camera_temp"] = round(widget.value, 2) # Round for display

            # Cooler Power
            cooler_prop = self.camera.getNumber("CCD_COOLER_POWER")
            if cooler_prop:
                # Common widget name variations
                widget = cooler_prop.findWidgetByName("CCD_COOLER_VALUE") or cooler_prop.findWidgetByName("POWER")
                if widget: status["camera_cooler_power"] = round(widget.value, 1)

            # Binning
            bin_prop = self.camera.getNumber("CCD_BINNING")
            if bin_prop:
                w1 = bin_prop.findWidgetByName("HOR_BIN")
                w2 = bin_prop.findWidgetByName("VER_BIN")
                # Ensure values are valid before converting to int
                if w1 and w2 and w1.value is not None and w2.value is not None:
                     status["camera_binning"] = {"x": int(w1.value), "y": int(w2.value)}

            # Gain
            gain_prop = self.camera.getNumber("CCD_GAIN")
            if gain_prop:
                widget = gain_prop.findWidgetByName("GAIN") or gain_prop.findWidgetByName("CCD_GAIN_VALUE")
                if widget and widget.value is not None: status["camera_gain"] = int(widget.value)

            # Offset
            offset_prop = self.camera.getNumber("CCD_OFFSET")
            if offset_prop:
                 widget = offset_prop.findWidgetByName("OFFSET") or offset_prop.findWidgetByName("CCD_OFFSET_VALUE")
                 if widget and widget.value is not None: status["camera_offset"] = int(widget.value)


        # --- Focuser Status ---
        if self.focuser:
            # Position
            pos_prop = self.focuser.getNumber("ABS_FOCUS_POSITION")
            if pos_prop:
                widget = pos_prop.findWidgetByName("FOCUS_ABSOLUTE_POSITION")
                if widget and widget.value is not None: status["focuser_pos"] = int(widget.value)

            # Temperature
            focuser_temp_prop = self.focuser.getNumber("FOCUSER_TEMPERATURE")
            if focuser_temp_prop:
                widget = focuser_temp_prop.findWidgetByName("FOCUSER_TEMPERATURE_VALUE")
                if widget and widget.value is not None: status["focuser_temp"] = round(widget.value, 1)

        return status

    def _get_current_az_el_internal(self) -> tuple[float, float]:
        """
        Internal helper to read mount Az/El.
        Prefers HORIZONTAL_COORD, falls back to EQUATORIAL_EOD_COORD + conversion.
        """
        if not self.mount:
             raise ValueError("Mount device not available")

        # Try Horizontal Coordinates first
        horiz = self.mount.getNumber("HORIZONTAL_COORD")
        if horiz:
            az_w = horiz.findWidgetByName("AZ")
            alt_w = horiz.findWidgetByName("ALT")
             # Ensure widgets exist and state is OK or Idle
            if az_w and alt_w and horiz.s in [PyIndi.IPS_OK, PyIndi.IPS_IDLE]:
                return (az_w.value, alt_w.value)

        # Fallback to Equatorial Coordinates if Horizontal unavailable or not OK
        eq = self.mount.getNumber("EQUATORIAL_EOD_COORD")
        if eq:
            ra_w = eq.findWidgetByName("RA") # RA in hours
            dec_w = eq.findWidgetByName("DEC") # Dec in degrees
             # Ensure widgets exist and state is OK or Idle
            if ra_w and dec_w and eq.s in [PyIndi.IPS_OK, PyIndi.IPS_IDLE]:
                ra_hours, dec_deg = ra_w.value, dec_w.value
                # Convert RA/Dec (ICRS/J2000 assumed by default) to AltAz for NOW
                # Use current time for the transformation frame
                now_time = Time.now()
                sky_coord = SkyCoord(ra=ra_hours*u.hourangle, dec=dec_deg*u.deg, frame='icrs')
                altaz_frame = AltAz(obstime=now_time, location=self.observer_loc)
                altaz_coord = sky_coord.transform_to(altaz_frame)
                return (altaz_coord.az.deg, altaz_coord.alt.deg)

        # If neither coordinate system is available or OK, raise error
        raise ValueError("Mount coordinates (Horizontal or Equatorial) not found or not in OK state")


    def get_current_az_el(self) -> tuple[float, float]:
        """Public method to get mount az/el, honoring dry-run simulation."""
        if CONFIG['development']['dry_run']:
            return self.simulated_az_el
        try:
            return self._get_current_az_el_internal()
        except ValueError as e:
             print(f"Error getting mount coordinates: {e}")
             # Return a default/last known value or raise? Returning default for now.
             return (0.0, 0.0)


    def slew_to_az_el(
        self,
        az: float,
        el: float,
        progress_cb: Optional[Callable[[float, float, str], None]] = None
    ) -> bool:
        """
        Slews mount to target Az/El and returns True on success.
        Includes predictive sun safety check.
        """
        # --- Dry Run Simulation ---
        if CONFIG['development']['dry_run']:
            try:
                current_az, current_el = self.simulated_az_el
                target_az, target_el = az % 360.0, el # Normalize target az
                frame = get_altaz_frame(self.observer_loc) # Use helper for current frame
                max_slew = float(CONFIG['hardware'].get('max_slew_deg_s', 6.0))
                # Use slew_time_needed for realistic dry run duration
                realistic_slew_time = slew_time_needed((current_az, current_el), (target_az, target_el), max_slew, frame)

                print(f"[DRY RUN] Slewing from ({current_az:.2f}, {current_el:.2f}) to az/el: ({target_az:.2f}, {target_el:.2f}). Est time: {realistic_slew_time:.1f}s")

                # Simulate movement in steps for progress callback
                steps = max(1, int(realistic_slew_time / 0.5)) # ~0.5s steps
                start_az, start_el = current_az, current_el
                # Calculate shortest path in azimuth, handling wrap-around
                az_delta = ((target_az - start_az + 180.0) % 360.0) - 180.0
                el_delta = target_el - start_el

                for i in range(1, steps + 1):
                    if self.shutdown_event and self.shutdown_event.is_set():
                        if progress_cb:
                            try: progress_cb(*self.simulated_az_el, "abort")
                            except Exception: pass # Ignore callback errors
                        print("[DRY RUN] Slew aborted by shutdown.")
                        return False

                    fraction = i / steps
                    cur_az = (start_az + fraction * az_delta) % 360.0
                    cur_el = start_el + fraction * el_delta
                    self.simulated_az_el = (cur_az, cur_el)
                    if progress_cb:
                         try: progress_cb(cur_az, cur_el, "slewing")
                         except Exception: pass
                    # Simulate time passing
                    time.sleep(realistic_slew_time / steps)

                # Final position after simulation
                self.simulated_az_el = (target_az, target_el)
                if progress_cb:
                    try: progress_cb(self.simulated_az_el[0], self.simulated_az_el[1], "arrived")
                    except Exception: pass
                print(f"[DRY RUN] Slew complete. Mount is now at: ({target_az:.2f}, {target_el:.2f})")
                return True
            except Exception as e:
                # Catch errors during dry run simulation
                 print(f"[DRY RUN] Error during slew simulation: {e}")
                 return False


        # --- Real Hardware Slew ---
        if not self.mount:
            print("Error: Mount device not available for slew.")
            return False

        # Normalize target coordinates
        target_az = az % 360.0
        # Add basic safety clamp for elevation
        target_el = max(-5.0, min(95.0, el))

        # --- Predictive Sun Safety Check --- FIX Applied Here ---
        try:
            current_az, current_el = self.get_current_az_el()
            max_slew = float(CONFIG['hardware'].get('max_slew_deg_s', 6.0))
            now_frame = get_altaz_frame(self.observer_loc) # Frame for now

            # Estimate slew time
            est_slew_time = slew_time_needed((current_az, current_el), (target_az, target_el), max_slew, now_frame)
            # Add a small buffer just in case
            est_slew_time += 2.0

            # Calculate arrival time and sun position AT arrival
            arrival_time_unix = time.time() + est_slew_time
            arrival_time_obj = Time(arrival_time_unix, format='unix')
            sun_az_el_future = get_sun_azel(arrival_time_unix, self.observer_loc)

            # Create frame for arrival time
            future_frame = AltAz(obstime=arrival_time_obj, location=self.observer_loc)

            # Calculate separation at arrival time using the future frame
            separation_deg_future = angular_sep_deg((target_az, target_el), sun_az_el_future, future_frame)

            min_sep = float(CONFIG['selection'].get('min_sun_separation_deg', 15.0))
            if separation_deg_future < min_sep:
                print(f"SAFETY ABORT (Predictive): Target ({target_az:.1f}, {target_el:.1f}) will be {separation_deg_future:.2f}° from Sun "
                      f"(at ~{est_slew_time:.1f}s). Min separation is {min_sep}°. Sun will be at ({sun_az_el_future[0]:.1f}, {sun_az_el_future[1]:.1f}).")
                if progress_cb:
                     try: progress_cb(current_az, current_el, "abort") # Report current position on abort
                     except Exception: pass
                return False
            # print(f"  - Predictive sun check OK: Target sep {separation_deg_future:.2f}° > {min_sep}° at arrival.") # Optional debug log

        except Exception as e:
            print(f"Warning: Predictive sun safety check failed: {e}. Proceeding with caution.")
            # Optionally, you could abort here if the check fails for any reason
            # return False
        # --- End Predictive Sun Check ---

        # --- Hardware Altitude Limit Check ---
        alt_limit_prop = self.mount.getNumber("ALTITUDE_LIMITS")
        if alt_limit_prop:
            min_el_w = alt_limit_prop.findWidgetByName("MIN")
            if min_el_w and target_el < min_el_w.value:
                print(f"SAFETY ABORT: Target elevation {target_el:.2f}° is below min hardware limit {min_el_w.value:.2f}°.")
                if progress_cb:
                     try: progress_cb(*self.get_current_az_el(), "abort")
                     except Exception: pass
                return False
            max_el_w = alt_limit_prop.findWidgetByName("MAX")
            if max_el_w and target_el > max_el_w.value:
                print(f"SAFETY ABORT: Target elevation {target_el:.2f}° exceeds max hardware limit {max_el_w.value:.2f}°.")
                if progress_cb:
                    try: progress_cb(*self.get_current_az_el(), "abort")
                    except Exception: pass
                return False

        # --- Send Slew Command ---
        coord_prop_name = ""
        command_sent = False
        # Prefer Horizontal Coordinates if available
        horiz = self.mount.getNumber("HORIZONTAL_COORD")
        if horiz:
             # Check if mount allows setting horizontal coords directly
             # Some mounts are read-only for HORIZONTAL_COORD
             if horiz.p == PyIndi.IP_RW or horiz.p == PyIndi.IP_WO:
                az_widget = horiz.findWidgetByName("AZ")
                alt_widget = horiz.findWidgetByName("ALT")
                if az_widget and alt_widget:
                    coord_prop_name = "HORIZONTAL_COORD"
                    alt_widget.value = target_el
                    az_widget.value = target_az
                    self.sendNewNumber(horiz)
                    command_sent = True
                    print(f"  - Slewing using HORIZONTAL_COORD to ({target_az:.2f}, {target_el:.2f}).")
                else:
                     print("  - Warning: HORIZONTAL_COORD widgets not found.")
             else:
                  print("  - Info: HORIZONTAL_COORD is read-only, using EQUATORIAL_EOD_COORD.")


        # Fallback to Equatorial Coordinates
        if not command_sent:
            eq = self.mount.getNumber("EQUATORIAL_EOD_COORD")
            if eq and (eq.p == PyIndi.IP_RW or eq.p == PyIndi.IP_WO):
                ra_widget = eq.findWidgetByName("RA") # Expects hours
                dec_widget = eq.findWidgetByName("DEC") # Expects degrees
                if ra_widget and dec_widget:
                    coord_prop_name = "EQUATORIAL_EOD_COORD"
                    # Convert target AltAz TO Equatorial for the command time
                    command_time = Time.now()
                    altaz_frame = AltAz(obstime=command_time, location=self.observer_loc)
                    target_altaz = SkyCoord(az=target_az*u.deg, alt=target_el*u.deg, frame=altaz_frame)
                    # Transform to ICRS (effectively J2000 RA/Dec)
                    target_icrs = target_altaz.transform_to('icrs')

                    ra_widget.value = target_icrs.ra.hour # Send RA in hours
                    dec_widget.value = target_icrs.dec.deg # Send Dec in degrees
                    self.sendNewNumber(eq)
                    command_sent = True
                    print(f"  - Slewing using EQUATORIAL_EOD_COORD to RA/Dec ({target_icrs.ra.hour:.4f}h, {target_icrs.dec.deg:.4f}°).")
                else:
                     print("  - Warning: EQUATORIAL_EOD_COORD widgets not found.")

            else:
                 print(f"Error: Cannot send slew command. Neither HORIZONTAL_COORD nor EQUATORIAL_EOD_COORD is available/writable on mount '{self.mount.getDeviceName()}'.")
                 if progress_cb:
                     try: progress_cb(*self.get_current_az_el(), "abort")
                     except Exception: pass
                 return False


        # --- Wait for Slew Completion ---
        # Estimate timeout based on distance, add buffer
        try:
            # Recalculate start coords in case they changed slightly
            cur_az, cur_el = self.get_current_az_el()
            # Use angular_sep_deg for accurate distance on sphere
            dist = angular_sep_deg((cur_az, cur_el), (target_az, target_el), get_altaz_frame(self.observer_loc))
            # Use max slew rate from config
            max_slew = float(CONFIG['hardware'].get('max_slew_deg_s', 6.0))
            est_time = (dist / max_slew) if max_slew > 0 else 60.0 # Default 60s if rate is 0
            # Generous timeout: max(minimum_time, estimated_time + buffer)
            slew_timeout_at = time.time() + max(20.0, est_time + 15.0)
        except Exception as e:
             print(f"Warning: Could not estimate slew time accurately: {e}. Using fixed timeout.")
             slew_timeout_at = time.time() + 90.0 # Fixed timeout if estimation fails


        # Poll mount status until slew finishes, times out, or is aborted
        last_reported_az, last_reported_el = -1.0, -1.0 # Track last reported coords
        while time.time() < slew_timeout_at:
            coord_prop = self.mount.getNumber(coord_prop_name)

            # Check property status
            if coord_prop and coord_prop.s == PyIndi.IPS_BUSY:
                try:
                    # Get current position during slew for progress callback
                    current_az, current_el = self._get_current_az_el_internal()
                    # Only call callback if position changed significantly to avoid spam
                    if abs(current_az - last_reported_az) > 0.1 or abs(current_el - last_reported_el) > 0.1:
                        if progress_cb:
                            try: progress_cb(current_az, current_el, "slewing")
                            except Exception: pass
                        last_reported_az, last_reported_el = current_az, current_el
                except Exception:
                     # Ignore errors getting position during slew, just keep polling
                     pass

                # Check for shutdown signal during busy wait
                if self.shutdown_event and self.shutdown_event.is_set():
                    print("Slew aborted by shutdown signal.")
                    # Optionally send abort command to mount if available
                    # abort_prop = self.mount.getSwitch("TELESCOPE_ABORT_MOTION")
                    # if abort_prop: ... self.sendNewSwitch(...)
                    if progress_cb:
                        try: progress_cb(*self.get_current_az_el(), "abort")
                        except Exception: pass
                    return False

                # Wait before next poll
                time.sleep(0.5)
                continue # Continue loop while busy

            # If not busy, break the loop (slew finished, failed, or property lost)
            break

        # --- Check Final Slew Status ---
        final_coord_prop = self.mount.getNumber(coord_prop_name)

        if not final_coord_prop or final_coord_prop.s == PyIndi.IPS_BUSY:
            # Slew timed out or property was lost
            print("Warning: Slew command timed out or mount property lost.")
            if progress_cb:
                 try: progress_cb(*self.get_current_az_el(), "abort") # Report last known position
                 except Exception: pass
            return False

        if final_coord_prop.s == PyIndi.IPS_ALERT:
            # Slew failed explicitly
            print("Warning: Slew command failed (Mount reported Alert state). Check INDI logs.")
            if progress_cb:
                try: progress_cb(*self.get_current_az_el(), "abort")
                except Exception: pass
            return False

        # If status is OK or IDLE, assume success
        if final_coord_prop.s in [PyIndi.IPS_OK, PyIndi.IPS_IDLE]:
            try:
                # Get final position after slew completion
                final_az, final_el = self.get_current_az_el()
                print(f"Slew complete. Arrived at az/el: ({final_az:.2f}, {final_el:.2f})")
                if progress_cb:
                    try: progress_cb(final_az, final_el, "arrived")
                    except Exception: pass
                return True
            except Exception as e:
                 # If reading final position fails, log but still consider slew successful based on state
                 print(f"Warning: Slew likely completed (state={final_coord_prop.s}), but failed to read final position: {e}")
                 # Report target as final position if read fails
                 if progress_cb:
                      try: progress_cb(target_az, target_el, "arrived") # Report target coords
                      except Exception: pass
                 return True

        # Should not be reached if logic is correct, but handle unexpected states
        print(f"Warning: Slew ended in unexpected state: {final_coord_prop.s}")
        if progress_cb:
             try: progress_cb(*self.get_current_az_el(), "abort")
             except Exception: pass
        return False


    def autofocus(self, alt_ft: Optional[float] = None) -> bool:
        """Runs an autofocus routine based on sharpness."""
        if self.shutdown_event and self.shutdown_event.is_set():
            print("Autofocus aborted: shutdown requested.")
            return False

        if CONFIG['development']['dry_run']:
            print(f"[DRY RUN] Simulating autofocus (Target Alt: {alt_ft if alt_ft else 'N/A'} ft).")
            # Simulate success
            return True

        if not self.focuser or not self.camera:
             print("Error: Focuser or Camera device not available for autofocus.")
             return False

        print("Starting autofocus...")
        af_cfg = CONFIG['capture'].get('autofocus', {}) # Get autofocus config section
        # Check for absolute focus support
        abs_pos_prop = self.focuser.getNumber("ABS_FOCUS_POSITION")
        if not abs_pos_prop:
            print("Error: Focuser does not support absolute position (ABS_FOCUS_POSITION). Autofocus requires absolute positioning.")
            return False

        # Timeout for the entire routine
        routine_timeout_at = time.time() + af_cfg.get('max_duration_s', 180.0) # Default 3 mins

        pos_widget = abs_pos_prop.findWidgetByName("FOCUS_ABSOLUTE_POSITION")
        if not pos_widget:
             print("Error: Cannot find FOCUS_ABSOLUTE_POSITION widget on focuser.")
             return False

        # Read current position and limits safely
        try:
             current_pos = int(pos_widget.value)
             focuser_min = int(pos_widget.min)
             focuser_max = int(pos_widget.max)
             print(f"  - Current focuser position: {current_pos}")
             print(f"  - Focuser limits: [{focuser_min} - {focuser_max}]")
        except Exception as e:
             print(f"Error reading focuser position or limits: {e}")
             return False


        # --- Calculate Scan Positions ---
        step_base = af_cfg.get('step_base', 50)
        scan_range = af_cfg.get('scan_range', 5) # Number of steps +/- from center

        # Optional: Adjust step size based on target altitude (empirical adjustment)
        # Avoid division by zero or negative altitude
        if alt_ft and alt_ft > 500: # Apply only if altitude is known and reasonable
            # Example: Smaller steps for higher altitude (less atmospheric effect?)
            # This logic might need tuning based on experience
            step_size = max(10, int(step_base * (20000.0 / max(500.0, alt_ft))))
            print(f"  - Adjusting step size based on altitude ({alt_ft:.0f} ft) to: {step_size}")
        else:
            step_size = step_base
            print(f"  - Using base step size: {step_size}")

        # Generate scan positions around the current position
        raw_positions = [current_pos + i * step_size for i in range(-scan_range, scan_range + 1)]
        # Filter positions to be within the focuser's hardware limits
        positions = [p for p in raw_positions if focuser_min <= p <= focuser_max]

        if not positions:
            print(f"Error: No valid scan positions calculated within hardware limits [{focuser_min}-{focuser_max}] around {current_pos} with step {step_size}.")
            return False
        print(f"  - Scanning positions: {positions}")

        # --- Perform Scan ---
        sharpness_scores: Dict[int, float] = {} # Store sharpness for each position
        # Use exposure from autofocus config, fallback to guide exposure
        scan_exposure = af_cfg.get('exposure_s', CONFIG['capture']['guiding'].get('guide_exposure_s', 0.5))

        for pos in positions:
            # Check for overall routine timeout
            if time.time() > routine_timeout_at:
                print("Autofocus aborted: Routine exceeded max duration.")
                break # Exit scan loop

            # --- Move Focuser ---
            pos_widget.value = pos
            self.sendNewNumber(abs_pos_prop)

            # Wait for focuser move to complete (with timeout and shutdown check)
            move_timeout_at = time.time() + 20.0 # Generous timeout for focuser move
            while time.time() < move_timeout_at:
                 current_pos_prop = self.focuser.getNumber("ABS_FOCUS_POSITION") # Re-fetch property
                 if not current_pos_prop or current_pos_prop.s != PyIndi.IPS_BUSY:
                     break # Exit wait loop if not busy or property lost
                 if self.shutdown_event and self.shutdown_event.is_set():
                     print("Autofocus aborted while moving focuser.")
                     return False # Abort completely
                 time.sleep(0.2)
            else: # Loop finished due to timeout
                 print(f"Warning: Focuser move to {pos} timed out.")
                 continue # Skip taking image for this position


            # --- Capture and Measure ---
            try:
                # Use a unique label for autofocus snaps
                snap_label = f"autofocus_p{pos}"
                snap_path = self.snap_image(snap_label)
                if not snap_path:
                     print(f"  - Warning: Failed to capture image at position {pos}.")
                     continue # Skip measurement if capture failed

                # --- FIX: Load data once, call in-memory function ---
                guide_data = _load_fits_data(snap_path)
                if guide_data is None:
                    print(f"  - Warning: Failed to load image data for {snap_path}.")
                    continue

                sharpness = _measure_sharpness_from_data(guide_data)
                # --- End FIX ---
                
                # Store valid sharpness scores (measure_sharpness returns 0 on error)
                if sharpness > 0:
                    sharpness_scores[pos] = sharpness
                    print(f"  - Position {pos}: Sharpness {sharpness:.2f}")
                else:
                     print(f"  - Warning: Failed to measure sharpness at position {pos}.")

            except Exception as e:
                 print(f"  - Error during capture/measure at position {pos}: {e}")
                 # Continue to next position even if one fails

        # --- Find Best Position ---
        if not sharpness_scores:
            print("Error: Focus failed - No valid sharpness scores were recorded during scan.")
            return False

        # Find position with the maximum sharpness score
        best_pos = max(sharpness_scores, key=sharpness_scores.get)
        best_sharp = sharpness_scores[best_pos]

        # Check if best sharpness meets minimum threshold from config
        min_sharp_threshold = af_cfg.get('sharpness_threshold', 20.0) # Default 20
        if best_sharp < min_sharp_threshold:
            print(f"Error: Focus failed - Best sharpness ({best_sharp:.2f}) is below threshold ({min_sharp_threshold:.2f}). Scene might be too blurry or empty.")
             # Optionally, move back to starting position? For now, just fail.
            return False

        # --- Move to Best Position ---
        print(f"  - Best focus found at position {best_pos} (Sharpness: {best_sharp:.2f}). Moving...")
        pos_widget.value = best_pos
        self.sendNewNumber(abs_pos_prop)

        # Wait for final move (with timeout and shutdown check)
        move_timeout_at = time.time() + 20.0
        while time.time() < move_timeout_at:
             current_pos_prop = self.focuser.getNumber("ABS_FOCUS_POSITION")
             if not current_pos_prop or current_pos_prop.s != PyIndi.IPS_BUSY:
                 break
             if self.shutdown_event and self.shutdown_event.is_set():
                 print("Autofocus aborted while returning to best position.")
                 return False
             time.sleep(0.2)
        else: # Loop finished due to timeout
             print(f"Warning: Final focuser move to {best_pos} timed out.")
             # Return success anyway? Or failure? Let's return success but log warning.
             pass

        print(f"Autofocus successful. Final position: {best_pos}")

        # Log autofocus results
        log_path = os.path.join(LOG_DIR, 'focus_logs.json')
        log_entry = {
            'timestamp': time.time(),
            'target_alt_ft': alt_ft,
            'best_pos': best_pos,
            'best_sharpness': best_sharp,
            'scan_details': sharpness_scores # Log all scores
        }
        try:
             # Use the robust append_to_json from storage module
             append_to_json([log_entry], log_path)
        except Exception as e:
             print(f"Warning: Could not write autofocus log: {e}")

        return True


    def capture_image(self, exposure_s: float, save_path: str) -> Optional[str]:
        """Captures a single image, saves to FITS, adds header info, and creates PNG."""
        # Ensure save directory exists
        try:
             os.makedirs(os.path.dirname(save_path), exist_ok=True)
        except OSError as e:
             print(f"Error creating directory for image {save_path}: {e}")
             return None # Cannot save if directory fails


        # --- Dry Run Simulation --- Using Original working version logic ---
        if CONFIG['development']['dry_run']:
            print(f"[DRY RUN] Simulating capture of: {os.path.basename(save_path)}")
            try:
                png_path = os.path.splitext(save_path)[0] + ".png"
                img = np.full((180, 320), 40, np.uint8) # Original used uint8
                cv2.putText(img, "DRY RUN", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 200, 2)
                cv2.putText(img, os.path.basename(save_path), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 180, 1)
                cv2.imwrite(png_path, img)

                # Save the image *with text* to FITS, cast to uint16
                hdu = fits.PrimaryHDU(img.astype(np.uint16))
                hdu.header['EXPTIME'] = (exposure_s, 'Simulated exposure time (s)') # Add header info
                hdu.header['DATE-OBS'] = Time.now().isot
                hdu.header['INSTRUME'] = 'SIMULATOR'
                fits.HDUList([hdu]).writeto(save_path, overwrite=True)

                return save_path
            except Exception as e:
                print(f"[DRY RUN] Error creating simulated files for {save_path}: {e}")
                return None
        # --- End Dry Run ---


        # --- Real Hardware Capture ---
        if not self.camera:
             print("Error: Camera device not available for capture.")
             return None

        # Clamp exposure to configured min/max limits
        min_exp = float(CONFIG['capture'].get('exposure_min_s', 0.001))
        max_exp = float(CONFIG['capture'].get('exposure_max_s', 60.0)) # Increased default max

        clamped_exp = max(min_exp, min(max_exp, exposure_s))
        if abs(clamped_exp - exposure_s) > 1e-6: # Use tolerance for float comparison
            print(f"  - Warning: Requested exposure {exposure_s:.4f}s clamped to {clamped_exp:.4f}s (limits: [{min_exp:.4f} - {max_exp:.4f}]s).")
        exposure_s = clamped_exp # Use the clamped value

        # Get exposure property
        exp_prop = self.camera.getNumber("CCD_EXPOSURE")
        if not exp_prop:
             print("Error: CCD_EXPOSURE property not found on camera.")
             return None

        # --- Prepare for Capture ---
        # Drain any old BLOBs before starting exposure
        self._drain_blobs(self._blob_name)

        # Set exposure time
        exp_widget = exp_prop.findWidgetByName("CCD_EXPOSURE_VALUE")
        if not exp_widget:
             print("Error: CCD_EXPOSURE_VALUE widget not found.")
             return None
        exp_widget.value = exposure_s
        self.sendNewNumber(exp_prop)

        # --- Wait for Exposure to Complete ---
        # Monitor exposure property state until it's no longer busy
        exposure_start_time = time.time()
        # Timeout slightly longer than exposure + readout guess
        exposure_timeout_at = exposure_start_time + exposure_s + 15.0
        while time.time() < exposure_timeout_at:
             current_exp_prop = self.camera.getNumber("CCD_EXPOSURE") # Re-fetch
             if not current_exp_prop or current_exp_prop.s != PyIndi.IPS_BUSY:
                  break # Exit loop if not busy or property lost
             if self.shutdown_event and self.shutdown_event.is_set():
                  print("Capture aborted during exposure by shutdown signal.")
                  return None # Abort capture
             time.sleep(0.1) # Short sleep while busy
        else: # Loop finished due to timeout
             print(f"Warning: Exposure command timed out after {time.time() - exposure_start_time:.1f}s (requested {exposure_s:.3f}s).")
             # Continue anyway, image might still arrive
             pass


        # --- Wait for Image BLOB ---
        # Timeout for receiving the image data
        blob_timeout_at = time.time() + max(15.0, exposure_s * 1.5 + 10.0) # Generous BLOB timeout

        while time.time() < blob_timeout_at:
            if self.shutdown_event and self.shutdown_event.is_set():
                 print("Capture aborted while waiting for image data by shutdown signal.")
                 return None # Abort capture

            # Get the BLOB property using the detected name
            blob_prop = self.camera.getBLOB(self._blob_name)

            # Check if BLOB property exists and has data
            if (blob_prop and
                len(blob_prop) > 0 and
                hasattr(blob_prop[0], 'bloblen') and
                blob_prop[0].bloblen > 0 and
                # Check state is OK (data ready)
                blob_prop.s == PyIndi.IPS_OK):

                # --- Save FITS Data ---
                try:
                    fits_data = blob_prop[0].getblobdata()
                    with open(save_path, 'wb') as f:
                        f.write(fits_data)
                    print(f"  - Image received and saved to {os.path.basename(save_path)}")
                except Exception as e:
                     print(f"Error saving FITS data to {save_path}: {e}")
                     return None # Abort if save fails


                # --- Annotate FITS Header ---
                try:
                    with fits.open(save_path, mode="update") as hdul:
                        hdr = hdul[0].header
                        # Standard headers
                        hdr['EXPTIME'] = (float(exposure_s), 'Exposure time (s)')
                        hdr['DATE-OBS'] = Time.now().isot # Use current time as DATE-OBS
                        hdr['INSTRUME'] = (self.camera.getDeviceName(), 'Instrument name')

                        # Add camera settings if available
                        b = self.camera.getNumber("CCD_BINNING")
                        if b:
                            xw, yw = b.findWidgetByName("HOR_BIN"), b.findWidgetByName("VER_BIN")
                            if xw and yw:
                                hdr['XBINNING'] = int(xw.value)
                                hdr['YBINNING'] = int(yw.value)

                        g = self.camera.getNumber("CCD_GAIN")
                        if g:
                             gw = g.findWidgetByName("GAIN") or g.findWidgetByName("CCD_GAIN_VALUE")
                             if gw: hdr['GAIN'] = float(gw.value)

                        off = self.camera.getNumber("CCD_OFFSET")
                        if off:
                             ow = off.findWidgetByName("OFFSET") or off.findWidgetByName("CCD_OFFSET_VALUE")
                             if ow: hdr['OFFSET'] = float(ow.value)

                        t = self.camera.getNumber("CCD_TEMPERATURE")
                        if t:
                            tw = t.findWidgetByName("CCD_TEMPERATURE_VALUE")
                            if tw: hdr['CCD-TEMP'] = (float(tw.value), 'CCD Temperature (C)')

                        # Add observer location
                        hdr['OBSERVER'] = 'ADS-B Tracker' # Generic observer name
                        hdr['SITELAT'] = (self.observer_loc.lat.deg, 'Observer Latitude (deg)')
                        hdr['SITELONG'] = (self.observer_loc.lon.deg, 'Observer Longitude (deg)')
                        hdr['SITEELEV'] = (self.observer_loc.height.to_value(u.m), 'Observer Altitude (m)')

                        hdul.flush() # Save header changes
                except Exception as e:
                    # Log warning but continue - header annotation is non-critical
                    print(f"Warning: Could not fully annotate FITS header for {save_path}: {e}")

                # --- Create PNG Preview ---
                try:
                    # Use the robust save_png_preview from image_analyzer
                    png_path = save_png_preview(save_path)
                    if not png_path:
                         print(f"Warning: Could not generate PNG preview for {save_path} (save_png_preview returned empty).")
                except Exception as e:
                    print(f"Warning: Error calling save_png_preview for {save_path}: {e}")

                # Successfully captured, saved, and processed
                return save_path

            # Short sleep before checking for BLOB again
            time.sleep(0.1)

        # --- Timeout Waiting for BLOB ---
        print(f"Error: Image BLOB ('{self._blob_name}') did not arrive from camera within timeout for exposure {exposure_s:.3f}s.")
        return None # Return None on timeout


    def guide_pulse(self, direction: str, duration_ms: int):
        """Sends a guide pulse to the mount."""
        duration_ms = max(0, int(duration_ms)) # Ensure non-negative integer
        # Validate direction
        if direction not in ('N','S','E','W'):
            print(f"Warning: Invalid guide direction '{direction}'")
            return

        if CONFIG['development']['dry_run']:
            print(f"[DRY RUN] Simulating {duration_ms}ms guide pulse: {direction}")
            # Simulate the duration of the pulse
            time.sleep(duration_ms / 1000.0)
            return

        if not self.mount:
             print("Error: Mount device not available for guiding.")
             return

        # --- Try Timed Guiding First (Preferred) ---
        timed_prop_name, timed_widget_name = "", ""
        if direction in ['N', 'S']:
            timed_prop_name = "TELESCOPE_TIMED_GUIDE_NS"
            timed_widget_name = "TIMED_GUIDE_N" if direction == 'N' else "TIMED_GUIDE_S"
        elif direction in ['W', 'E']: # Use W/E consistently
            timed_prop_name = "TELESCOPE_TIMED_GUIDE_WE"
            timed_widget_name = "TIMED_GUIDE_W" if direction == 'W' else "TIMED_GUIDE_E"

        timed_prop = self.mount.getNumber(timed_prop_name) if timed_prop_name else None
        if timed_prop:
            # Check if property is writable
            if timed_prop.p == PyIndi.IP_RO: # Read Only
                 print(f"  - Info: Timed guide property '{timed_prop_name}' is read-only. Falling back.")
            else:
                widget = timed_prop.findWidgetByName(timed_widget_name)
                if widget:
                    # Check if widget is writable
                    if widget.p == PyIndi.IP_RO:
                        print(f"  - Info: Timed guide widget '{timed_widget_name}' is read-only. Falling back.")
                    else:
                        widget.value = duration_ms
                        self.sendNewNumber(timed_prop)
                        # Optionally wait briefly for command receipt?
                        # time.sleep(0.05)
                        # print(f"  - Sent timed guide pulse: {direction} {duration_ms}ms") # Debug log
                        return # Success using timed pulse

                else:
                    print(f"  - Warning: Timed guide widget '{timed_widget_name}' not found on '{timed_prop_name}'. Falling back.")


        # --- Fallback to Manual On/Off Guiding ---
        # print(f"  - Info: Falling back to manual on/off guide pulse for {direction}.") # Debug log
        prop_name, widget_name = "", ""
        if direction in ['N', 'S']:
            prop_name = "TELESCOPE_MOTION_NS"
            widget_name = "MOTION_NORTH" if direction == 'N' else "MOTION_SOUTH"
        elif direction in ['W', 'E']:
            prop_name = "TELESCOPE_MOTION_WE"
            widget_name = "MOTION_WEST" if direction == 'W' else "MOTION_EAST"
        else:
             # Should not happen due to initial check, but safeguard
             return

        prop = self.mount.getSwitch(prop_name)
        if not prop:
            print(f"Error: Manual guide property '{prop_name}' not found for direction {direction}.")
            return

        # Check if property is writable
        if prop.p == PyIndi.IP_RO:
            print(f"Error: Manual guide property '{prop_name}' is read-only.")
            return

        widget = prop.findWidgetByName(widget_name)
        if not widget:
            print(f"Error: Manual guide widget '{widget_name}' not found in '{prop_name}'.")
            return

        # Check if widget is writable (should be part of a Rule_ANYOFMANY switch)
        if widget.p == PyIndi.IP_RO:
             print(f"Error: Manual guide widget '{widget_name}' is read-only.")
             return

        # --- Execute Pulse ---
        try:
             # Send ON command
             widget.s = PyIndi.ISS_ON
             # Need to ensure other switches in the group are OFF for Rule_ONEOFMANY
             for w in prop:
                  if w.name != widget.name: w.s = PyIndi.ISS_OFF
             self.sendNewSwitch(prop)

             # Wait for the specified duration
             time.sleep(duration_ms / 1000.0)

             # Send OFF command
             widget.s = PyIndi.ISS_OFF
             # Ensure other switches remain OFF
             for w in prop: w.s = PyIndi.ISS_OFF
             self.sendNewSwitch(prop)

        except Exception as e:
             print(f"Error during manual guide pulse ({direction}, {duration_ms}ms): {e}")
             # Attempt to send OFF command again in case of error during sleep
             try:
                  if widget: widget.s = PyIndi.ISS_OFF
                  if prop:
                       for w in prop: w.s = PyIndi.ISS_OFF
                       self.sendNewSwitch(prop)
             except Exception:
                  print("  - Error trying to ensure guide pulse OFF after failure.")


    def park_mount(self) -> bool:
        """Parks the mount if supported."""
        if CONFIG['development']['dry_run']:
            print("[DRY RUN] Simulating park mount.")
            self.simulated_az_el = (180.0, 0.0) # Assume park position is Az=180, El=0
            return True

        if not self.mount:
             print("Error: Mount device not available for parking.")
             return False

        # Find park property (common names)
        park_prop = self.mount.getSwitch("TELESCOPE_PARK") or self.mount.getSwitch("MOUNT_PARK")
        if not park_prop:
            print("Warning: Mount does not support PARK command (TELESCOPE_PARK or MOUNT_PARK not found).")
            return False

        # Check if writable
        if park_prop.p == PyIndi.IP_RO:
             print("Error: Park property is read-only.")
             return False

        # Find the specific PARK switch (or assume the first one if named differently)
        park_widget = park_prop.findWidgetByName("PARK") or park_prop.findWidgetByName("MOUNT_PARK")
        # Fallback to the first switch in the property if specific name not found
        if not park_widget and len(park_prop) > 0:
             park_widget = park_prop[0]

        if not park_widget:
             print("Error: Could not find a park switch widget within the park property.")
             return False

        # Check if widget is writable
        if park_widget.p == PyIndi.IP_RO:
             print(f"Error: Park switch '{park_widget.name}' is read-only.")
             return False

        # --- Send Park Command ---
        # Set the park widget ON, others OFF (for ONEOFMANY rule)
        for w in park_prop:
             w.s = PyIndi.ISS_ON if w.name == park_widget.name else PyIndi.ISS_OFF
        self.sendNewSwitch(park_prop)

        print("Parking mount...")
        park_timeout_at = time.time() + 180.0 # Generous timeout for parking (e.g., 3 mins)

        # --- Wait for Park Completion ---
        while time.time() < park_timeout_at:
            current_park_prop = self.mount.getSwitch(park_prop.name) # Re-fetch property by name
            if not current_park_prop:
                 print("Warning: Park property lost during parking.")
                 return False # Abort if property disappears

            # Check property state
            if current_park_prop.s == PyIndi.IPS_BUSY:
                # Check for shutdown signal while parking
                if self.shutdown_event and self.shutdown_event.is_set():
                    print("Park aborted by shutdown signal.")
                    # Optionally try to abort park if mount supports it
                    # abort_prop = self.mount.getSwitch("TELESCOPE_ABORT_MOTION") ...
                    return False
                time.sleep(1) # Wait while busy
                continue # Continue polling

            # If not busy, check final state
            if current_park_prop.s == PyIndi.IPS_OK:
                # Additionally check if the PARK widget itself is OFF (some drivers do this on completion)
                final_park_widget = current_park_prop.findWidgetByName(park_widget.name)
                if final_park_widget and final_park_widget.s == PyIndi.ISS_OFF:
                     print("Mount parked successfully (widget OFF).")
                     return True
                # Or if the property state is OK (some drivers leave widget ON)
                print("Mount parked successfully (property OK).")
                return True

            if current_park_prop.s == PyIndi.IPS_ALERT:
                print("Error: Mount park command failed (Alert state). Check INDI logs.")
                return False

            # If state is IDLE, might indicate completion or unexpected state
            if current_park_prop.s == PyIndi.IPS_IDLE:
                 print("Warning: Park property became IDLE. Assuming park completed or was interrupted.")
                 # Check final position? For now, assume success if idle after busy.
                 return True


            # Break loop if state is unexpected but not busy
            break

        # --- Timeout ---
        print("Warning: Mount park command timed out.")
        return False


    def snap_image(self, label: str) -> Optional[str]:
        """Captures a single image with appropriate exposure based on label."""
        # Construct save path within logs/images
        images_dir = os.path.join(LOG_DIR, 'images')
        # Ensure directory exists (moved here from capture_image)
        try:
             os.makedirs(images_dir, exist_ok=True)
        except OSError as e:
            print(f"Error creating images directory {images_dir}: {e}")
            return None

        # Generate filename with nanosecond precision to avoid collisions
        filename = f"snap_{label}_{time.time_ns()}.fits"
        path = os.path.join(images_dir, filename)

        # Determine exposure based on label context
        exposure_s = CONFIG['capture'].get('snap_exposure_s', 0.1) # Default snap exposure
        if label.startswith("autofocus"):
             # Use autofocus exposure, fallback to guide, then default snap
            exp = CONFIG['capture'].get('autofocus', {}).get('exposure_s')
            if exp is None:
                 exp = CONFIG['capture'].get('guiding', {}).get('guide_exposure_s', exposure_s)
            exposure_s = exp
        elif label.startswith("guide") or "_g" in label: # Assume labels like "icao_g1" are guide frames
            # Use guide exposure, fallback to default snap
            exp = CONFIG['capture'].get('guiding', {}).get('guide_exposure_s', exposure_s)
            exposure_s = exp
        # Add more label cases if needed (e.g., "calibration")

        # Call the main capture function
        return self.capture_image(exposure_s, path)


    def capture_sequence(self, icao: str, exposure_s: float, num_images: Optional[int] = None) -> List[str]:
        """
        Captures a burst of images and schedules them for background stacking.
        Returns the list of successfully captured FITS paths.
        """
        if num_images is None:
            # Get number from config, default to 5
            num_images = int(CONFIG['capture'].get('num_sequence_images', 5))

        # Ensure num_images is positive
        if num_images <= 0: return []

        print(f"  - Starting capture sequence: {num_images} frames, {exposure_s:.3f}s exposure for {icao}.")

        paths: List[str] = []
        images_dir = os.path.join(LOG_DIR, 'images')
        # Ensure directory exists (though capture_image also checks)
        os.makedirs(images_dir, exist_ok=True)

        for i in range(num_images):
            # Check for shutdown signal before each capture
            if self.shutdown_event and self.shutdown_event.is_set():
                print(f"  - Capture sequence for {icao} interrupted by shutdown signal after {len(paths)} frames.")
                break # Exit capture loop

            # Generate unique filename for each frame in the sequence
            # Using nanoseconds for uniqueness
            filename = f"capture_{icao}_{time.time_ns()}_{i+1}_of_{num_images}.fits"
            path = os.path.join(images_dir, filename)

            try:
                # Call capture_image for the individual frame
                captured_path = self.capture_image(exposure_s, path)
                if captured_path:
                    paths.append(captured_path)
                    print(f"    - Captured frame {i+1}/{num_images} -> {os.path.basename(captured_path)}")
                else:
                     # Log if capture_image returned None (indicating failure)
                     print(f"    - Failed to capture frame {i+1}/{num_images} (capture_image returned None).")
                     # Optionally break or continue? Continuing for now.

            except Exception as e:
                 # Catch unexpected errors during the capture call
                 print(f"    - Error capturing frame {i+1}/{num_images}: {e}")
                 # Stop sequence on error? Or just skip frame? Skipping for now.
                 continue

        # --- Schedule Stacking (only if images were captured) ---
        if paths:
            try:
                # Gather relevant camera settings for the manifest
                cam_settings = self.get_hardware_status() # Get current settings
                stacking_meta = {
                    "gain": cam_settings.get("camera_gain"),
                    "offset": cam_settings.get("camera_offset"),
                    "binning": cam_settings.get("camera_binning"),
                    "temperature": cam_settings.get("camera_temp"),
                    "exposure_s": exposure_s # Include exposure used for the sequence
                }
                # Use a unique sequence ID including timestamp
                sequence_id = f"{icao}_{int(time.time())}"

                # Call the orchestrator to handle stacking in background
                schedule_stack_and_publish(sequence_id, paths, stacking_meta)
                print(f"  - Submitted sequence {sequence_id} ({len(paths)} frames) for background stacking.")
            except Exception as e:
                # Log error if scheduling fails, but don't crash main thread
                print(f"Warning: Could not schedule stacking for sequence {sequence_id}: {e}")
        else:
             print(f"  - No images captured in sequence for {icao}, skipping stacking.")

        # Return the list of paths that were successfully captured
        return paths

    def disconnect(self):
        """Disconnects from the INDI server."""
        if CONFIG['development']['dry_run']:
            print("[DRY RUN] Simulated disconnect.")
            return
        # Check connection before disconnecting
        if self.isConnected():
             print("Disconnecting from INDI server...")
             self.disconnectServer()
        # Clear device references
        self.mount = None
        self.camera = None
        self.focuser = None