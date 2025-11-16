#hardware_control.py
"""
Hardware Control module:
"""
import PyIndi
import time
import os
import numpy as np
import cv2
import threading
import queue
import logging
from typing import List, Optional, Callable
from storage import append_to_json
from config_loader import CONFIG, LOG_DIR
from image_analyzer import measure_sharpness, save_png_preview
from coord_utils import get_sun_azel, slew_time_needed, get_altaz_frame, angular_sep_deg, distance_km # Use patched distance_km
from astropy.coordinates import SkyCoord, AltAz, EarthLocation
import astropy.units as u
from astropy.time import Time
from astropy.io import fits
from stack_orchestrator import schedule_stack_and_publish

logger = logging.getLogger(__name__)


class IndiController(PyIndi.BaseClient):
    """INDI client for mount, camera, and focuser."""
    def __init__(self):
        super().__init__() # Call BaseClient init first
        self.shutdown_event: Optional[threading.Event] = threading.Event()

        # BLOB handling structures
        self._blob_lock = threading.Lock()
        self._exposure_queue = queue.Queue()
        self._received_blobs = {}
        self._blob_name = "CCD1" # Default

        is_dry_run = CONFIG['development']['dry_run']

        if is_dry_run:
            logger.info("[DRY RUN] IndiController initialized in dry-run mode.")
            self.simulated_az_el = (180.0, 45.0)
            # Dummy devices for dry run to prevent errors later if attributes are accessed
            self.mount = None
            self.camera = None
            self.focuser = None
            # No early return here anymore
        else:
            # --- Real Hardware Init ---
            logger.info("Connecting to INDI server...")
            self.setServer(CONFIG['hardware']['indi_host'], CONFIG['hardware']['indi_port'])
            if not self.connectServer():
                raise ConnectionError("Failed to connect to INDI server")
            logger.info("Waiting for devices...")
            time.sleep(5)

            self.mount = self.getDevice(CONFIG['hardware']['mount_device_name'])
            self.camera = self.getDevice(CONFIG['hardware']['camera_device_name'])
            self.focuser = self.getDevice(CONFIG['hardware']['focuser_device_name'])
            if not all([self.mount, self.camera, self.focuser]):
                device_names = [d.getDeviceName() for d in self.getDevices()]
                logger.critical(f"ERROR: One or more hardware devices not found. Available devices: {device_names}")
                raise ValueError("One or more hardware devices not found")

            # Detect BLOB name only for real hardware
            self._blob_name = self._detect_ccd_blob_name()
            logger.info(f"  Setting BLOB mode for camera '{self.camera.getDeviceName()}' property '{self._blob_name}'")
            self.setBLOBMode(PyIndi.B_CLIENT, self.camera.getDeviceName(), self._blob_name)
            time.sleep(1)
            blob_prop = self.camera.getBLOB(self._blob_name)
            if blob_prop is None:
                 logger.warning(f"  - Warning: BLOB vector '{self._blob_name}' not available. Subscribing to all BLOBs.")
                 self.setBLOBMode(PyIndi.B_CLIENT, self.camera.getDeviceName(), None)

            logger.info("--- Hardware Connected & Configured ---")
            # Print driver info only for real hardware
            for device, name in [(self.mount, "Mount"), (self.camera, "Camera"), (self.focuser, "Focuser")]:
                 dev_name_str = getattr(device, 'getDeviceName', lambda: 'UNKNOWN')()
                 driver_info_prop = device.getText("DRIVER_INFO") if device else None
                 driver_info_text = "Driver info not available."
                 if driver_info_prop and len(driver_info_prop) > 0:
                      driver_info_text = getattr(driver_info_prop[0], 'text', driver_info_text)
                 logger.info(f"  - {name} ({dev_name_str}): {driver_info_text}")

            # Configure Camera Settings only for real hardware
            cam_specs = CONFIG['camera_specs']
            upload_mode = self._wait_switch(self.camera, "UPLOAD_MODE")
            if upload_mode:
                widget = upload_mode.findWidgetByName("UPLOAD_CLIENT")
                if widget and widget.s != PyIndi.ISS_ON:
                    self._set_switch_state(upload_mode, "UPLOAD_CLIENT", PyIndi.ISS_ON)
                    logger.info("  - Set Camera: Upload mode to 'Client'.")

            if cam_specs.get('cooling', {}).get('enabled', False):
                 cooler_switch = self._wait_switch(self.camera, "CCD_COOLER")
                 if cooler_switch:
                     widget = cooler_switch.findWidgetByName("COOLER_ON")
                     if widget and widget.s != PyIndi.ISS_ON:
                          self._set_switch_state(cooler_switch, "COOLER_ON", PyIndi.ISS_ON)
                 temp_prop = self._wait_number(self.camera, "CCD_TEMPERATURE")
                 if temp_prop:
                     widget = temp_prop.findWidgetByName("CCD_TEMPERATURE_VALUE")
                     if widget:
                         setpoint = float(cam_specs['cooling']['setpoint_c'])
                         if abs(widget.value - setpoint) > 0.1:
                              widget.value = setpoint
                              self.sendNewNumber(temp_prop)
                              logger.info(f"  - Set Camera: Cooling enabled, setpoint {setpoint}°C.")

            if 'binning' in cam_specs:
                 bin_prop = self._wait_number(self.camera, "CCD_BINNING")
                 if bin_prop:
                     hor_widget = bin_prop.findWidgetByName("HOR_BIN")
                     ver_widget = bin_prop.findWidgetByName("VER_BIN")
                     bin_x = int(cam_specs['binning']['x'])
                     bin_y = int(cam_specs['binning']['y'])
                     if hor_widget and ver_widget and (hor_widget.value != bin_x or ver_widget.value != bin_y):
                         hor_widget.value = bin_x
                         ver_widget.value = bin_y
                         self.sendNewNumber(bin_prop)
                         logger.info(f"  - Set Camera: Binning to {bin_x}x{bin_y}.")

            if 'gain' in cam_specs:
                 gain_prop = self._wait_number(self.camera, "CCD_GAIN")
                 if gain_prop:
                     widget = gain_prop.findWidgetByName("GAIN")
                     gain_val = float(cam_specs['gain'])
                     if widget and abs(widget.value - gain_val) > 0.1:
                         widget.value = gain_val
                         self.sendNewNumber(gain_prop)
                         logger.info(f"  - Set Camera: Gain to {gain_val}.")

            if 'offset' in cam_specs:
                 offset_prop = self._wait_number(self.camera, "CCD_OFFSET")
                 if offset_prop:
                     widget = offset_prop.findWidgetByName("OFFSET")
                     offset_val = float(cam_specs['offset'])
                     if widget and abs(widget.value - offset_val) > 0.1:
                         widget.value = offset_val
                         self.sendNewNumber(offset_prop)
                         logger.info(f"  - Set Camera: Offset to {offset_val}.")

            logger.info("---------------------------------------")

        # Observer location is needed in both modes for coordinate conversions
        obs_cfg = CONFIG['observer']
        self.observer_loc = EarthLocation(
            lat=obs_cfg['latitude_deg']*u.deg,
            lon=obs_cfg['longitude_deg']*u.deg,
            height=obs_cfg['altitude_m']*u.m
        )
        logger.info("--- IndiController Init Complete ---") # Add marker


    # --- Helper to safely set switch states ---
    def _set_switch_state(self, switch_vector, widget_name, state):
         """Sets a single widget in a switch vector to the desired state."""
         widget = switch_vector.findWidgetByName(widget_name)
         if widget:
             # Ensure all others are OFF if it's a RULE_ONE_OF_MANY
             if switch_vector.r == PyIndi.SR_1OFMANY:
                 for i in range(switch_vector.size()):
                     switch_vector[i].s = PyIndi.ISS_OFF
             widget.s = state
             self.sendNewSwitch(switch_vector)
         else:
             logger.warning(f"Warning: Widget '{widget_name}' not found in switch vector '{switch_vector.name}'.")


    def _wait_prop(self, get_func, dev, name, timeout=5.0):
        """Waits for an INDI property to become available."""
        if not dev: return None # Handle case where device itself is None
        t0 = time.time()
        while time.time() - t0 < timeout:
            prop = get_func(dev, name)
            if prop:
                return prop
            time.sleep(0.2) # Longer sleep to reduce busy-waiting
        dev_name = getattr(dev, "getDeviceName", lambda: "UNKNOWN")()
        logger.warning(f"Warning: Property '{name}' not available on device '{dev_name}' after {timeout}s.")
        return None

    def _wait_number(self, dev, name, timeout=5.0):
        return self._wait_prop(PyIndi.BaseDevice.getNumber, dev, name, timeout)

    def _wait_switch(self, dev, name, timeout=5.0):
        return self._wait_prop(PyIndi.BaseDevice.getSwitch, dev, name, timeout)


    def _detect_ccd_blob_name(self) -> str:
        """Detects the likely name of the primary image BLOB property."""
        # Ensure camera device exists
        if not self.camera:
            logger.error("  - Error: Camera device not initialized during BLOB detection.")
            return "CCD1" # Fallback

        # Check common names
        prop_timeout = 2.0
        for name in ("CCD1", "CCD_IMAGE", "CCD Image", "PRIMARY_CCD", "VIDEO_STREAM"):
             t0 = time.time()
             while time.time() - t0 < prop_timeout:
                 blob_prop = self.camera.getBLOB(name)
                 if blob_prop is not None:
                     logger.info(f"  - Detected camera BLOB property: {name}")
                     return name
                 time.sleep(0.1)

        logger.warning("  - Warning: Could not detect primary BLOB property, falling back to 'CCD1'.")
        return "CCD1"

    def newBLOB(self, bp):
        """Handles incoming BLOB data from the INDI server."""
        is_expected_blob = (bp.name == self._blob_name) or (self._blob_name == "CCD1" and bp.name == "CCD1")
        if not is_expected_blob:
            PyIndi.BaseClient.newBLOB(self, bp); return
        blob_data = None
        try:
            if len(bp) > 0 and bp[0].bloblen > 0: blob_data = bp[0].getblobdata()
            else: logger.warning("  - Warning: Received empty BLOB."); PyIndi.BaseClient.newBLOB(self, bp); return
        except Exception as e: logger.error(f"  - Error getting BLOB data: {e}"); PyIndi.BaseClient.newBLOB(self, bp); return
        with self._blob_lock:
            try:
                exposure_token = self._exposure_queue.get_nowait()
                self._received_blobs[exposure_token] = blob_data
            except queue.Empty: logger.warning("  - Warning: Received BLOB but no exposure was pending. Discarding.")
            except Exception as e: logger.error(f"  - Error processing BLOB in newBLOB handler: {e}")
        PyIndi.BaseClient.newBLOB(self, bp)

    def get_hardware_status(self) -> dict:
        status = {}
        try:
            status["mount_az_el"] = self._get_current_az_el_internal() if not CONFIG['development']['dry_run'] else self.simulated_az_el
        except Exception as e:
             status["mount_az_el"] = (0.0, 0.0)
             if not CONFIG['development']['dry_run']:
                  logger.warning(f"Warning: could not get mount az/el: {e}")
        
        if not CONFIG['development']['dry_run']:
            if self.mount:
                pier_side_prop = self.mount.getSwitch("TELESCOPE_PIER_SIDE")
                if pier_side_prop:
                    for element in pier_side_prop:
                         if element.s == PyIndi.ISS_ON:
                             status["mount_pier_side"] = element.label
                             break
            if self.camera:
                 temp_prop = self.camera.getNumber("CCD_TEMPERATURE")
                 if temp_prop:
                     widget = temp_prop.findWidgetByName("CCD_TEMPERATURE_VALUE")
                     if widget: status["camera_temp"] = round(widget.value, 2)
                 
                 cooler_prop = self.camera.getNumber("CCD_COOLER_POWER")
                 if cooler_prop:
                     widget = cooler_prop.findWidgetByName("CCD_COOLER_VALUE")
                     if widget: status["camera_cooler_power"] = round(widget.value, 1)

                 bin_prop = self.camera.getNumber("CCD_BINNING")
                 if bin_prop:
                     w1 = bin_prop.findWidgetByName("HOR_BIN")
                     w2 = bin_prop.findWidgetByName("VER_BIN")
                     if w1 and w2: status["camera_binning"] = {"x": int(w1.value), "y": int(w2.value)}

                 gain_prop = self.camera.getNumber("CCD_GAIN")
                 if gain_prop:
                     widget = gain_prop.findWidgetByName("GAIN")
                     if widget: status["camera_gain"] = int(widget.value)

                 offset_prop = self.camera.getNumber("CCD_OFFSET")
                 if offset_prop:
                     widget = offset_prop.findWidgetByName("OFFSET")
                     if widget: status["camera_offset"] = int(widget.value)
            
            if self.focuser:
                 pos_prop = self.focuser.getNumber("ABS_FOCUS_POSITION")
                 if pos_prop:
                     widget = pos_prop.findWidgetByName("FOCUS_ABSOLUTE_POSITION")
                     if widget: status["focuser_pos"] = int(widget.value)

                 focuser_temp_prop = self.focuser.getNumber("FOCUSER_TEMPERATURE")
                 if focuser_temp_prop:
                     widget = focuser_temp_prop.findWidgetByName("FOCUSER_TEMPERATURE_VALUE")
                     if widget: status["focuser_temp"] = round(widget.value, 1)
        else: # Dry run defaults
             status.update({"camera_temp": -10.1, "camera_cooler_power": 45.5, "focuser_pos": 12345,
                            "focuser_temp": 22.8, "mount_pier_side": "East", "camera_binning": {"x": 1, "y": 1},
                            "camera_gain": 100, "camera_offset": 50})
        return status

    def _get_current_az_el_internal(self) -> tuple:
        if not self.mount: raise ValueError("Mount device not initialized")
        horiz = self.mount.getNumber("HORIZONTAL_COORD")
        if horiz:
            az_w = horiz.findWidgetByName("AZ")
            alt_w = horiz.findWidgetByName("ALT")
            if az_w and alt_w: return (az_w.value, alt_w.value)
        
        eq = self.mount.getNumber("EQUATORIAL_EOD_COORD")
        if eq:
            ra_w = eq.findWidgetByName("RA")
            dec_w = eq.findWidgetByName("DEC")
            if ra_w and dec_w:
                ra, dec = ra_w.value, dec_w.value
                if not hasattr(self, 'observer_loc'): raise ValueError("Observer location not initialized")
                ra_dec = SkyCoord(ra=ra*u.hourangle, dec=dec*u.deg, frame='icrs')
                altaz_frame = AltAz(obstime=Time.now(), location=self.observer_loc)
                altaz = ra_dec.transform_to(altaz_frame)
                return (altaz.az.deg, altaz.alt.deg)
        raise ValueError("Could not get mount coordinates")

    def slew_to_az_el(self, az: float, el: float, progress_cb: Optional[Callable[[float, float, str], None]] = None) -> bool:
        if CONFIG['development']['dry_run']:
            frame = get_altaz_frame(self.observer_loc)
            max_slew = CONFIG['hardware'].get('max_slew_deg_s', 6.0)
            realistic_slew_time = slew_time_needed(self.simulated_az_el, (az, el), max_slew, frame) if max_slew > 0 else 5.0
            logger.info(f"[DRY RUN] Slewing from {self.simulated_az_el} to az/el: ({az:.2f}, {el:.2f})")
            steps = max(1, int(max(0.5, realistic_slew_time) / 0.5))
            start_az, start_el = self.simulated_az_el
            az_delta = ((az - start_az + 180.0) % 360.0) - 180.0
            el_delta = el - start_el
            for i in range(1, steps + 1):
                if self.shutdown_event and self.shutdown_event.is_set():
                    if progress_cb:
                        try: progress_cb(*self.simulated_az_el, "abort")
                        except Exception: pass
                    logger.info("[DRY RUN] Slew aborted by shutdown.")
                    return False
                f = i / steps
                cur_az = (start_az + f * az_delta) % 360.0
                cur_el = start_el + f * el_delta
                self.simulated_az_el = (cur_az, cur_el)
                if progress_cb:
                    try: progress_cb(cur_az, cur_el, "slewing")
                    except Exception: pass
                time.sleep(0.5)
            self.simulated_az_el = (az % 360.0, el)
            if progress_cb:
                 try: progress_cb(self.simulated_az_el[0], self.simulated_az_el[1], "arrived")
                 except Exception: pass
            logger.info(f"[DRY RUN] Slew complete. Mount is now at: ({az:.2f}, {el:.2f})")
            return True

        if not self.mount:
             logger.error("Error: Mount not connected for slew.")
             if progress_cb:
                 try: progress_cb(0, 0, "abort")
                 except Exception: pass
             return False
        
        az = az % 360.0
        el = max(-5.0, min(95.0, el))
        
        try: # Sun safety check
            current_az, current_el = self._get_current_az_el_internal()
            max_slew = float(CONFIG['hardware'].get('max_slew_deg_s', 6.0))
            now_frame = get_altaz_frame(self.observer_loc)
            est_slew_time = slew_time_needed((current_az, current_el), (az, el), max_slew, now_frame) if max_slew > 0 else 10.0
            arrival_time_unix = time.time() + est_slew_time + 2.0
            sun_az_el_future = get_sun_azel(arrival_time_unix, self.observer_loc)
            arrival_frame = AltAz(obstime=Time(arrival_time_unix, format='unix'), location=self.observer_loc)
            separation_deg_future = angular_sep_deg((az, el), sun_az_el_future, arrival_frame)
            min_sep = float(CONFIG['selection'].get('min_sun_separation_deg', 15.0))
            if separation_deg_future < min_sep:
                logger.warning(f"SAFETY ABORT (Predictive): Target ({az:.1f}, {el:.1f}) will be {separation_deg_future:.2f}° from Sun.")
                if progress_cb:
                    try: progress_cb(current_az, current_el, "abort")
                    except Exception: pass
                return False
        except Exception as e:
            logger.warning(f"Warning: Predictive sun safety check failed: {e}.")
        
        alt_limit_prop = self.mount.getNumber("ALTITUDE_LIMITS")
        if alt_limit_prop: # Altitude limit check
            min_el_w = alt_limit_prop.findWidgetByName("MIN")
            if min_el_w and el < min_el_w.value:
                 logger.warning(f"SAFETY ABORT: El {el:.2f} < limit {min_el_w.value:.2f} deg.")
                 if progress_cb:
                     try: progress_cb(*self._get_current_az_el_internal(), "abort")
                     except Exception: pass
                 return False
            max_el_w = alt_limit_prop.findWidgetByName("MAX")
            if max_el_w and el > max_el_w.value:
                 logger.warning(f"SAFETY ABORT: El {el:.2f} > limit {max_el_w.value:.2f} deg.")
                 if progress_cb:
                     try: progress_cb(*self._get_current_az_el_internal(), "abort")
                     except Exception: pass
                 return False
        
        coord_prop = None; coord_prop_name = ""
        horiz = self.mount.getNumber("HORIZONTAL_COORD")
        if horiz: # Send Horizontal Coords
            coord_prop_name = "HORIZONTAL_COORD"
            coord_prop = horiz
            az_w = coord_prop.findWidgetByName("AZ")
            alt_w = coord_prop.findWidgetByName("ALT")
            if az_w and alt_w:
                az_w.value = az
                alt_w.value = el
                self.sendNewNumber(coord_prop)
            else:
                coord_prop = None
        else: # Send Equatorial Coords
            eq = self.mount.getNumber("EQUATORIAL_EOD_COORD")
            if eq:
                coord_prop_name = "EQUATORIAL_EOD_COORD"
                coord_prop = eq
                try:
                     altaz_frame = AltAz(obstime=Time.now(), location=self.observer_loc)
                     target_altaz = SkyCoord(az=az*u.deg, alt=el*u.deg, frame=altaz_frame)
                     target_icrs = target_altaz.transform_to('icrs')
                     ra_w = coord_prop.findWidgetByName("RA")
                     dec_w = coord_prop.findWidgetByName("DEC")
                     if ra_w and dec_w:
                         ra_w.value = target_icrs.ra.hour
                         dec_w.value = target_icrs.dec.deg
                         self.sendNewNumber(coord_prop)
                     else:
                         coord_prop = None
                except Exception as e:
                     logger.error(f"Error converting AltAz to EQ: {e}")
                     coord_prop = None
            else:
                 logger.error("Error: No HORIZONTAL_COORD or EQUATORIAL_EOD_COORD available.")
                 if progress_cb:
                     try:
                         progress_cb(*self._get_current_az_el_internal(), "abort")
                     except Exception:
                         progress_cb(0, 0, "abort")
                 return False
        
        if not coord_prop: # Check if send command failed
             logger.error("Error: Failed to send slew command.")
             if progress_cb:
                 try:
                     progress_cb(*self._get_current_az_el_internal(), "abort")
                 except Exception:
                     progress_cb(0, 0, "abort")
             return False
        
        logger.info(f"Slewing to az/el: ({az:.2f}, {el:.2f})...")
        try: # Estimate timeout
             cur_az_est, cur_el_est = self._get_current_az_el_internal()
             max_slew_est = float(CONFIG['hardware'].get('max_slew_deg_s', 6.0))
             dist_az_est = abs(((az - cur_az_est + 180.0) % 360.0) - 180.0)
             dist_el_est = abs(el - cur_el_est)
             dist_est = max(dist_az_est, dist_el_est)
             est_time = (dist_est / max_slew_est) if max_slew_est > 0 else 60.0
             slew_timeout_at = time.time() + max(20.0, est_time + 15.0)
        except Exception:
             slew_timeout_at = time.time() + 90.0
        
        slew_finished_state = PyIndi.IPS_BUSY
        while time.time() < slew_timeout_at: # Polling loop
            coord_prop = self.mount.getNumber(coord_prop_name)
            if not coord_prop:
                logger.warning("Warning: Mount property lost.")
                slew_finished_state = PyIndi.IPS_ALERT
                break
            slew_finished_state = coord_prop.s
            if slew_finished_state == PyIndi.IPS_BUSY:
                try:
                    cur_az_prog, cur_el_prog = self._get_current_az_el_internal()
                    if progress_cb:
                        try: progress_cb(cur_az_prog, cur_el_prog, "slewing")
                        except Exception: pass
                except Exception: pass
                if self.shutdown_event and self.shutdown_event.is_set():
                    logger.info("Slew aborted by shutdown.")
                    if progress_cb:
                        try: progress_cb(*self._get_current_az_el_internal(), "abort")
                        except Exception: progress_cb(0, 0, "abort")
                    return False
                time.sleep(0.5)
                continue
            else:
                break # Exit loop if not BUSY
        
        final_prop = self.mount.getNumber(coord_prop_name) # Check final state
        final_state = final_prop.s if final_prop else PyIndi.IPS_ALERT
        
        if final_state == PyIndi.IPS_OK: # SUCCESS Check
            logger.info("Slew completed successfully (IPS_OK).")
            try:
                cur_az_final, cur_el_final = self._get_current_az_el_internal()
                if progress_cb:
                    try: progress_cb(cur_az_final, cur_el_final, "arrived")
                    except Exception: pass
                logger.info(f"Mount final position: ({cur_az_final:.2f}, {cur_el_final:.2f})")
                return True
            except Exception as e:
                logger.error(f"Error getting final pos after OK slew: {e}")
                if progress_cb:
                    try: progress_cb(az, el, "arrived")
                    except Exception: pass
                return True
        elif final_state == PyIndi.IPS_ALERT: # ALERT check
            logger.warning("Warning: Slew failed (Alert).")
            if progress_cb:
                try: progress_cb(*self._get_current_az_el_internal(), "abort")
                except Exception: progress_cb(0, 0, "abort")
            return False
        elif final_state == PyIndi.IPS_BUSY: # TIMEOUT check
            logger.warning("Warning: Slew timed out.")
            if progress_cb:
                try: progress_cb(*self._get_current_az_el_internal(), "abort")
                except Exception: progress_cb(0, 0, "abort")
            return False
        else: # UNEXPECTED state check
            state_str = {PyIndi.IPS_IDLE:"IDLE", PyIndi.IPS_OK:"OK", PyIndi.IPS_BUSY:"BUSY", PyIndi.IPS_ALERT:"ALERT"}.get(final_state, str(final_state))
            logger.warning(f"Warning: Slew unexpected state ({state_str}). Failure.")
            if progress_cb:
                try: progress_cb(*self._get_current_az_el_internal(), "abort")
                except Exception: progress_cb(0, 0, "abort")
            return False

    def autofocus(self, alt_ft: float = None) -> bool:
        if self.shutdown_event and self.shutdown_event.is_set(): logger.info("Autofocus aborted: shutdown."); return False
        if CONFIG['development']['dry_run']: logger.info(f"[DRY RUN] Performing autofocus."); time.sleep(3.0); return True
        if not self.focuser or not self.camera: logger.error("Error: Focuser or Camera not connected."); return False
        
        logger.info("Starting autofocus...")
        af_cfg = CONFIG['capture'].get('autofocus', {})
        max_duration_s = float(af_cfg.get('max_duration_s', 180.0))
        routine_timeout_at = time.time() + max_duration_s
        sharp_thresh = float(af_cfg.get('sharpness_threshold', 20.0))
        
        abs_pos_prop = self._wait_number(self.focuser, "ABS_FOCUS_POSITION")
        if not abs_pos_prop: logger.error("Focus failed: No absolute position."); return False
        pos_widget = abs_pos_prop.findWidgetByName("FOCUS_ABSOLUTE_POSITION")
        if not pos_widget: logger.error("Focus failed: Cannot find position widget."); return False
        
        current_pos = pos_widget.value
        focuser_min, focuser_max = pos_widget.min, pos_widget.max
        logger.info(f"  Focuser limits: [{focuser_min:.0f} - {focuser_max:.0f}], Current: {current_pos:.0f}")
        
        step_base = int(af_cfg.get('step_base', 50))
        scan_range = int(af_cfg.get('scan_range', 5))
        step_size = step_base
        raw_positions = [current_pos + i * step_size for i in range(-scan_range, scan_range + 1)]
        positions = sorted(list(set(int(round(p)) for p in raw_positions if focuser_min <= p <= focuser_max)))
        
        if not positions or len(positions) < 3: logger.error(f"Focus failed: Insufficient positions ({len(positions)})."); return False
        
        logger.info(f"  Scanning positions: {positions}")
        sharpness_scores = {}
        focus_exposure = float(af_cfg.get('exposure_s', 0.1))
        
        for pos in positions: # Scan loop
            if time.time() > routine_timeout_at: logger.error("Autofocus aborted: timeout."); return False
            if self.shutdown_event and self.shutdown_event.is_set(): logger.info("Autofocus aborted: shutdown."); return False
            
            pos_widget.value = pos
            self.sendNewNumber(abs_pos_prop)
            
            move_timeout_at = time.time() + 20
            move_finished_state = PyIndi.IPS_BUSY
            while time.time() < move_timeout_at: # Wait for move
                pos_prop = self.focuser.getNumber("ABS_FOCUS_POSITION")
                if not pos_prop: logger.warning(f"  Warning: Focuser property lost moving to {pos}."); move_finished_state = PyIndi.IPS_ALERT; break
                move_finished_state = pos_prop.s
                if move_finished_state == PyIndi.IPS_BUSY:
                    if self.shutdown_event and self.shutdown_event.is_set(): return False
                    time.sleep(0.2); continue
                else: break
            
            final_prop = self.focuser.getNumber("ABS_FOCUS_POSITION")
            final_state = final_prop.s if final_prop else PyIndi.IPS_ALERT
            if final_state != PyIndi.IPS_OK: # Check completion
                 state_str = {PyIndi.IPS_IDLE: "IDLE", PyIndi.IPS_OK: "OK", PyIndi.IPS_BUSY: "BUSY", PyIndi.IPS_ALERT: "ALERT"}.get(final_state, str(final_state))
                 logger.error(f"  Focus move to {pos} failed (State: {state_str}). Aborting scan."); return False
            
            snap_path = None # Capture image
            try:
                snap_path = self.capture_image(focus_exposure, os.path.join(LOG_DIR, 'images', f"autofocus_{pos}_{time.time_ns()}.fits"))
                if not snap_path: raise IOError("Capture failed")
                sharpness = measure_sharpness(snap_path)
                sharpness_scores[pos] = sharpness
                logger.info(f"  Position {pos:.0f}: Sharpness {sharpness:.2f}")
            except Exception as e: logger.error(f"  Error capture/analysis at {pos}: {e}"); sharpness_scores[pos] = 0.0
        
        if not sharpness_scores: logger.error("Focus failed: No scores recorded."); return False
        
        best_pos = max(sharpness_scores, key=sharpness_scores.get)
        best_sharp = sharpness_scores[best_pos]
        
        if best_sharp < sharp_thresh:
            logger.error(f"Focus failed: Best sharpness ({best_sharp:.2f}) < threshold ({sharp_thresh:.1f}).")
            pos_widget.value = current_pos; self.sendNewNumber(abs_pos_prop); return False
        
        logger.info(f"  Best focus at {best_pos:.0f} (Sharpness: {best_sharp:.2f}). Moving...");
        pos_widget.value = best_pos
        self.sendNewNumber(abs_pos_prop)
        
        move_timeout_at = time.time() + 20
        move_finished_state = PyIndi.IPS_BUSY
        while time.time() < move_timeout_at: # Wait for final move
             pos_prop = self.focuser.getNumber("ABS_FOCUS_POSITION")
             if not pos_prop: logger.warning(f"  Warning: Focuser property lost on final move."); move_finished_state = PyIndi.IPS_ALERT; break
             move_finished_state = pos_prop.s
             if move_finished_state == PyIndi.IPS_BUSY:
                  if self.shutdown_event and self.shutdown_event.is_set(): return False
                  time.sleep(0.2); continue
             else: break
        
        final_prop = self.focuser.getNumber("ABS_FOCUS_POSITION")
        final_state = final_prop.s if final_prop else PyIndi.IPS_ALERT
        if final_state != PyIndi.IPS_OK:
            state_str = {PyIndi.IPS_IDLE: "IDLE", PyIndi.IPS_OK: "OK", PyIndi.IPS_BUSY: "BUSY", PyIndi.IPS_ALERT: "ALERT"}.get(final_state, str(final_state))
            logger.warning(f"Warning: Move to best focus {best_pos} failed (State: {state_str}).")
        
        logger.info(f"Autofocus complete. Final position: {best_pos:.0f}")
        log_entry = {'alt_ft': alt_ft, 'best_pos': best_pos, 'sharpness': best_sharp, 'timestamp': time.time(), 'scan_data': sharpness_scores}
        append_to_json([log_entry], os.path.join(LOG_DIR, 'focus_logs.json')); return True

    def guide_pulse(self, direction: str, duration_ms: int):
        duration_ms = max(0, int(duration_ms))
        if direction not in ('N','S','E','W'): logger.warning(f"Warning: Invalid guide direction '{direction}'"); return
        if CONFIG['development']['dry_run']: time.sleep(duration_ms / 1000.0); return
        if not self.mount: logger.error("Error: Mount not connected."); return
        
        timed_prop_name, timed_widget_name = "", ""
        if direction in ['N', 'S']: timed_prop_name = "TELESCOPE_TIMED_GUIDE_NS"; timed_widget_name = "TIMED_GUIDE_N" if direction == 'N' else "TIMED_GUIDE_S"
        elif direction in ['W', 'E']: timed_prop_name = "TELESCOPE_TIMED_GUIDE_WE"; timed_widget_name = "TIMED_GUIDE_W" if direction == 'W' else "TIMED_GUIDE_E"
        
        timed_prop = self.mount.getNumber(timed_prop_name) if timed_prop_name else None
        if timed_prop: # Use timed command if available
            widget = timed_prop.findWidgetByName(timed_widget_name)
            if widget: widget.value = duration_ms; self.sendNewNumber(timed_prop); return
        
        prop_name, widget_name = "", "" # Fallback to switch pulse
        if direction in ['N', 'S']: prop_name = "TELESCOPE_MOTION_NS"; widget_name = "MOTION_NORTH" if direction == 'N' else "MOTION_SOUTH"
        elif direction in ['W', 'E']: prop_name = "TELESCOPE_MOTION_WE"; widget_name = "MOTION_WEST" if direction == 'W' else "MOTION_EAST"
        else: return
        
        prop = self.mount.getSwitch(prop_name)
        if not prop: return
        widget = prop.findWidgetByName(widget_name)
        if not widget: return
        self._set_switch_state(prop, widget_name, PyIndi.ISS_ON); time.sleep(duration_ms / 1000.0); self._set_switch_state(prop, widget_name, PyIndi.ISS_OFF)

    def park_mount(self) -> bool:
        if CONFIG['development']['dry_run']: logger.info("[DRY RUN] Parking mount."); self.simulated_az_el = (180.0, 0.0); return True
        if not self.mount: logger.error("Error: Mount not connected."); return False
        
        park_prop = self.mount.getSwitch("TELESCOPE_PARK")
        if not park_prop: logger.warning("Warning: Mount does not support PARK."); return False
        park_widget = park_prop.findWidgetByName("PARK")
        if not park_widget and len(park_prop) > 0: park_widget = park_prop[0]
        if not park_widget: logger.error("Error: Cannot find PARK widget."); return False
        
        self._set_switch_state(park_prop, park_widget.name, PyIndi.ISS_ON)
        logger.info("Parking mount..."); park_timeout_at = time.time() + 120
        
        park_finished_state = PyIndi.IPS_BUSY # Wait for completion
        while time.time() < park_timeout_at:
            park_state_prop = self.mount.getSwitch("TELESCOPE_PARK")
            if not park_state_prop: logger.warning("Warning: Park property lost."); park_finished_state = PyIndi.IPS_ALERT; break
            park_finished_state = park_state_prop.s
            if park_finished_state == PyIndi.IPS_BUSY:
                 if self.shutdown_event and self.shutdown_event.is_set(): logger.info("Park aborted by shutdown."); return False
                 time.sleep(1); continue
            else: break
        
        final_prop = self.mount.getSwitch("TELESCOPE_PARK"); final_state = final_prop.s if final_prop else PyIndi.IPS_ALERT
        
        if final_state == PyIndi.IPS_OK: # Check final state
             final_widget = final_prop.findWidgetByName(park_widget.name) if final_prop and park_widget else None
             if final_widget and final_widget.s == PyIndi.ISS_ON: logger.info("Mount parked successfully."); return True
             else: logger.warning(f"Warning: Park OK, but widget state unexpected."); return True
        elif final_state == PyIndi.IPS_ALERT: logger.warning("Warning: Park failed (Alert)."); return False
        elif final_state == PyIndi.IPS_BUSY: logger.warning("Warning: Park timed out."); return False
        else: state_str = {PyIndi.IPS_IDLE:"IDLE", PyIndi.IPS_OK:"OK", PyIndi.IPS_BUSY:"BUSY", PyIndi.IPS_ALERT:"ALERT"}.get(final_state, str(final_state)); logger.warning(f"Warning: Park finished unexpected ({state_str}). Failure."); return False

    def snap_image(self, label: str) -> Optional[str]:
        images_dir = os.path.join(LOG_DIR, 'images')
        path = os.path.join(images_dir, f"snap_{label}_{time.time_ns()}.fits")
        if label.startswith("autofocus"): exp = float(CONFIG['capture'].get('autofocus', {}).get('exposure_s', 0.1))
        else: exp = float(CONFIG['capture'].get('guiding', {}).get('guide_exposure_s', 0.1))
        return self.capture_image(exp, path)

    # --- capture_image METHOD (Includes Dry Run uint16 fix + Text Overlay) ---
    def capture_image(self, exposure_s: float, save_path: str) -> Optional[str]:
        """Captures a single image using the BLOB queue mechanism."""
        try: os.makedirs(os.path.dirname(save_path), exist_ok=True)
        except OSError as e: logger.error(f"Error creating directory for {save_path}: {e}"); return None

        if CONFIG['development']['dry_run']:
            logger.info(f"[DRY RUN] Simulating capture of: {os.path.basename(save_path)}")
            png_path = os.path.splitext(save_path)[0] + ".png"; h = CONFIG.get('camera_specs', {}).get('resolution_height_px', 480); w = CONFIG.get('camera_specs', {}).get('resolution_width_px', 640)

            # 1. Create uint16 base image
            fits_baseline_value = 1000 # Example: low background value
            img_fits_uint16 = np.full((h, w), fits_baseline_value, np.uint16)

            # 2. Add text overlay directly onto the uint16 image
            text_color = 60000; font_scale = 0.8; font_thickness = 2
            cv2.putText(img_fits_uint16, "DRY RUN", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness)
            
            try: # Add dynamic text (e.g., frame number)
                 basename_no_ext = os.path.splitext(os.path.basename(save_path))[0]
                 parts = basename_no_ext.split('_')
                 # Expecting filename like '..._1_of_5'
                 if len(parts) >= 3:
                     dynamic_text = f"Frame: {parts[-3]} of {parts[-1]}"
                 else:
                     raise IndexError("Filename format not as expected")
            except Exception: # Fallback
                 dynamic_text = os.path.basename(save_path)[-20:-5] 
            cv2.putText(img_fits_uint16, dynamic_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)

            # 3. Save the uint16 array (flat background + text) as FITS
            hdu = fits.PrimaryHDU(img_fits_uint16)
            hdu.header['EXPTIME'] = (float(exposure_s), 'Exposure time (s)'); hdu.header['DATE-OBS'] = time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime())
            try: fits.HDUList([hdu]).writeto(save_path, overwrite=True, checksum=True)
            except Exception as e: logger.error(f"[DRY RUN] Failed to write dummy FITS {save_path}: {e}"); return None

            # 4. Generate PNG preview from the uint16 array (with text)
            try:
                img_preview_8bit = cv2.normalize(img_fits_uint16, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                cv2.imwrite(png_path, img_preview_8bit)
            except Exception as e: logger.error(f"[DRY RUN] Failed to write dummy PNG {png_path}: {e}")

            return save_path

        # --- Real Capture Logic ---
        min_exp = float(CONFIG['capture'].get('exposure_min_s', 0.001)); max_exp = float(CONFIG['capture'].get('exposure_max_s', 10.0))
        clamped_exp = max(min_exp, min(max_exp, exposure_s));
        if abs(clamped_exp - exposure_s) > 1e-6: logger.warning(f"  - Warning: Exposure {exposure_s:.4f}s clamped to {clamped_exp:.4f}s.")
        exposure_s = clamped_exp
        exp_prop = self._wait_number(self.camera, "CCD_EXPOSURE");
        if not exp_prop: logger.error("ERROR: CCD_EXPOSURE property not found."); return None
        exp_widget = exp_prop.findWidgetByName("CCD_EXPOSURE_VALUE");
        if not exp_widget: logger.error("ERROR: CCD_EXPOSURE_VALUE widget not found."); return None
        exposure_token = time.time_ns(); received_data = None
        with self._blob_lock: self._exposure_queue.put(exposure_token); self._received_blobs.pop(exposure_token, None)
        try: # Start exposure and wait for BLOB
             exp_widget.value = exposure_s; self.sendNewNumber(exp_prop)
             cmd_timeout = time.time() + max(5.0, exposure_s + 2.0); prop_state = PyIndi.IPS_BUSY
             while prop_state == PyIndi.IPS_BUSY and time.time() < cmd_timeout: # Wait for command accept
                 if self.shutdown_event and self.shutdown_event.is_set(): raise IOError("Capture aborted (shutdown)")
                 time.sleep(0.05); prop = self.camera.getNumber("CCD_EXPOSURE"); prop_state = prop.s if prop else PyIndi.IPS_ALERT
             if prop_state == PyIndi.IPS_ALERT: raise IOError("Exposure command failed (Alert)")
             if prop_state == PyIndi.IPS_BUSY: raise IOError("Exposure command timed out")
             blob_timeout_time = time.time() + max(15.0, exposure_s * 2.0 + 10.0)
             while time.time() < blob_timeout_time: # Wait for BLOB arrival
                 if self.shutdown_event and self.shutdown_event.is_set(): raise IOError("Capture aborted (shutdown waiting for BLOB)")
                 with self._blob_lock:
                     if exposure_token in self._received_blobs: received_data = self._received_blobs.pop(exposure_token); break
                 time.sleep(0.1)
             else: raise IOError(f"Timeout waiting for BLOB ({exposure_s:.3f}s exposure)")
        except Exception as e:
             logger.error(f"ERROR during capture for token {exposure_token}: {e}")
             with self._blob_lock: self._received_blobs.pop(exposure_token, None)
             return None
        if received_data: # Save data
            try:
                with open(save_path, 'wb') as f: f.write(received_data)
                try: # Annotate FITS
                    with fits.open(save_path, mode="update", output_verify='silentfix') as hdul:
                        if hdul and hdul[0].header is not None:
                            hdr = hdul[0].header; hdr['EXPTIME'] = (float(exposure_s), 's'); hdr['DATE-OBS'] = Time.now().utc.isot
                            if self.camera:
                                b=self.camera.getNumber("CCD_BINNING"); g=self.camera.getNumber("CCD_GAIN"); off=self.camera.getNumber("CCD_OFFSET"); t=self.camera.getNumber("CCD_TEMPERATURE")
                                if b: xw,yw=b.findWidgetByName("HOR_BIN"),b.findWidgetByName("VER_BIN"); hdr['XBINNING'],hdr['YBINNING']=int(xw.value),int(yw.value)
                                if g: gw=g.findWidgetByName("GAIN"); hdr['GAIN']=float(gw.value)
                                if off: ow=off.findWidgetByName("OFFSET"); hdr['OFFSET']=float(ow.value)
                                if t: tw=t.findWidgetByName("CCD_TEMPERATURE_VALUE"); hdr['CCD-TEMP']=float(tw.value)
                            hdul.flush(output_verify='silentfix')
                except Exception as e: logger.warning(f"Warning: FITS annotation failed: {e}")
                try: save_png_preview(save_path) # Generate PNG
                except Exception as e: logger.warning(f"Warning: PNG preview failed: {e}")
                return save_path
            except Exception as e: logger.error(f"Error saving BLOB data: {e}"); return None
        else: logger.error(f"Error: No BLOB data received for token {exposure_token}."); return None

    def capture_sequence(self, icao: str, exposure_s: float, num_images: Optional[int] = None) -> List[str]:
        if num_images is None: num_images = int(CONFIG['capture'].get('num_sequence_images', 5))
        if num_images <= 0: return []
        logger.info(f"  Starting capture sequence of {num_images} images with exposure: {exposure_s:.4f}s"); paths = []; images_dir = os.path.join(LOG_DIR, 'images')
        for i in range(num_images):
            if self.shutdown_event and self.shutdown_event.is_set(): logger.info("  Capture sequence interrupted."); break
            path = os.path.join(images_dir, f"capture_{icao}_{time.time_ns()}_{i+1}_of_{num_images}.fits")
            try:
                captured_path = self.capture_image(exposure_s, path) # Calls patched capture_image
                if captured_path: paths.append(captured_path)
                else: logger.error(f"  Frame {i+1}/{num_images} failed.")
            except Exception as e: logger.error(f"  Unexpected error capturing frame {i+1}: {e}"); break
        if paths:
            try:
                cam_settings = {"gain": CONFIG['camera_specs'].get('gain'), "offset": CONFIG['camera_specs'].get('offset'), "binning": CONFIG['camera_specs'].get('binning', {"x":1,"y":1})}
                sequence_id = f"{icao}_{int(time.time())}"; schedule_stack_and_publish(sequence_id, paths, cam_settings)
                logger.info(f"  Scheduled stacking for burst {sequence_id} ({len(paths)} frames)")
            except Exception as e: logger.warning(f"Warning: could not schedule stacking: {e}")
        return paths

    def disconnect(self):
        if CONFIG['development']['dry_run']: logger.info("[DRY RUN] Simulating hardware disconnect."); return
        if self.isConnected(): logger.info("Disconnecting from INDI server..."); self.disconnectServer()
