#main.py
"""
Main entry point: Monitors ADS-B data, processes flight paths, and tracks aircraft.
"""

import time
import threading
import os
import math
import json
import numpy as np # <-- Import NumPy
from typing import Optional, Callable # <-- Import Optional and Callable
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from astropy.coordinates import EarthLocation
import astropy.units as u

from config_loader import CONFIG, LOG_DIR
from data_reader import read_aircraft_data
from dead_reckoning import estimate_positions_at_times
from storage import ensure_log_dir, append_to_json
from status_writer import write_status
from aircraft_selector import (
    select_aircraft,
    calculate_quality,
    calculate_expected_value,  # used for single-aircraft EV reason
    evaluate_manual_target_viability,  # detailed manual viability reasons
)
from hardware_control import IndiController
# --- Import internal functions and loader from image_analyzer ---
from image_analyzer import (
    _load_fits_data,
    _detect_aircraft_from_data,
    _estimate_exposure_adjustment_from_data,
    _save_png_preview_from_data
)
# --- Keep original functions for dry run simulation if needed ---
from image_analyzer import detect_aircraft
# --- End Import Changes ---
from coord_utils import (calculate_plate_scale, latlonalt_to_azel, haversine_distance,
                         get_sun_azel, angular_sep_deg, angular_speed_deg_s, get_altaz_frame)

def _to_img_url(fs_path: str) -> str:
    """Converts a filesystem image path to its corresponding dashboard URL."""
    # Ensure fs_path is a string before calling os.path.basename
    return "/images/" + os.path.basename(str(fs_path)) if fs_path else ""


class CommandHandler(FileSystemEventHandler):
    """A simple handler to watch for manual override commands."""
    def __init__(self, main_handler):
        self.main_handler = main_handler
        command_filename = CONFIG['logging']['command_file']
        self.command_file = os.path.normpath(os.path.join(LOG_DIR, command_filename))

    def on_modified(self, event):
        if os.path.normpath(event.src_path) == self.command_file:
            self._process_command()

    def on_created(self, event):
        if os.path.normpath(event.src_path) == self.command_file:
            self._process_command()

    def on_moved(self, event):
        # Watchdog triggers moved event even for atomic replace if dest exists
        if os.path.normpath(event.dest_path) == self.command_file:
            self._process_command()

    def _process_command(self):
        """Reads and processes the command file, handling race conditions."""
        print("--- Manual command detected! ---")
        try:
            # Read command immediately after detection
            with open(self.command_file, 'r') as f:
                command = json.load(f)

            # Attempt to delete immediately to prevent reprocessing
            try:
                os.remove(self.command_file)
                print(f"  Processed and removed command file.")
            except OSError as e:
                # Handle case where file might already be gone (another handler?)
                print(f"  Note: Could not remove command file (possibly already gone): {e}")
                pass # Continue processing the command we read

            if 'track_icao' in command:
                icao = command['track_icao'].lower().strip()
                print(f"  Manual override to track ICAO: {icao}")

                with self.main_handler.track_lock:
                    # Cancel any pending scheduled task
                    if self.main_handler._scheduler_timer and self.main_handler._scheduler_timer.is_alive():
                        self.main_handler._scheduler_timer.cancel()
                        self.main_handler._scheduler_timer = None
                        print("  Cancelled pending scheduler timer.")
                    # If currently tracking, request preemption
                    if self.main_handler.current_track_icao and self.main_handler.active_thread and self.main_handler.active_thread.is_alive():
                        print("  Interrupting current track for manual override...")
                        self.main_handler.preempt_requested = True
                        self.main_handler.shutdown_event.set() # Signal current track to stop
                    # Set or replace the manual request
                    self.main_handler.manual_target_icao = icao
                    # Clear previous manual viability info to recompute fresh
                    self.main_handler.manual_viability_info = None
                # Immediately try to run scheduler with the new manual target
                # (needs to happen outside the lock to avoid deadlock if scheduler needs lock)
                print("  Triggering scheduler for manual target...")
                self.main_handler._run_scheduler()

            elif command.get('command') == 'abort_track':
                print("  Abort command received.")
                with self.main_handler.track_lock:
                    if self.main_handler._scheduler_timer and self.main_handler._scheduler_timer.is_alive():
                        self.main_handler._scheduler_timer.cancel()
                        self.main_handler._scheduler_timer = None
                        print("  Cancelled pending scheduler timer.")
                    # Clear any manual target requests
                    self.main_handler.preempt_requested = False
                    self.main_handler.manual_target_icao = None
                    self.main_handler.manual_viability_info = None
                    # Signal current track (if any) to stop
                    if self.main_handler.current_track_icao:
                        print(f"  Signalling current track ({self.main_handler.current_track_icao}) to abort.")
                        self.main_handler.shutdown_event.set()
                    else:
                        print("  No active track to abort.")


            elif command.get('command') == 'park_mount':
                print("  Park command received.")
                active_thread_to_join = None
                with self.main_handler.track_lock:
                    # Cancel scheduler, clear manual target
                    if self.main_handler._scheduler_timer and self.main_handler._scheduler_timer.is_alive():
                        self.main_handler._scheduler_timer.cancel()
                        self.main_handler._scheduler_timer = None
                    self.main_handler.preempt_requested = False
                    self.main_handler.manual_target_icao = None
                    self.main_handler.manual_viability_info = None
                    # Signal current track to stop and get reference for joining
                    if self.main_handler.current_track_icao:
                        print(f"  Signalling current track ({self.main_handler.current_track_icao}) to stop before parking.")
                        self.main_handler.shutdown_event.set()
                        active_thread_to_join = self.main_handler.active_thread

                # Wait for tracking thread to finish (outside lock)
                if active_thread_to_join and active_thread_to_join.is_alive():
                    print("  Waiting for tracking thread to finish...")
                    active_thread_to_join.join(timeout=10.0) # Increased timeout
                    if active_thread_to_join.is_alive():
                         print("  Warning: Tracking thread did not finish cleanly before parking.")

                # Initiate parking in a separate thread (outside lock)
                if self.main_handler.indi_controller:
                    print("  Initiating park command...")
                    # Run park in its own thread so it doesn't block command handler
                    threading.Thread(target=self.main_handler.indi_controller.park_mount, daemon=True).start()
                else:
                     print("  Cannot park: Hardware controller not initialized.")


        except FileNotFoundError:
            # This can happen in a race condition if file is deleted between event and open
            print("  Command file not found (likely already processed or deleted). Ignoring.")
        except json.JSONDecodeError as e:
            print(f"  Could not read command file (invalid JSON): {e}")
        except Exception as e:
             # Catch unexpected errors during command processing
             print(f"  Unexpected error processing command: {e}")


class FileHandler(FileSystemEventHandler):
    """Handles file events and orchestrates the tracking process."""
    def __init__(self, watch_file):
        self.watch_file = os.path.normpath(watch_file)
        self.last_process_time = 0
        self.debounce_seconds = 0.5 # Reduce debounce slightly?
        self.indi_controller = None
        self.current_track_icao = None
        self.current_track_ev = 0.0
        self.track_lock = threading.Lock() # Protects access to tracking state vars
        obs_cfg = CONFIG['observer']
        # Ensure lat/lon/alt are floats
        try:
            lat = float(obs_cfg['latitude_deg'])
            lon = float(obs_cfg['longitude_deg'])
            alt = float(obs_cfg['altitude_m'])
            self.observer_loc = EarthLocation(lat=lat*u.deg, lon=lon*u.deg, height=alt*u.m)
        except (ValueError, KeyError) as e:
             print(f"FATAL: Invalid observer location in config: {e}")
             # Or raise a more specific error
             raise ValueError(f"Invalid observer location config: {e}") from e

        self.shutdown_event = threading.Event() # Signals tracking thread to stop
        self.active_thread = None # Reference to the current tracking thread
        self.last_file_stat = None # To detect redundant file events
        self.manual_target_icao = None # ICAO requested manually
        self.manual_viability_info = None  # Cache last manual viability details for status
        self._scheduler_timer = None # Timer for delayed scheduling
        self.preempt_requested = False # Flag for preemptive track switching

    def process_file(self):
        """Processes aircraft.json, runs the scheduler, and manages state."""
        # --- File Stat Check & Debounce ---
        try:
            # Check if file actually changed content/time (watchdog can be noisy)
            current_stat = os.stat(self.watch_file)
            if self.last_file_stat and (current_stat.st_mtime == self.last_file_stat.st_mtime and
                                        current_stat.st_size == self.last_file_stat.st_size):
                return # No actual change, ignore event
            self.last_file_stat = current_stat
        except FileNotFoundError:
             # If file disappears briefly, just return and wait for next event
            return
        except Exception as e:
             print(f"Warning: Error stating file {self.watch_file}: {e}")
             return


        current_time = time.time()
        if current_time - self.last_process_time < self.debounce_seconds:
            return # Debounce rapid events
        self.last_process_time = current_time

        # --- Read Aircraft Data ---
        aircraft_dict = read_aircraft_data() # Returns {} on error or stale file

        # --- Initialize Hardware (if not already) ---
        if not self.indi_controller:
            try:
                print("Initializing hardware controller...")
                self.indi_controller = IndiController()
                # Pass shutdown event to controller so it can abort actions
                self.indi_controller.shutdown_event = self.shutdown_event
                print("Hardware controller initialized.")
            except Exception as e:
                # Log error and prevent further processing if hardware fails
                print(f"FATAL: Hardware initialization failed: {e}")
                # Optionally, set a status indicating hardware error
                write_status({"mode": "error", "error_message": f"Hardware init failed: {e}"})
                self.indi_controller = None # Ensure it's None so we retry later
                return # Stop processing this cycle


        # --- Get Current Hardware Status ---
        hardware_status = self.indi_controller.get_hardware_status() or {}

        # --- Handle Empty Airspace ---
        if not aircraft_dict:
            # Update status even if no aircraft, report current mode
            status_payload = hardware_status.copy() # Start with hardware status
            status_payload["mode"] = "tracking" if self.current_track_icao else "idle"
            # Include manual target info if applicable
            if self.manual_target_icao:
                status_payload["manual_target"] = {
                    "icao": self.manual_target_icao,
                    # Show last known viability, or default if never checked
                    "viability": self.manual_viability_info or {"viable": False, "reasons": ["no ADS-B data available"]},
                }
            # Write status and return
            write_status(status_payload)
            return

        # --- Select Candidates & Check Preemption ---
        current_az_el = hardware_status.get("mount_az_el", (0.0, 0.0))
        # Ensure select_aircraft handles potential errors gracefully
        try:
             candidates = select_aircraft(aircraft_dict, current_az_el) or []
        except Exception as e:
             print(f"Error during aircraft selection: {e}")
             candidates = [] # Treat as no candidates on error


        # Check for preemptive switch if currently tracking
        if self.current_track_icao and candidates:
            new_best_target = candidates[0]
            preempt_factor = float(CONFIG['selection'].get('preempt_factor', 1.25)) # Ensure float
            # Check if new target is significantly better AND not the current one
            if new_best_target['icao'] != self.current_track_icao and \
               new_best_target['ev'] > (self.current_track_ev * preempt_factor):

                print(f"--- PREEMPTIVE SWITCH: New target {new_best_target['icao']} (EV {new_best_target['ev']:.1f}) "
                      f"> Current {self.current_track_icao} (EV {self.current_track_ev:.1f} * {preempt_factor:.2f}). ---")
                with self.track_lock:
                    # Cancel scheduler if waiting
                    if self._scheduler_timer and self._scheduler_timer.is_alive():
                        self._scheduler_timer.cancel()
                        self._scheduler_timer = None
                    # Signal current track to stop for preemption
                    self.preempt_requested = True
                    self.shutdown_event.set()
                    # Note: Scheduler will be called again by track_aircraft finally block

        # --- Build Status Payload ---
        all_aircraft_list = []
        obs_lat = float(CONFIG['observer']['latitude_deg'])
        obs_lon = float(CONFIG['observer']['longitude_deg'])
        for icao, data in aircraft_dict.items():
            lat, lon = data.get('lat'), data.get('lon')
            # Skip aircraft without position for distance calc
            if lat is None or lon is None: continue

            # Calculate distance (haversine_distance now returns km)
            dist_km = haversine_distance(obs_lat, obs_lon, lat, lon)
            dist_nm = dist_km / 1.852 if dist_km is not None else None

            # Handle stale position display (match SkyAware logic)
            # Use 'age_s' calculated by data_reader
            age_s = data.get('age_s')
            # Stale threshold (e.g., 15 seconds) - make configurable?
            is_stale = (age_s is not None) and (age_s > 15.0)
            dist_nm_display = None if is_stale else dist_nm

            flight_data = data.get('flight', '').strip() # Default to empty string

            all_aircraft_list.append({
                "icao": icao,
                "flight": flight_data or 'N/A', # Display N/A if empty
                "alt": data.get('alt'), # Already validated by reader
                "gs": data.get('gs'),   # Already validated
                "track": data.get('track'), # Already validated
                "dist_nm": dist_nm_display # Use display value (can be None)
            })
        # Sort list by distance for display (closest first)
        all_aircraft_list.sort(key=lambda x: x['dist_nm'] if x['dist_nm'] is not None else float('inf'))


        status_payload = {
            # Top 5 candidates for queue display (excluding the best one if tracking)
            "queue": candidates[1:6] if self.current_track_icao else candidates[:5],
            # Best candidate (potential next target)
            "track_candidates": candidates[:1],
            "all_aircraft": all_aircraft_list,
            "observer_loc": {"lat": obs_lat, "lon": obs_lon},
            # Include camera specs for reference? Maybe subset?
            "camera_settings": {
                 'binning': CONFIG.get('camera_specs', {}).get('binning'),
                 'gain': CONFIG.get('camera_specs', {}).get('gain'),
                 'offset': CONFIG.get('camera_specs', {}).get('offset'),
                 'cooling': CONFIG.get('camera_specs', {}).get('cooling')
             }
        }
        status_payload.update(hardware_status) # Merge hardware status
        status_payload["mode"] = "tracking" if self.current_track_icao else "idle"

        # Include manual target info if set
        if self.manual_target_icao:
            status_payload["manual_target"] = {
                "icao": self.manual_target_icao,
                "viability": self.manual_viability_info # Last known viability
            }

        # --- Write Status ---
        write_status(status_payload)

        # --- Run Scheduler (if not already tracking) ---
        # Scheduler logic is now inside _run_scheduler, called conditionally
        if not self.current_track_icao:
             self._run_scheduler(candidates) # Pass current candidates


    def on_modified(self, event):
        # Trigger processing only if the specific file we watch is modified
        if os.path.normpath(event.src_path) == self.watch_file:
            self.process_file()

    def on_moved(self, event):
        # Trigger processing if the file is moved *to* the path we watch
        # (Handles atomic replaces where file is written then renamed)
        if os.path.normpath(event.dest_path) == self.watch_file:
            self.process_file()

    # --- Helpers for live target coords during slews ---
    def _predict_target_az_el(self, aircraft_data: dict, when: Optional[float] = None) -> Optional[tuple[float, float]]:
        """
        Predict target AZ/EL at 'when' using dead-reckoning.
        Returns (az, el) tuple or None if prediction is unavailable/fails.
        """
        t = when or time.time() # Use current time if 'when' not specified
        try:
            pos_list = estimate_positions_at_times(aircraft_data, [t])
            if not pos_list:
                return None # No prediction available
            pos = pos_list[0]
            # Ensure necessary keys exist before conversion
            if not all(k in pos for k in ['est_lat', 'est_lon', 'est_alt']):
                 print("Warning: Prediction missing required keys (lat/lon/alt).")
                 return None

            az, el = latlonalt_to_azel(pos['est_lat'], pos['est_lon'], pos['est_alt'], t, self.observer_loc)
            # Ensure results are finite floats
            if not (math.isfinite(az) and math.isfinite(el)):
                 print(f"Warning: Non-finite az/el prediction ({az}, {el}).")
                 return None
            return (float(az), float(el))
        except Exception as e:
            # Catch errors during prediction or coordinate conversion
            print(f"Error predicting target az/el: {e}")
            return None


    def _run_scheduler(self, candidates: Optional[list] = None):
        """Decides whether to start a new track or wait."""
        # This function should only run if not currently tracking
        with self.track_lock:
            if self.current_track_icao:
                # print("  Scheduler: Already tracking, returning.") # Debug log
                return # Exit if already tracking

            # --- Refresh Data if needed ---
            # If candidates weren't passed, get fresh data and selection
            if candidates is None:
                # print("  Scheduler: Fetching fresh data and candidates...") # Debug log
                aircraft_dict = read_aircraft_data()
                # Need hardware controller for current mount position
                if not self.indi_controller:
                    print("  Scheduler: Cannot run, hardware controller not ready.")
                    return
                hs = self.indi_controller.get_hardware_status() or {}
                current_az_el = hs.get("mount_az_el", (0.0, 0.0))
                try:
                    candidates = select_aircraft(aircraft_dict, current_az_el) or []
                except Exception as e:
                     print(f"  Scheduler: Error selecting aircraft: {e}")
                     candidates = []
            else:
                 # Use candidates passed from process_file
                 aircraft_dict = read_aircraft_data() # Still need latest full dict for manual check

            # --- Determine Target ---
            target_to_consider = None
            is_manual_override = False

            # Prioritize Manual Target
            if self.manual_target_icao:
                icao = self.manual_target_icao
                print(f"  Scheduler: Evaluating manual target {icao}...")
                # Check if manual target is in the current viable candidates list
                manual_target_in_candidates = next((c for c in candidates if c['icao'] == icao), None)

                if manual_target_in_candidates:
                    # Great — it’s viable now; use it and clear manual request
                    print(f"  Manual target {icao} is now viable (EV: {manual_target_in_candidates['ev']:.1f}). Selecting.")
                    target_to_consider = manual_target_in_candidates
                    is_manual_override = True
                    self.manual_target_icao = None # Clear request
                    self.manual_viability_info = None
                else:
                    # Manual target not viable *right now*. Check why.
                    print(f"  Manual target {icao} not in top candidates. Checking viability...")
                    ok, reasons, details = evaluate_manual_target_viability(
                        icao, aircraft_dict, observer_loc=self.observer_loc
                    )

                    # Try single-aircraft EV for extra clarity if basic checks pass
                    ev_reason_text = None
                    ev_val = None
                    if ok and icao in aircraft_dict: # Only run EV if basic checks passed
                         try:
                              hs = self.indi_controller.get_hardware_status() or {}
                              current_az_el = hs.get("mount_az_el", (0.0, 0.0))
                              ev_res = calculate_expected_value(current_az_el, icao, aircraft_dict[icao])
                              ev_val = float(ev_res.get('ev', 0.0))
                              if ev_val <= 0.0:
                                   ev_reason = ev_res.get('reason', 'unknown_ev_reason')
                                   # Map reason codes to human-readable text
                                   reason_map = {
                                       'no_intercept': "cannot slew to intercept within limits",
                                       'late_intercept': "intercept occurs too late (outside horizon)",
                                       'low_quality': "predicted quality below minimum during track",
                                       'no_prediction': "cannot predict flight path",
                                       'prediction_failed': "prediction calculation error"
                                   }
                                   ev_reason_text = reason_map.get(ev_reason, f"EV too low ({ev_val:.2f})")
                                   # Avoid duplicating similar reasons
                                   if ev_reason_text not in reasons:
                                        reasons.append(ev_reason_text)
                         except Exception as e:
                              print(f"  Warning: EV calculation failed for manual target {icao}: {e}")
                              reasons.append("EV calculation failed")


                    # Cache detailed viability info for dashboard
                    self.manual_viability_info = {
                        "viable": bool(ok and ev_val is not None and ev_val > 0.0), # Must pass basic checks AND have positive EV
                        "reasons": reasons,
                        "details": details,
                        "ev": ev_val,
                    }

                    reason_str = "; ".join(reasons) if reasons else "basic checks passed, but EV <= 0 or not calculated"
                    print(f"  Manual target {icao} not viable: {reason_str}. Will keep trying.")

                    # Update dashboard status immediately
                    write_status({
                        "manual_target": { "icao": icao, "viability": self.manual_viability_info }
                    })

                    # Schedule a retry, but don't constantly spin
                    if not self._scheduler_timer or not self._scheduler_timer.is_alive():
                        retry_s = float(CONFIG.get('selection', {}).get('manual_retry_interval_s', 5.0)) # Shortened retry
                        print(f"  Scheduling retry for manual target in {retry_s}s.")
                        self._scheduler_timer = threading.Timer(retry_s, self._run_scheduler)
                        self._scheduler_timer.daemon = True
                        self._scheduler_timer.start()
                    return # Do not proceed with automatic selection

            # Automatic Target Selection (if no viable manual target)
            if not target_to_consider:
                if candidates:
                    target_to_consider = candidates[0]
                    # print(f"  Scheduler: Best automatic target is {target_to_consider['icao']} (EV: {target_to_consider['ev']:.1f}).") # Debug log
                else:
                    # print("  Scheduler: No viable automatic targets found.") # Debug log
                    return # No targets at all

            # --- Decide Whether to Start Track Now or Wait ---
            now = time.time()
            # Ensure necessary keys exist in the target dict
            if not all(k in target_to_consider for k in ['intercept_time', 'start_state']) or \
               not all(k in target_to_consider['start_state'] for k in ['time']):
                 print(f"  Scheduler: Target {target_to_consider['icao']} missing required timing data. Skipping.")
                 return


            intercept_duration = target_to_consider['intercept_time'] # Duration of slew
            # Time when track should ideally start (intercept finishes just before this)
            track_start_time_abs = target_to_consider['start_state']['time']
            # Time when slew needs to BEGIN to arrive on time
            # Add a small buffer (e.g., 5s) before track start
            slew_start_time_abs = track_start_time_abs - intercept_duration - 5.0

            delay_needed = slew_start_time_abs - now

            if delay_needed > 1.0: # Only schedule delay if > 1 second needed
                # Limit maximum wait time before re-evaluating (e.g., 30s)
                wait_duration = min(delay_needed, 30.0)
                # Cancel existing timer if present
                if self._scheduler_timer and self._scheduler_timer.is_alive():
                    self._scheduler_timer.cancel()
                # Schedule self to run again after wait_duration
                self._scheduler_timer = threading.Timer(wait_duration, self._run_scheduler)
                self._scheduler_timer.daemon = True
                self._scheduler_timer.start()
                print(f"  Scheduler: Waiting {wait_duration:.1f}s (of {delay_needed:.1f}s total) to start slew for {target_to_consider['icao']}.")
                return # Exit, will re-evaluate after delay

            # --- Start Tracking ---
            # Cancel any existing timer if we proceed immediately
            if self._scheduler_timer and self._scheduler_timer.is_alive():
                self._scheduler_timer.cancel()
                self._scheduler_timer = None

            icao_to_track = target_to_consider['icao']
            # Get the absolute latest data for this ICAO before starting thread
            latest_data = read_aircraft_data().get(icao_to_track)
            if not latest_data:
                print(f"  Scheduler: No current data for {icao_to_track} just before starting track; deferring.")
                # Schedule a quick retry?
                if not self._scheduler_timer or not self._scheduler_timer.is_alive():
                     self._scheduler_timer = threading.Timer(2.0, self._run_scheduler) # Retry in 2s
                     self._scheduler_timer.daemon = True
                     self._scheduler_timer.start()
                return

            # Set tracking state variables
            self.current_track_icao = icao_to_track
            self.current_track_ev = target_to_consider['ev']
            self.shutdown_event.clear() # Ensure shutdown is not set
            # Update status immediately
            write_status({"mode": "acquiring", "icao": icao_to_track})
            print(f"  Scheduler: Starting track thread for {icao_to_track} (EV: {target_to_consider['ev']:.1f})")

            # Prepare start state info for the tracking thread
            start_state_info = target_to_consider['start_state'].copy()
            # Pass intercept duration (slew time) explicitly
            start_state_info['slew_duration_s'] = intercept_duration

            # Launch the tracking thread
            self.active_thread = threading.Thread(
                target=self.track_aircraft,
                args=(icao_to_track, latest_data, start_state_info),
                daemon=True # Ensure thread exits if main program exits
            )
            self.active_thread.start()


    def track_aircraft(self, icao: str, aircraft_data: dict, start_state: dict):
        """The main tracking logic thread."""
        try:
            # Check for immediate shutdown request before starting
            if self.shutdown_event.is_set():
                print(f"  Track [{icao}]: Aborted immediately before slew.")
                return

            # --- Initial Slew ---
            target_az = start_state.get('az')
            target_el = start_state.get('el')
            if target_az is None or target_el is None:
                 print(f"  Track [{icao}]: Invalid start state coordinates. Aborting.")
                 return

            target_az_el = (target_az, target_el)
            # Update status before slew
            write_status({"mode": "slewing", "target_az_el": list(target_az_el), "icao": icao})
            print(f"  Track [{icao}]: Slewing to intercept point ({target_az:.2f}, {target_el:.2f})...")

            # Progress callback for dashboard updates during slew
            def _slew_progress(cur_az: float, cur_el: float, state: str):
                # Predict target's current position for dashboard display
                live_target_az_el = self._predict_target_az_el(aircraft_data, when=time.time()) or target_az_el
                payload = {
                    "mode": "slewing",
                    "icao": icao,
                    "mount_az_el": [float(cur_az), float(cur_el)],
                    "target_az_el": list(live_target_az_el), # Show where target is NOW
                }
                write_status(payload)

            # Execute slew command
            if not self.indi_controller.slew_to_az_el(target_az, target_el, progress_cb=_slew_progress):
                print(f"  Track [{icao}]: Initial slew failed or aborted. Ending track.")
                # Status might have been set to abort by callback, ensure idle if not
                current_status = {"mode": "idle", "icao": None}
                # Check if manual target failed, keep it visible if so
                if self.manual_target_icao == icao:
                     current_status["manual_target"] = {
                         "icao": icao,
                         "viability": self.manual_viability_info or {"viable": False, "reasons": ["slew failed"]},
                     }
                else: # Clear potentially stale manual target info if this was automatic
                     current_status["manual_target"] = None
                write_status(current_status)
                return # Exit thread

            # Check for shutdown again after potentially long slew
            if self.shutdown_event.is_set():
                print(f"  Track [{icao}]: Aborted after slew completion.")
                return

            # --- Autofocus ---
            current_alt_ft = aircraft_data.get('alt') # Get altitude from initial data
            write_status({"mode": "focusing", "autofocus_alt_ft": current_alt_ft, "icao": icao})
            print(f"  Track [{icao}]: Performing autofocus...")
            if not self.indi_controller.autofocus(current_alt_ft):
                print(f"  Track [{icao}]: Autofocus failed. Ending track.")
                write_status({"mode": "idle", "icao": None, "error_message": "Autofocus failed"})
                return # Exit thread if focus fails

            # Check for shutdown again after potentially long focus
            if self.shutdown_event.is_set():
                print(f"  Track [{icao}]: Aborted after focus completion.")
                return

            # --- Optical Guiding Loop ---
            print(f"  Track [{icao}]: Starting optical guiding loop.")
            guide_cfg = CONFIG['capture'].get('guiding', {})
            calib_cfg = CONFIG.get('pointing_calibration', {})
            cam_specs = CONFIG.get('camera_specs', {})
            # Calculate frame center safely with defaults
            frame_w = cam_specs.get('resolution_width_px', 640)
            frame_h = cam_specs.get('resolution_height_px', 480)
            frame_center = (frame_w / 2.0, frame_h / 2.0)

            # Get AltAz frame for coordinate calculations inside loop
            frame = get_altaz_frame(self.observer_loc)
            plate_scale = calculate_plate_scale() # Arcsec/px

            # --- Dry Run Simulation Setup ---
            simulated_detection_result = None
            if CONFIG['development']['dry_run']:
                # Simulate initial error based on timing difference
                slew_duration_s = start_state.get('slew_duration_s', 30.0) # From scheduler
                # Use a smaller timing error simulation factor
                timing_error_s = abs(start_state.get('time', time.time()) - (time.time() + slew_duration_s)) * 0.5
                # Call original detect_aircraft ONLY for simulation setup
                simulated_detection_result = detect_aircraft(None, sim_initial_error_s=timing_error_s)


            # --- Loop Variables ---
            captures_taken = 0
            consecutive_losses = 0
            iteration = 0
            last_seq_ts = 0.0 # Timestamp of the last capture sequence start
            # Guiding parameters from config with defaults
            max_losses = int(guide_cfg.get('max_consecutive_losses', 5))
            deadzone_px = float(guide_cfg.get('deadzone_px', 5.0))
            pulse_ms = int(guide_cfg.get('pulse_duration_ms', 100))
            settle_s = float(guide_cfg.get('settle_time_s', 0.5))
            save_guide_png = guide_cfg.get('save_guide_png', True)
            min_sequence_interval_s = float(CONFIG['capture'].get('min_sequence_interval_s', 1.0))
            min_quality_threshold = float(CONFIG['selection'].get('track_quality_min', 0.2))


            # --- Main Guide Loop ---
            while True:
                iteration += 1
                loop_start_time = time.time()

                if self.shutdown_event.is_set():
                    print(f"  Track [{icao}]: Shutdown signaled, exiting guide loop.")
                    break # Exit loop cleanly

                # --- Capture Guide Image ---
                try:
                    guide_path = self.indi_controller.snap_image(f"{icao}_g{iteration}")
                    if not guide_path:
                        # Handle case where snap_image returns None (e.g., capture failed)
                        raise RuntimeError("snap_image returned None or empty path")
                except Exception as e:
                    print(f"  Track [{icao}]: Guide frame {iteration} capture failed: {e}")
                    consecutive_losses += 1
                    if consecutive_losses >= max_losses:
                        print(f"  Track [{icao}]: Exceeded max consecutive capture failures ({max_losses}). Ending track.")
                        break # Exit loop
                    time.sleep(1.0) # Wait before retrying
                    continue # Skip rest of loop iteration

                # --- Load Image Data ONCE ---
                guide_data = _load_fits_data(guide_path)
                if guide_data is None:
                    print(f"  Track [{icao}]: Failed to load guide image data for {guide_path}. Skipping frame.")
                    consecutive_losses += 1
                    if consecutive_losses >= max_losses:
                         print(f"  Track [{icao}]: Exceeded max consecutive load failures ({max_losses}). Ending track.")
                         break
                    time.sleep(0.5)
                    continue
                guide_shape = guide_data.shape # Store shape for detection function

                # --- Create PNG Preview (from data) ---
                guide_png_url = "" # Default to empty URL
                guide_png_path_generated = "" # Store the path confirmed by the save function

                if save_guide_png:
                    try:
                        # Construct the expected PNG path based on FITS path
                        guide_png_path_expected = os.path.splitext(guide_path)[0] + ".png"
                        # print(f"  Track [{icao}]: Attempting to save PNG preview to {os.path.basename(guide_png_path_expected)}") # DEBUG

                        # Call internal save function with data
                        saved_png_path = _save_png_preview_from_data(guide_data, guide_png_path_expected)

                        # --- Refined Check ---
                        if saved_png_path and os.path.exists(saved_png_path):
                            guide_png_path_generated = saved_png_path # Store the successfully saved path
                            guide_png_url = _to_img_url(guide_png_path_generated) # Convert THIS path to URL
                            # print(f"  Track [{icao}]: PNG saved successfully. URL: {guide_png_url}") # DEBUG
                        else:
                            print(f"  Track [{icao}]: Warning - _save_png_preview_from_data did not return a valid path or file doesn't exist.")
                        # --- End Refined Check ---

                    except Exception as e:
                        print(f"  Track [{icao}]: Warning - Error occurred during PNG creation/check: {e}")


                # --- Detect Aircraft (from data) ---
                detection = simulated_detection_result if simulated_detection_result else _detect_aircraft_from_data(guide_data, original_shape=guide_shape)


                # --- Handle Detection Failure ---
                if not detection or not detection.get('detected') or not detection.get('center_px'):
                    consecutive_losses += 1
                    reason = (detection or {}).get('reason', 'unknown')
                    sharp = detection.get('sharpness', -1)
                    conf = detection.get('confidence', -1)
                    print(f"  Track [{icao}]: Guide frame {iteration}: Target lost ({consecutive_losses}/{max_losses}). "
                          f"Reason: {reason} (Sharp: {sharp:.1f}, Conf: {conf:.2f})")
                    # Update status indicating loss but still tracking
                    write_status({
                        "mode": "tracking", "icao": icao, "iteration": iteration,
                        "stable": False, "last_guide_png": guide_png_url, # Use generated URL
                        "guide_offset_px": None # Clear offset on loss
                    })
                    if consecutive_losses >= max_losses:
                        print(f"  Track [{icao}]: Target lost for {max_losses} consecutive frames. Ending track.")
                        break # Exit loop
                    time.sleep(1.0) # Wait longer after a loss before next attempt
                    continue # Skip rest of loop

                # --- Process Detection Success ---
                consecutive_losses = 0 # Reset loss counter on success
                center_px = detection['center_px']
                # Validate centroid coordinates
                if not (len(center_px) == 2 and math.isfinite(center_px[0]) and math.isfinite(center_px[1])):
                    print(f"  Track [{icao}]: Invalid centroid coordinates received: {center_px}. Treating as lost.")
                    # Treat as loss, but don't increment loss counter here? Or should we? Incrementing for safety.
                    consecutive_losses += 1
                    write_status({"mode": "tracking", "icao": icao, "iteration": iteration, "stable": False, "last_guide_png": guide_png_url}) # Use URL
                    time.sleep(0.5)
                    continue

                # --- Calculate Guiding Offset ---
                offset_px_x = center_px[0] - frame_center[0]
                offset_px_y = center_px[1] - frame_center[1]

                # Correct for camera rotation if angle is non-zero
                rotation_rad = math.radians(calib_cfg.get('rotation_angle_deg', 0.0))
                if abs(rotation_rad) > 1e-3: # Apply correction only if significant rotation
                    cos_rot, sin_rot = math.cos(rotation_rad), math.sin(rotation_rad)
                    # Apply rotation matrix to offset vector
                    corrected_dx = offset_px_x * cos_rot - offset_px_y * sin_rot
                    corrected_dy = offset_px_x * sin_rot + offset_px_y * cos_rot
                else:
                     corrected_dx = offset_px_x
                     corrected_dy = offset_px_y

                # Check if guiding is stable (within deadzone)
                guide_error_mag = max(abs(corrected_dx), abs(corrected_dy))
                is_stable = guide_error_mag <= deadzone_px

                # --- Live Quality Check ---
                now = time.time()
                try:
                    # Get sun position for now
                    sun_az, sun_el = get_sun_azel(now, self.observer_loc)
                    # Predict aircraft position for now and 1 second later
                    est_list = estimate_positions_at_times(aircraft_data, [now, now + 1.0])
                    if len(est_list) < 2:
                        print(f"  Track [{icao}]: Could not predict motion for quality check. Ending track.")
                        break # Cannot proceed without prediction

                    current_pos_est, next_sec_pos_est = est_list[0], est_list[1]
                    # Convert predicted positions to Az/El
                    current_az, current_el = latlonalt_to_azel(current_pos_est['est_lat'], current_pos_est['est_lon'], current_pos_est['est_alt'], now, self.observer_loc)
                    next_sec_az, next_sec_el = latlonalt_to_azel(next_sec_pos_est['est_lat'], next_sec_pos_est['est_lon'], next_sec_pos_est['est_alt'], now + 1.0, self.observer_loc)

                    # Calculate current angular speed
                    ang_speed = angular_speed_deg_s((current_az, current_el), (next_sec_az, next_sec_el), 1.0, frame)

                    # Calculate current range (haversine_distance returns km)
                    current_range_km = haversine_distance(
                        self.observer_loc.lat.deg, self.observer_loc.lon.deg,
                        current_pos_est['est_lat'], current_pos_est['est_lon']
                    )
                    # Calculate current sun separation
                    current_sun_sep = angular_sep_deg((current_az, current_el), (sun_az, sun_el), frame)

                    # Build state dictionary for quality function
                    current_quality_state = {
                        'el': current_el,
                        'sun_sep': current_sun_sep,
                        'range_km': current_range_km,
                        'ang_speed': ang_speed
                    }
                    current_quality = calculate_quality(current_quality_state)

                    # Check if quality dropped below minimum threshold
                    if current_quality < min_quality_threshold:
                        print(f"  Track [{icao}]: Target quality ({current_quality:.2f}) dropped below threshold ({min_quality_threshold}). Ending track.")
                        break # Exit loop due to low quality

                except Exception as e:
                     print(f"  Track [{icao}]: Error during live quality check: {e}. Ending track for safety.")
                     break # Exit loop if quality check fails


                # --- Update Status ---
                print(f"  Track [{icao}]: Frame {iteration}: Offset ({corrected_dx:+.1f}, {corrected_dy:+.1f}) px. Error: {guide_error_mag:.1f}px. Stable: {is_stable}")
                # Get fresh hardware status before writing full update
                hardware_status = self.indi_controller.get_hardware_status() or {}
                status_payload = {
                    "mode": "tracking", "icao": icao, "iteration": iteration,
                    "guide_offset_px": {"dx": float(corrected_dx), "dy": float(corrected_dy)},
                    "stable": is_stable, "last_guide_png": guide_png_url, # Use URL
                    # Use basename of the FITS path for file reference
                    "last_guide_file": os.path.basename(guide_path),
                    # Use latest aircraft data used for predictions/quality
                    "target_details": {
                        "lat": current_pos_est.get('est_lat'), "lon": current_pos_est.get('est_lon'),
                        "alt": current_pos_est.get('est_alt'),
                        # Include original data too? Maybe subset?
                        "flight": aircraft_data.get('flight', '').strip() or 'N/A',
                        "gs": aircraft_data.get('gs'), "track": aircraft_data.get('track')
                    },
                    "sun_pos": {"az": sun_az, "el": sun_el},
                    "sun_sep_target_deg": current_sun_sep,
                    "sun_sep_mount_deg": angular_sep_deg(hardware_status.get("mount_az_el", (0,0)), (sun_az, sun_el), frame),
                    "image_vitals": {"sharpness": detection.get('sharpness'), "confidence": detection.get('confidence')},
                    "current_quality": round(current_quality, 3), # Add live quality score
                    "current_range_km": round(current_range_km, 1) # Add live range
                }
                status_payload.update(hardware_status) # Merge hardware status
                write_status(status_payload)


                # --- Send Guide Pulses (if needed) ---
                pulse_sent = False
                # Use configured offset signs
                az_sign = calib_cfg.get('az_offset_sign', 1)
                el_sign = calib_cfg.get('el_offset_sign', 1)

                if corrected_dy > deadzone_px:
                    self.indi_controller.guide_pulse('N' if el_sign > 0 else 'S', pulse_ms)
                    pulse_sent = True
                elif corrected_dy < -deadzone_px:
                    self.indi_controller.guide_pulse('S' if el_sign > 0 else 'N', pulse_ms)
                    pulse_sent = True

                if corrected_dx > deadzone_px:
                    self.indi_controller.guide_pulse('E' if az_sign > 0 else 'W', pulse_ms)
                    pulse_sent = True
                elif corrected_dx < -deadzone_px:
                    self.indi_controller.guide_pulse('W' if az_sign > 0 else 'E', pulse_ms)
                    pulse_sent = True

                # Wait for mount to settle after pulse(s)
                if pulse_sent:
                    time.sleep(settle_s)

                # --- Dry Run: Simulate Guide Correction ---
                if CONFIG['development']['dry_run'] and simulated_detection_result:
                    # Simulate imperfect correction (e.g., reduce error by 70%)
                    new_offset_x = offset_px_x * 0.3
                    new_offset_y = offset_px_y * 0.3
                    # Add small random walk to prevent perfect centering
                    new_offset_x += (np.random.rand() - 0.5) * 2.0 # +/- 1 pixel noise
                    new_offset_y += (np.random.rand() - 0.5) * 2.0
                    # Update simulated result for next iteration
                    simulated_detection_result = detection.copy() # Start with current detection
                    simulated_detection_result['center_px'] = (
                        frame_center[0] + new_offset_x,
                        frame_center[1] + new_offset_y
                    )


                # --- Capture Image Sequence (if stable) ---
                now = time.time() # Get current time again
                if is_stable and (now - last_seq_ts) > min_sequence_interval_s:
                    print(f"  Track [{icao}]: Guiding stable. Capturing sequence...")

                    capture_cfg = CONFIG['capture']
                    base_exposure = float(capture_cfg.get('sequence_exposure_s', 0.5))
                    # Use guide exposure for adjustment calculation if available
                    guide_exposure_s = float(guide_cfg.get('guide_exposure_s', capture_cfg.get('snap_exposure_s', 0.1)))


                    # Estimate exposure adjustment using the guide frame data
                    exposure_factor = 1.0
                    if not CONFIG['development']['dry_run']:
                        try:
                            # Call internal function with data
                            exposure_factor = _estimate_exposure_adjustment_from_data(guide_data)
                            print(f"    - Auto-exposure factor: {exposure_factor:.2f}")
                        except Exception as e:
                             print(f"    - Warning: Failed to estimate exposure adjustment: {e}")
                             exposure_factor = 1.0 # Default to 1.0 on error


                    # Apply min/max factor limits from config
                    adj_min = float(capture_cfg.get('exposure_adjust_factor_min', 0.2)) # Wider default range
                    adj_max = float(capture_cfg.get('exposure_adjust_factor_max', 5.0))
                    exposure_factor = max(adj_min, min(adj_max, exposure_factor))

                    # Calculate recommended exposure based on factor
                    recommended_exp = base_exposure * exposure_factor

                    # --- Calculate Max Exposure based on Motion Blur ---
                    blur_px_limit = float(capture_cfg.get('target_blur_px', 1.0))
                    # Calculate max exposure to keep blur below limit
                    # t_max = (blur_pixels * plate_scale_arcsec) / (angular_speed_deg_sec * 3600)
                    t_max_blur = float('inf')
                    if ang_speed > 1e-3 and plate_scale > 1e-3: # Avoid division by zero
                         t_max_blur = (blur_px_limit * plate_scale) / (ang_speed * 3600.0)

                    # Determine final exposure, applying all limits
                    min_exp_limit = float(capture_cfg.get('exposure_min_s', 0.001))
                    max_exp_limit = float(capture_cfg.get('exposure_max_s', 5.0)) # Sequence max exposure

                    final_exp = recommended_exp
                    limit_reason = "recommended"

                    if final_exp > t_max_blur:
                        final_exp = t_max_blur
                        limit_reason = f"blur_limit ({t_max_blur:.3f}s)"

                    if final_exp > max_exp_limit:
                         final_exp = max_exp_limit
                         # Update reason only if this is the binding constraint
                         if limit_reason == "recommended" or t_max_blur > max_exp_limit:
                              limit_reason = f"config_max ({max_exp_limit:.3f}s)"

                    if final_exp < min_exp_limit:
                         final_exp = min_exp_limit
                         # Update reason only if this is the binding constraint
                         if limit_reason == "recommended" or t_max_blur < min_exp_limit:
                             limit_reason = f"config_min ({min_exp_limit:.3f}s)"


                    print(f"    - Final sequence exposure: {final_exp:.4f}s (Reason: {limit_reason})")

                    try:
                        seq_start_ts = time.time() # Record sequence start time
                        # Call hardware controller to capture the sequence
                        captured_paths = self.indi_controller.capture_sequence(icao, final_exp)

                        if captured_paths:
                            last_seq_ts = seq_start_ts # Update timestamp only if successful
                            # Log sequence details
                            seq_log = {
                                'type': 'sequence',
                                'sequence_id': int(seq_start_ts * 1000), # Use timestamp as ID
                                'icao': icao,
                                'timestamp': seq_start_ts,
                                'per_frame_exposure_s': float(final_exp),
                                'exposure_limit_reason': limit_reason,
                                'n_frames': len(captured_paths),
                                'image_paths': captured_paths, # Store relative paths? or full? Full for now.
                                'guide_offset_px': (float(corrected_dx), float(corrected_dy)),
                                'plate_scale_arcsec_px': float(plate_scale),
                                'predicted_ang_speed_deg_s': float(ang_speed), # Speed at time of capture decision
                                'target_blur_px_limit': float(blur_px_limit),
                                # Include key detection results?
                                'last_guide_sharpness': detection.get('sharpness'),
                                'last_guide_confidence': detection.get('confidence'),
                            }
                            append_to_json([seq_log], os.path.join(LOG_DIR, 'captures.json'))

                            captures_taken += len(captured_paths)

                            # --- Update Status with Capture Info ---
                            status_update = {
                                "sequence_exposure_s": float(final_exp),
                                "sequence_count": len(captured_paths),
                                "captures_taken": captures_taken,
                                "sequence_meta": { # Add some context
                                    "ang_speed_deg_s": float(ang_speed),
                                    "target_blur_px_limit": float(blur_px_limit),
                                    "exposure_limit_reason": limit_reason
                                }
                            }
                            # Get path of the last image captured in sequence for display
                            if captured_paths:
                                 status_update["last_capture_file"] = os.path.basename(captured_paths[-1])
                                 # Optionally generate PNG URL here? Orchestrator does stack preview.
                                 # Maybe link to the last *raw* frame's PNG?
                                 last_raw_png_path = os.path.splitext(captured_paths[-1])[0] + ".png"
                                 # Check if the PNG file actually exists (it should have been created by capture_image)
                                 if os.path.exists(last_raw_png_path):
                                      status_update["capture_png"] = _to_img_url(last_raw_png_path)


                            write_status(status_update)

                    except Exception as e:
                        print(f"  Track [{icao}]: Capture sequence failed: {e}")
                        # Don't break loop, just log error and continue guiding


                # Small delay at end of loop to prevent excessive CPU usage if guiding is fast
                loop_duration = time.time() - loop_start_time
                sleep_time = max(0.01, 0.1 - loop_duration) # Aim for ~10Hz max loop rate
                time.sleep(sleep_time)


        # --- Cleanup after Loop Exit ---
        finally:
            print(f"  Track [{icao}]: Exiting tracking thread.")
            with self.track_lock:
                # Clear tracking state variables
                was_preempted = self.preempt_requested
                # Store ICAO before clearing, in case needed
                finished_icao = self.current_track_icao

                self.current_track_icao = None
                self.current_track_ev = 0.0
                self.active_thread = None
                self.preempt_requested = False # Reset preemption flag

            was_aborted_by_signal = self.shutdown_event.is_set()
            # Clear the shutdown event now that the thread is exiting
            self.shutdown_event.clear()

            # --- Update Final Status ---
            final_status = {"mode": "idle", "icao": None}
            # If aborted by external signal (Ctrl+C, Abort button, Park), report idle
            if was_aborted_by_signal and not was_preempted:
                 print(f"  Tracking for {finished_icao} was interrupted.")
                 # If it was a manual target that got aborted, keep showing it as manual
                 if self.manual_target_icao == finished_icao:
                      final_status["manual_target"] = {
                           "icao": finished_icao,
                           "viability": {"viable": False, "reasons": ["track aborted"]},
                      }
                 else: # Clear manual target if automatic track was aborted
                      final_status["manual_target"] = None

            # If preempted, report idle but immediately trigger scheduler
            elif was_preempted:
                print(f"  Tracking for {finished_icao} preempted. Re-evaluating targets...")
                final_status["manual_target"] = None # Clear any old manual target info

            # If loop finished normally (e.g., target lost, low quality)
            else:
                print(f"  Tracking for {finished_icao} finished normally. System idle.")
                final_status["manual_target"] = None # Clear any old manual target info

            # Write the final status update
            write_status(final_status)

            # --- Trigger Scheduler if needed ---
            # Trigger immediately if preempted or finished normally to find next target
            if was_preempted or not was_aborted_by_signal:
                 print("  Triggering scheduler to find next target...")
                 # Run scheduler in a new thread to avoid blocking cleanup? Or just call directly?
                 # Calling directly should be okay as it's the last action.
                 self._run_scheduler()



if __name__ == "__main__":
    ensure_log_dir() # Create log dirs at startup

    # --- Setup Watchdog Observers ---
    adsb_watch_file = os.path.normpath(os.path.abspath(CONFIG['adsb']['json_file_path']))
    adsb_watch_path = os.path.dirname(adsb_watch_file)
    if not os.path.exists(adsb_watch_path):
         print(f"FATAL: ADS-B watch directory '{adsb_watch_path}' does not exist.")
         exit(1)
    if not os.path.exists(adsb_watch_file):
         print(f"Warning: ADS-B file '{adsb_watch_file}' does not exist yet. Waiting for creation...")


    event_handler = FileHandler(adsb_watch_file)

    # Perform initial read attempt
    print("Performing initial read of aircraft data...")
    event_handler.process_file() # This will also init hardware if possible

    # ADS-B File Observer
    adsb_observer = Observer()
    adsb_observer.schedule(event_handler, path=adsb_watch_path, recursive=False)

    # Command File Observer
    command_handler = CommandHandler(event_handler)
    command_observer = Observer()
    # Ensure LOG_DIR exists before scheduling observer
    os.makedirs(LOG_DIR, exist_ok=True)
    command_observer.schedule(command_handler, path=LOG_DIR, recursive=False)

    # --- Start Observers ---
    try:
        adsb_observer.start()
        command_observer.start()
        print(f"Monitoring '{adsb_watch_path}' for ADS-B data...")
        print(f"Monitoring '{LOG_DIR}' for command file '{CONFIG['logging']['command_file']}'...")

        # --- Main Loop (Keep Alive) ---
        while True:
            # Check if hardware init failed previously and retry periodically?
            # Or just rely on file events to trigger retries.
            # Check if observer threads are alive?
            if not adsb_observer.is_alive() or not command_observer.is_alive():
                 print("Error: Watchdog observer thread died. Exiting.")
                 break
            time.sleep(5) # Keep main thread alive, check observers periodically

    except KeyboardInterrupt:
        print("\nShutdown requested (Ctrl+C)...")
        # Signal tracking thread to stop first
        event_handler.shutdown_event.set()
        # Cancel any pending scheduler timer
        if event_handler._scheduler_timer and event_handler._scheduler_timer.is_alive():
            event_handler._scheduler_timer.cancel()
            print("Cancelled pending scheduler timer.")
        # Wait for tracking thread to finish
        active_thread = event_handler.active_thread # Get ref before stopping observers
        if active_thread and active_thread.is_alive():
            print("Waiting for tracking thread to finish...")
            active_thread.join(timeout=15.0) # Increased timeout
            if active_thread.is_alive():
                 print("Warning: Tracking thread did not exit cleanly.")

    except Exception as e:
         print(f"FATAL: Unhandled exception in main loop: {e}")
         # Ensure shutdown is signalled on any main loop error
         event_handler.shutdown_event.set()

    finally:
        # --- Cleanup ---
        print("Stopping observers...")
        if adsb_observer.is_alive(): adsb_observer.stop()
        if command_observer.is_alive(): command_observer.stop()

        # Disconnect hardware if initialized
        if event_handler.indi_controller:
            print("Disconnecting hardware...")
            # Consider parking before disconnect? Park command handles this now.
            event_handler.indi_controller.disconnect()

        # Wait for observer threads to join
        try:
             if adsb_observer.is_alive(): adsb_observer.join(timeout=2.0)
             if command_observer.is_alive(): command_observer.join(timeout=2.0)
        except RuntimeError:
             # Can happen if threads already stopped
             pass

        print("Program terminated.")