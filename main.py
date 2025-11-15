#main.py
"""
Main entry point: Monitors ADS-B data, processes flight paths, and tracks aircraft.
"""

import time
import threading
import os
import math
import json
import queue
import numpy as np
from typing import Optional, Callable
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
from image_analyzer import (
    _load_fits_data,
    _detect_aircraft_from_data,
    _estimate_exposure_adjustment_from_data,
    _save_png_preview_from_data
)
# --- Keep original functions for dry run simulation if needed ---
from image_analyzer import detect_aircraft
from coord_utils import (calculate_plate_scale, latlonalt_to_azel, distance_km,  # <-- USE THE NEW NAME
                         get_sun_azel, angular_sep_deg, angular_speed_deg_s, get_altaz_frame)
from stack_orchestrator import shutdown as shutdown_stack_orchestrator, request_shutdown


def _to_img_url(fs_path: str) -> str:
    """Converts a filesystem image path to its corresponding dashboard URL."""
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
                pass  # Continue processing the command we read

            if 'track_icao' in command:
                icao = command['track_icao'].lower().strip()
                print(f"  Manual override to track ICAO: {icao}")

                with self.main_handler.track_lock:
                    # Cancel any pending scheduled task
                    if self.main_handler._scheduler_timer and self.main_handler._scheduler_timer.is_alive():
                        self.main_handler._scheduler_timer.cancel()
                        self.main_handler._scheduler_timer = None
                        self.main_handler.is_scheduler_waiting = False  # Reset flag (#6)
                        print("  Cancelled pending scheduler timer.")
                    # If currently tracking, request preemption
                    if self.main_handler.current_track_icao and self.main_handler.active_thread and self.main_handler.active_thread.is_alive():
                        print("  Interrupting current track for manual override...")
                        self.main_handler.preempt_requested = True
                        self.main_handler.shutdown_event.set()  # Signal current track to stop
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
                        self.main_handler.is_scheduler_waiting = False  # Reset flag (#6)
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
                        self.main_handler.is_scheduler_waiting = False  # Reset flag (#6)
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
                    active_thread_to_join.join(timeout=10.0)  # Increased timeout
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
        self.debounce_seconds = 0.5  # Reduce debounce slightly?
        self.indi_controller = None
        self.current_track_icao = None
        self.current_track_ev = 0.0
        self.track_lock = threading.Lock()  # Protects access to tracking state vars
        obs_cfg = CONFIG['observer']
        # Ensure lat/lon/alt are floats
        try:
            lat = float(obs_cfg['latitude_deg'])
            lon = float(obs_cfg['longitude_deg'])
            alt = float(obs_cfg['altitude_m'])
            self.observer_loc = EarthLocation(lat=lat * u.deg, lon=lon * u.deg, height=alt * u.m)
        except (ValueError, KeyError) as e:
            print(f"FATAL: Invalid observer location in config: {e}")
            raise ValueError(f"Invalid observer location config: {e}") from e

        self.shutdown_event = threading.Event()  # Signals tracking thread to stop
        self.active_thread = None  # Reference to the current tracking thread
        self.last_file_stat = None  # To detect redundant file events
        self.manual_target_icao = None  # ICAO requested manually
        self.manual_viability_info = None  # Cache last manual viability details for status
        self._scheduler_timer = None  # Timer for delayed scheduling
        self.preempt_requested = False  # Flag for preemptive track switching
        self.is_scheduler_waiting = False
        self.capture_queue = queue.Queue()
        self.capture_worker_thread = threading.Thread(target=self._capture_worker_loop, daemon=True)
        self.capture_worker_thread.start()

        # Shared counter for the number of images captured for the current track.
        # This counter is protected by track_lock and reset whenever a new track begins.
        self.captures_taken = 0

    def _capture_worker_loop(self):
        """
        Worker thread that runs blocking captures from the queue.
        This frees the main tracking loop from blocking I/O.
        """
        print("  Capture worker thread started.")
        while not self.shutdown_event.is_set():
            try:
                # Wait for a job, but timeout to check shutdown_event
                job = self.capture_queue.get(timeout=1.0)
                if job is None:
                    continue

                (icao, final_exp, seq_log_base,
                 captures_taken_so_far, status_payload_base) = job

                captured_paths = self.indi_controller.capture_sequence(icao, final_exp)

                if captured_paths:
                    # Update log and status *after* capture is complete
                    seq_log = seq_log_base.copy()
                    seq_log['n_frames'] = len(captured_paths)
                    seq_log['image_paths'] = captured_paths
                    append_to_json([seq_log], os.path.join(LOG_DIR, 'captures.json'))

                    captures_taken = captures_taken_so_far + len(captured_paths)
                    status_update = status_payload_base.copy()
                    status_update["sequence_count"] = len(captured_paths)
                    status_update["captures_taken"] = captures_taken

                    if captured_paths:
                        status_update["last_capture_file"] = os.path.basename(captured_paths[-1])
                        last_raw_png_path = os.path.splitext(captured_paths[-1])[0] + ".png"
                        if os.path.exists(last_raw_png_path):
                            status_update["capture_png"] = _to_img_url(last_raw_png_path)
                    
                    # Update the shared captures_taken counter atomically.
                    with self.track_lock:
                        if self.current_track_icao == icao:
                            self.captures_taken = captures_taken
                    # Write status with updated capture count.
                    write_status(status_update)

            except queue.Empty:
                # This is normal, just loop and check shutdown_event
                continue
            except Exception as e:
                # Log errors from the capture worker
                print(f"  Capture worker error: {e}")
                import traceback
                traceback.print_exc()

        print("  Capture worker thread shutting down.")

    def process_file(self):
        """Processes aircraft.json, runs the scheduler, and manages state."""
        # --- File Stat Check & Debounce ---
        try:
            current_stat = os.stat(self.watch_file)
            if self.last_file_stat and (current_stat.st_mtime == self.last_file_stat.st_mtime and
                                         current_stat.st_size == self.last_file_stat.st_size):
                return
            self.last_file_stat = current_stat
        except FileNotFoundError:
            return
        except Exception as e:
            print(f"Warning: Error stating file {self.watch_file}: {e}")
            return

        current_time = time.time()
        if current_time - self.last_process_time < self.debounce_seconds:
            return
        self.last_process_time = current_time

        # --- Read Aircraft Data ---
        aircraft_dict = read_aircraft_data()

        # --- Initialize Hardware (if not already) ---
        if not self.indi_controller:
            try:
                print("Initializing hardware controller...")
                self.indi_controller = IndiController()
                self.indi_controller.shutdown_event = self.shutdown_event
                print("Hardware controller initialized.")
            except Exception as e:
                print(f"FATAL: Hardware initialization failed: {e}")
                write_status({"mode": "error", "error_message": f"Hardware init failed: {e}"})
                self.indi_controller = None
                return

        # --- Get Current Hardware Status ---
        hardware_status = self.indi_controller.get_hardware_status() or {}

        # --- Handle Empty Airspace ---
        if not aircraft_dict:
            status_payload = hardware_status.copy()
            status_payload["mode"] = "tracking" if self.current_track_icao else "idle"
            if self.manual_target_icao:
                status_payload["manual_target"] = {
                    "icao": self.manual_target_icao,
                    "viability": self.manual_viability_info or {"viable": False, "reasons": ["no ADS-B data available"]},
                }
            write_status(status_payload)
            return

        # --- Select Candidates & Check Preemption ---
        current_az_el = hardware_status.get("mount_az_el", (0.0, 0.0))
        try:
            candidates = select_aircraft(aircraft_dict, current_az_el) or []
        except Exception as e:
            print(f"Error during aircraft selection: {e}")
            candidates = []

        # Check for preemptive switch
        if self.current_track_icao and candidates:
            new_best_target = candidates[0]
            preempt_factor = float(CONFIG['selection'].get('preempt_factor', 1.25))
            if new_best_target['icao'] != self.current_track_icao and \
                    new_best_target['ev'] > (self.current_track_ev * preempt_factor):
                print(f"--- PREEMPTIVE SWITCH: New target {new_best_target['icao']} (EV {new_best_target['ev']:.1f}) "
                      f"> Current {self.current_track_icao} (EV {self.current_track_ev:.1f} * {preempt_factor:.2f}). ---")
                with self.track_lock:
                    if self._scheduler_timer and self._scheduler_timer.is_alive():
                        self._scheduler_timer.cancel()
                        self._scheduler_timer = None
                        self.is_scheduler_waiting = False  # Reset flag (#6)
                    self.preempt_requested = True
                    self.shutdown_event.set()

        # --- Build Status Payload ---
        all_aircraft_list = []
        obs_lat = float(CONFIG['observer']['latitude_deg'])
        obs_lon = float(CONFIG['observer']['longitude_deg'])
        for icao, data in aircraft_dict.items():
            lat, lon = data.get('lat'), data.get('lon')
            if lat is None or lon is None: continue

            dist_km = distance_km(obs_lat, obs_lon, lat, lon)
            dist_nm = dist_km / 1.852 if np.isfinite(dist_km) else None

            age_s = data.get('age_s')
            is_stale = (age_s is not None) and (age_s > 15.0)
            dist_nm_display = None if is_stale or dist_nm is None else dist_nm
            flight_data = data.get('flight', '').strip()

            all_aircraft_list.append({
                "icao": icao,
                "flight": flight_data or 'N/A',
                "alt": data.get('alt'),
                "gs": data.get('gs'),
                "track": data.get('track'),
                "dist_nm": dist_nm_display
            })
        all_aircraft_list.sort(key=lambda x: x['dist_nm'] if x['dist_nm'] is not None else float('inf'))

        status_payload = {
            "queue": candidates[1:6] if self.current_track_icao else candidates[:5],
            "track_candidates": candidates[:1],
            "all_aircraft": all_aircraft_list,
            "observer_loc": {"lat": obs_lat, "lon": obs_lon},
            "camera_settings": {
                'binning': CONFIG.get('camera_specs', {}).get('binning'),
                'gain': CONFIG.get('camera_specs', {}).get('gain'),
                'offset': CONFIG.get('camera_specs', {}).get('offset'),
                'cooling': CONFIG.get('camera_specs', {}).get('cooling')
            }
        }
        status_payload.update(hardware_status)
        status_payload["mode"] = "tracking" if self.current_track_icao else "idle"

        if self.manual_target_icao:
            status_payload["manual_target"] = {
                "icao": self.manual_target_icao,
                "viability": self.manual_viability_info
            }

        # --- Write Status ---
        write_status(status_payload)

        # --- Run Scheduler (if not already tracking and not waiting) ---
        # Do not invoke the scheduler if a shutdown has been requested
        if not self.shutdown_event.is_set() and not self.current_track_icao and not self.is_scheduler_waiting:
            self._run_scheduler(candidates)


    def on_modified(self, event):
        if os.path.normpath(event.src_path) == self.watch_file:
            self.process_file()

    def on_moved(self, event):
        if os.path.normpath(event.dest_path) == self.watch_file:
            self.process_file()

    def _predict_target_az_el(self, aircraft_data: dict, when: Optional[float] = None) -> Optional[tuple[float, float]]:
        t = when or time.time()
        try:
            pos_list = estimate_positions_at_times(aircraft_data, [t])
            if not pos_list: return None
            pos = pos_list[0]
            if not all(k in pos for k in ['est_lat', 'est_lon', 'est_alt']):
                print("Warning: Prediction missing required keys (lat/lon/alt).")
                return None
            az, el = latlonalt_to_azel(pos['est_lat'], pos['est_lon'], pos['est_alt'], t, self.observer_loc)
            if not (math.isfinite(az) and math.isfinite(el)):
                print(f"Warning: Non-finite az/el prediction ({az}, {el}).")
                return None
            return (float(az), float(el))
        except Exception as e:
            print(f"Error predicting target az/el: {e}")
            return None


    def _run_scheduler(self, candidates: Optional[list] = None):
        """Decides whether to start a new track or wait."""
        # Do not schedule new tracks during shutdown
        if self.shutdown_event.is_set():
            # Skip scheduling entirely if the system is shutting down
            return
        if self.is_scheduler_waiting:
            return

        with self.track_lock:
            # Check again inside the lock to avoid races
            if self.shutdown_event.is_set():
                return
            if self.current_track_icao:
                return

            if candidates is None:
                aircraft_dict = read_aircraft_data()
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
                aircraft_dict = read_aircraft_data()

            target_to_consider = None
            is_manual_override = False

            if self.manual_target_icao:
                icao = self.manual_target_icao
                print(f"  Scheduler: Evaluating manual target {icao}...")
                manual_target_in_candidates = next((c for c in candidates if c['icao'] == icao), None)

                if manual_target_in_candidates:
                    print(f"  Manual target {icao} is now viable (EV: {manual_target_in_candidates['ev']:.1f}). Selecting.")
                    target_to_consider = manual_target_in_candidates
                    is_manual_override = True
                    self.manual_target_icao = None
                    self.manual_viability_info = None
                else:
                    print(f"  Manual target {icao} not in top candidates. Checking viability...")
                    ok, reasons, details = evaluate_manual_target_viability(
                        icao, aircraft_dict, observer_loc=self.observer_loc
                    )
                    ev_reason_text = None
                    ev_val = None
                    if ok and icao in aircraft_dict:
                        try:
                            hs = self.indi_controller.get_hardware_status() or {}
                            current_az_el_ev = hs.get("mount_az_el", (0.0, 0.0))  # Use separate var name
                            ev_res = calculate_expected_value(current_az_el_ev, icao, aircraft_dict[icao])
                            ev_val = float(ev_res.get('ev', 0.0))
                            if ev_val <= 0.0:
                                ev_reason = ev_res.get('reason', 'unknown_ev_reason')
                                reason_map = {
                                    'no_intercept': "cannot slew to intercept within limits",
                                    'late_intercept': "intercept occurs too late (outside horizon)",
                                    'low_quality': "predicted quality below minimum during track",
                                    'no_prediction': "cannot predict flight path",
                                    'prediction_failed': "prediction calculation error"
                                }
                                ev_reason_text = reason_map.get(ev_reason, f"EV too low ({ev_val:.2f})")
                                if ev_reason_text not in reasons:
                                    reasons.append(ev_reason_text)
                        except Exception as e:
                            print(f"  Warning: EV calculation failed for manual target {icao}: {e}")
                            reasons.append("EV calculation failed")

                    self.manual_viability_info = {
                        "viable": bool(ok and ev_val is not None and ev_val > 0.0),
                        "reasons": reasons, "details": details, "ev": ev_val,
                    }
                    reason_str = "; ".join(reasons) if reasons else "basic checks passed, but EV <= 0 or not calculated"
                    print(f"  Manual target {icao} not viable: {reason_str}. Will keep trying.")
                    write_status({"manual_target": {"icao": icao, "viability": self.manual_viability_info}})

                    if not self._scheduler_timer or not self._scheduler_timer.is_alive():
                        retry_s = float(CONFIG.get('selection', {}).get('manual_retry_interval_s', 5.0))
                        print(f"  Scheduling retry for manual target in {retry_s}s.")
                        self.is_scheduler_waiting = True
                        self._scheduler_timer = threading.Timer(retry_s, self._run_scheduler_callback)  # Use callback
                        self._scheduler_timer.daemon = True
                        self._scheduler_timer.start()
                    return

            if not target_to_consider:
                if candidates:
                    target_to_consider = candidates[0]
                else:
                    return

            now = time.time()
            if not all(k in target_to_consider for k in ['intercept_time', 'start_state']) or \
                    not all(k in target_to_consider['start_state'] for k in ['time']):
                print(f"  Scheduler: Target {target_to_consider['icao']} missing required timing data. Skipping.")
                return

            intercept_duration = target_to_consider['intercept_time']
            track_start_time_abs = target_to_consider['start_state']['time']
            slew_start_time_abs = track_start_time_abs - intercept_duration - 5.0
            delay_needed = slew_start_time_abs - now

            if delay_needed > 1.0:
                wait_duration = min(delay_needed, 30.0)
                if self._scheduler_timer and self._scheduler_timer.is_alive():
                    self._scheduler_timer.cancel()
                self.is_scheduler_waiting = True
                self._scheduler_timer = threading.Timer(wait_duration, self._run_scheduler_callback)  # Use callback
                self._scheduler_timer.daemon = True
                self._scheduler_timer.start()
                print(f"  Scheduler: Waiting {wait_duration:.1f}s (of {delay_needed:.1f}s total) to start slew for {target_to_consider['icao']}.")
                return

            # --- Start Tracking ---
            if self._scheduler_timer and self._scheduler_timer.is_alive():
                self._scheduler_timer.cancel()
                self._scheduler_timer = None
            self.is_scheduler_waiting = False

            icao_to_track = target_to_consider['icao']
            latest_data = read_aircraft_data().get(icao_to_track)
            if not latest_data:
                print(f"  Scheduler: No current data for {icao_to_track} just before starting track; deferring.")
                if not self._scheduler_timer or not self._scheduler_timer.is_alive():
                    self.is_scheduler_waiting = True
                    self._scheduler_timer = threading.Timer(2.0, self._run_scheduler_callback)  # Retry in 2s, use callback
                    self._scheduler_timer.daemon = True
                    self._scheduler_timer.start()
                return

            self.current_track_icao = icao_to_track
            self.current_track_ev = target_to_consider['ev']
            self.shutdown_event.clear()

            # Reset captures_taken for the new track.  Because we're already under track_lock at
            # the start of this function, we do not acquire it again here.
            self.captures_taken = 0

            write_status({"mode": "acquiring", "icao": icao_to_track})
            print(f"  Scheduler: Starting track thread for {icao_to_track} (EV: {target_to_consider['ev']:.1f})")

            start_state_info = target_to_consider['start_state'].copy()
            start_state_info['slew_duration_s'] = intercept_duration

            self.active_thread = threading.Thread(
                target=self.track_aircraft,
                args=(icao_to_track, latest_data, start_state_info),
                daemon=True
            )
            self.active_thread.start()

    def _run_scheduler_callback(self):
        """Callback function used by the scheduler's timer."""
        # If shutdown has been signaled, do not invoke the scheduler again
        self.is_scheduler_waiting = False  # Clear flag before running scheduler again
        if self.shutdown_event.is_set():
            return
        # Now call the actual scheduler
        self._run_scheduler()


    def track_aircraft(self, icao: str, aircraft_data: dict, start_state: dict):
        """The main tracking logic thread."""
        try:
            if self.shutdown_event.is_set():
                print(f"  Track [{icao}]: Aborted immediately before slew.")
                return

            target_az = start_state.get('az')
            target_el = start_state.get('el')
            if target_az is None or target_el is None:
                print(f"  Track [{icao}]: Invalid start state coordinates. Aborting.")
                return

            target_az_el = (target_az, target_el)
            write_status({"mode": "slewing", "target_az_el": list(target_az_el), "icao": icao})
            print(f"  Track [{icao}]: Slewing to intercept point ({target_az:.2f}, {target_el:.2f})...")

            def _slew_progress(cur_az: float, cur_el: float, state: str):
                live_target_az_el = self._predict_target_az_el(aircraft_data, when=time.time()) or target_az_el
                payload = {
                    "mode": "slewing", "icao": icao,
                    "mount_az_el": [float(cur_az), float(cur_el)],
                    "target_az_el": list(live_target_az_el),
                }
                write_status(payload)

            if not self.indi_controller.slew_to_az_el(target_az, target_el, progress_cb=_slew_progress):
                print(f"  Track [{icao}]: Initial slew failed or aborted. Ending track.")
                current_status = {"mode": "idle", "icao": None}
                if self.manual_target_icao == icao:
                    current_status["manual_target"] = {
                        "icao": icao,
                        "viability": self.manual_viability_info or {"viable": False, "reasons": ["slew failed"]},
                    }
                else:
                    current_status["manual_target"] = None
                write_status(current_status)
                return

            if self.shutdown_event.is_set():
                print(f"  Track [{icao}]: Aborted after slew completion.")
                return

            current_alt_ft = aircraft_data.get('alt')
            write_status({"mode": "focusing", "autofocus_alt_ft": current_alt_ft, "icao": icao})
            print(f"  Track [{icao}]: Performing autofocus...")
            if not self.indi_controller.autofocus(current_alt_ft):
                print(f"  Track [{icao}]: Autofocus failed. Ending track.")
                write_status({"mode": "idle", "icao": None, "error_message": "Autofocus failed"})
                return

            if self.shutdown_event.is_set():
                print(f"  Track [{icao}]: Aborted after focus completion.")
                return

            print(f"  Track [{icao}]: Starting optical guiding loop.")
            guide_cfg = CONFIG['capture'].get('guiding', {})
            calib_cfg = CONFIG.get('pointing_calibration', {})
            cam_specs = CONFIG.get('camera_specs', {})
            frame_w = cam_specs.get('resolution_width_px', 640)
            frame_h = cam_specs.get('resolution_height_px', 480)
            frame_center = (frame_w / 2.0, frame_h / 2.0)
            frame = get_altaz_frame(self.observer_loc)
            plate_scale = calculate_plate_scale()

            simulated_detection_result = None
            if CONFIG['development']['dry_run']:
                slew_duration_s = start_state.get('slew_duration_s', 30.0)
                timing_error_s = abs(start_state.get('time', time.time()) - (time.time() + slew_duration_s)) * 0.5
                if timing_error_s > 0:
                    # This logic is moved from image_analyzer._detect_aircraft_from_data
                    # to avoid calling it with a null path. We can create a synthetic
                    # detection result without needing to load a fake image first.
                    error_px = float(timing_error_s) * 20.0
                    max_offset = 150.0
                    dx = min(error_px * np.random.choice([-1, 1]), max_offset * np.sign(error_px))
                    dy = 0
                    cx = frame_center[0] + dx
                    cy = frame_center[1] + dy
                    cx = max(0, min(frame_w - 1, cx))
                    cy = max(0, min(frame_h - 1, cy))
                    simulated_detection_result = {
                        'detected': True,
                        'center_px': (cx, cy),
                        'confidence': 0.95,
                        'sharpness': 100.0  # Use a plausible dummy sharpness
                    }

            # Initialize the local capture counter from the shared state.
            with self.track_lock:
                captures_taken = self.captures_taken
            consecutive_losses = 0
            iteration = 0
            last_seq_ts = 0.0
            max_losses = int(guide_cfg.get('max_consecutive_losses', 5))
            deadzone_px = float(guide_cfg.get('deadzone_px', 5.0))
            settle_s = float(guide_cfg.get('settle_time_s', 0.5))
            save_guide_png = guide_cfg.get('save_guide_png', True)
            min_sequence_interval_s = float(CONFIG['capture'].get('min_sequence_interval_s', 1.0))
            min_quality_threshold = float(CONFIG['selection'].get('track_quality_min', 0.2))

            # --- Main Guide Loop ---
            while True:
                try:
                    latest_full_dict = read_aircraft_data()
                    if icao in latest_full_dict:
                        latest_data = latest_full_dict[icao]
                        update_keys = ['lat', 'lon', 'gs', 'track', 'alt', 'vert_rate', 'timestamp', 'age_s']
                        for k in update_keys:
                            if k in latest_data:
                                aircraft_data[k] = latest_data[k]
                    else:
                        print(f"  Track [{icao}]: Target lost from ADS-B feed. Ending track.")
                        break
                    current_age = aircraft_data.get('age_s', float('inf'))
                    if current_age > 60.0:
                        print(f"  Track [{icao}]: ADS-B data is too old ({current_age:.1f}s > 60s). Aborting track.")
                        break
                except Exception as e:
                    print(f"  Track [{icao}]: Error refreshing ADS-B data: {e}. Continuing with previous data.")

                iteration += 1
                loop_start_time = time.time()

                if self.shutdown_event.is_set():
                    print(f"  Track [{icao}]: Shutdown signaled, exiting guide loop.")
                    break

                guide_path = None
                guide_data = None
                # In dry run, we still need to "create" the guide image file path and data
                # so that it can be loaded for analysis (sharpness, etc.)
                if CONFIG['development']['dry_run']:
                    try:
                        # Call snap_image, which creates the flat+text FITS
                        guide_path = self.indi_controller.snap_image(f"{icao}_g{iteration}")
                        if guide_path and os.path.exists(guide_path):
                            # Load the FITS file we just created
                            guide_data = _load_fits_data(guide_path)
                        else:
                            # Fallback if snap_image fails or file not found
                            print(f"  Track [{icao}]: [DRY RUN] snap_image failed or file not found.")
                            sim_h = CONFIG.get('camera_specs', {}).get('resolution_height_px', 480)
                            sim_w = CONFIG.get('camera_specs', {}).get('resolution_width_px', 640)
                            guide_data = np.full((sim_h, sim_w), 1000, np.float32)  # Simple flat fallback
                        if guide_data is None:  # Final fallback
                            guide_data = np.full((480, 640), 1000, np.float32)
                    except Exception as e:
                        print(f"  Track [{icao}]: [DRY RUN] Error simulating guide snap: {e}")
                        sim_h = CONFIG.get('camera_specs', {}).get('resolution_height_px', 480)
                        sim_w = CONFIG.get('camera_specs', {}).get('resolution_width_px', 640)
                        guide_data = np.full((sim_h, sim_w), 1000, np.float32)
                else:
                    # --- Real Hardware Capture ---
                    try:
                        guide_path = self.indi_controller.snap_image(f"{icao}_g{iteration}")
                        if not guide_path:
                            raise RuntimeError("snap_image returned None or empty path")
                    except Exception as e:
                        print(f"  Track [{icao}]: Guide frame {iteration} capture failed: {e}")
                        consecutive_losses += 1
                        if consecutive_losses >= max_losses: break
                        time.sleep(1.0);
                        continue
                    guide_data = _load_fits_data(guide_path)
                    if guide_data is None:
                        print(f"  Track [{icao}]: Failed to load guide image data for {guide_path}. Skipping frame.")
                        consecutive_losses += 1
                        if consecutive_losses >= max_losses: break
                        time.sleep(0.5);
                        continue

                if guide_data is None:  # Handle case where guide_data failed to load/simulate
                    print(f"  Track [{icao}]: guide_data is None. Skipping frame.")
                    consecutive_losses += 1
                    if consecutive_losses >= max_losses: break
                    time.sleep(0.5);
                    continue

                guide_shape = guide_data.shape

                guide_png_url = ""
                guide_png_path_generated = ""
                if save_guide_png and guide_path:
                    try:
                        guide_png_path_expected = os.path.splitext(guide_path)[0] + ".png"
                        saved_png_path = _save_png_preview_from_data(guide_data, guide_png_path_expected)
                        if saved_png_path and os.path.exists(saved_png_path):
                            guide_png_path_generated = saved_png_path
                            guide_png_url = _to_img_url(guide_png_path_generated)
                        elif not CONFIG['development']['dry_run']:
                            print(f"  Track [{icao}]: Warning - _save_png_preview_from_data did not return valid path.")
                    except Exception as e:
                        print(f"  Track [{icao}]: Warning - Error occurred during PNG creation/check: {e}")

                # Use the simulated result on the first iteration, then do real detection
                if iteration == 1 and simulated_detection_result:
                    detection = simulated_detection_result
                else:
                    detection = _detect_aircraft_from_data(guide_data, original_shape=guide_shape)

                if not detection or not detection.get('detected') or not detection.get('center_px'):
                    consecutive_losses += 1
                    reason = (detection or {}).get('reason', 'unknown');
                    sharp = detection.get('sharpness', -1);
                    conf = detection.get('confidence', -1)
                    print(f"  Track [{icao}]: Guide frame {iteration}: Target lost ({consecutive_losses}/{max_losses}). Reason: {reason} (Sharp: {sharp:.1f}, Conf: {conf:.2f})")
                    write_status({"mode": "tracking", "icao": icao, "iteration": iteration, "stable": False, "last_guide_png": guide_png_url, "guide_offset_px": None})
                    if consecutive_losses >= max_losses: break
                    time.sleep(1.0);
                    continue

                consecutive_losses = 0
                center_px = detection['center_px']
                if not (len(center_px) == 2 and math.isfinite(center_px[0]) and math.isfinite(center_px[1])):
                    print(f"  Track [{icao}]: Invalid centroid coordinates received: {center_px}. Treating as lost.")
                    consecutive_losses += 1
                    write_status({"mode": "tracking", "icao": icao, "iteration": iteration, "stable": False, "last_guide_png": guide_png_url})
                    time.sleep(0.5);
                    continue

                offset_px_x = center_px[0] - frame_center[0]
                offset_px_y = center_px[1] - frame_center[1]
                rotation_rad = math.radians(calib_cfg.get('rotation_angle_deg', 0.0))
                if abs(rotation_rad) > 1e-3:
                    cos_rot, sin_rot = math.cos(rotation_rad), math.sin(rotation_rad)
                    corrected_dx = offset_px_x * cos_rot - offset_px_y * sin_rot
                    corrected_dy = offset_px_x * sin_rot + offset_px_y * cos_rot
                else:
                    corrected_dx = offset_px_x;
                    corrected_dy = offset_px_y

                guide_error_mag = max(abs(corrected_dx), abs(corrected_dy))
                is_stable_np = guide_error_mag <= deadzone_px
                is_stable = bool(is_stable_np)

                # In dry run, we can now simulate the mount correcting the error
                if CONFIG['development']['dry_run']:
                    # Simulate the mount correcting the error over time
                    new_offset_x = offset_px_x * 0.3 + (np.random.rand() - 0.5) * 2.0
                    new_offset_y = offset_px_y * 0.3 + (np.random.rand() - 0.5) * 2.0
                    # Update the 'simulated_detection_result' for the *next* iteration
                    simulated_detection_result = detection.copy()
                    simulated_detection_result['center_px'] = (frame_center[0] + new_offset_x, frame_center[1] + new_offset_y)
                    # The first frame might not be stable, but subsequent ones should be
                    if iteration > 1:
                        is_stable = True

                now = time.time()
                try:
                    sun_az, sun_el = get_sun_azel(now, self.observer_loc)
                    est_list = estimate_positions_at_times(aircraft_data, [now, now + 1.0])
                    if len(est_list) < 2: print(f"  Track [{icao}]: Could not predict motion for quality check. Ending track."); break
                    current_pos_est, next_sec_pos_est = est_list[0], est_list[1]
                    current_az, current_el = latlonalt_to_azel(current_pos_est['est_lat'], current_pos_est['est_lon'], current_pos_est['est_alt'], now, self.observer_loc)
                    next_sec_az, next_sec_el = latlonalt_to_azel(next_sec_pos_est['est_lat'], next_sec_pos_est['est_lon'], next_sec_pos_est['est_alt'], now + 1.0, self.observer_loc)
                    ang_speed = angular_speed_deg_s((current_az, current_el), (next_sec_az, next_sec_el), 1.0, frame)
                    current_range_km = distance_km(self.observer_loc.lat.deg, self.observer_loc.lon.deg, current_pos_est['est_lat'], current_pos_est['est_lon'])
                    current_sun_sep = angular_sep_deg((current_az, current_el), (sun_az, sun_el), frame)
                    current_quality_state = {'el': current_el, 'sun_sep': current_sun_sep, 'range_km': current_range_km, 'ang_speed': ang_speed}
                    current_quality = calculate_quality(current_quality_state)
                    if current_quality < min_quality_threshold:
                        print(f"  Track [{icao}]: Target quality ({current_quality:.2f}) dropped below threshold ({min_quality_threshold}). Ending track.")
                        break
                except Exception as e:
                    print(f"  Track [{icao}]: Error during live quality check: {e}. Ending track for safety.")
                    break

                hardware_status = self.indi_controller.get_hardware_status() or {}
                status_payload = {
                    "mode": "tracking", "icao": icao, "iteration": iteration,
                    "guide_offset_px": {"dx": float(corrected_dx), "dy": float(corrected_dy)},
                    "stable": is_stable, "last_guide_png": guide_png_url,
                    "last_guide_file": os.path.basename(guide_path) if guide_path else "N/A",
                    "target_details": {"lat": current_pos_est.get('est_lat'), "lon": current_pos_est.get('est_lon'), "alt": current_pos_est.get('est_alt'),
                                       "flight": aircraft_data.get('flight', '').strip() or 'N/A', "gs": aircraft_data.get('gs'), "track": aircraft_data.get('track')},
                    "sun_pos": {"az": sun_az, "el": sun_el},
                    "sun_sep_target_deg": current_sun_sep,
                    "sun_sep_mount_deg": angular_sep_deg(hardware_status.get("mount_az_el", (0, 0)), (sun_az, sun_el), frame),
                    "image_vitals": {"sharpness": detection.get('sharpness'), "confidence": detection.get('confidence')},
                    "current_quality": round(current_quality, 3),
                    "current_range_km": round(current_range_km, 1) if np.isfinite(current_range_km) else None,
                    "captures_taken": captures_taken
                }
                status_payload.update(hardware_status)
                write_status(status_payload)

                pulse_sent = False
                az_sign = calib_cfg.get('az_offset_sign', 1)
                el_sign = calib_cfg.get('el_offset_sign', 1)

                guide_params = CONFIG['capture'].get('guiding', {})
                gain = float(guide_params.get('proportional_gain', 20.0))
                min_p = int(guide_params.get('min_pulse_ms', 10))
                max_p = int(guide_params.get('max_pulse_ms', 500))

                def calculate_pulse(error_px):
                    duration = int(abs(error_px) * gain)
                    return max(min_p, min(max_p, duration))


                pulse_ms_x, pulse_ms_y = 0, 0  # Initialize pulse durations
                if corrected_dy > deadzone_px:
                    pulse_ms_y = calculate_pulse(corrected_dy)
                    self.indi_controller.guide_pulse('N' if el_sign > 0 else 'S', pulse_ms_y)
                    pulse_sent = True
                elif corrected_dy < -deadzone_px:
                    pulse_ms_y = calculate_pulse(corrected_dy)
                    self.indi_controller.guide_pulse('S' if el_sign > 0 else 'N', pulse_ms_y)
                    pulse_sent = True
                if corrected_dx > deadzone_px:
                    pulse_ms_x = calculate_pulse(corrected_dx)
                    self.indi_controller.guide_pulse('E' if az_sign > 0 else 'W', pulse_ms_x)
                    pulse_sent = True
                elif corrected_dx < -deadzone_px:
                    pulse_ms_x = calculate_pulse(corrected_dx)
                    self.indi_controller.guide_pulse('W' if az_sign > 0 else 'E', pulse_ms_x)
                    pulse_sent = True

                if pulse_sent:
                    # if pulse_ms_x > 0 or pulse_ms_y > 0: print(f"    - Guide Pulses: X={pulse_ms_x}ms, Y={pulse_ms_y}ms")
                    time.sleep(settle_s)

                # This simulation logic is now handled inside the `is_stable` block for dry run
                # if CONFIG['development']['dry_run'] and simulated_detection_result:
                #     new_offset_x = offset_px_x * 0.3 + (np.random.rand() - 0.5) * 2.0
                #     new_offset_y = offset_px_y * 0.3 + (np.random.rand() - 0.5) * 2.0
                #     simulated_detection_result = detection.copy()
                #     simulated_detection_result['center_px'] = (frame_center[0] + new_offset_x, frame_center[1] + new_offset_y)

                now = time.time()
                if is_stable and (now - last_seq_ts) > min_sequence_interval_s:
                    # Before scheduling a new sequence, ensure we are not shutting down
                    if self.shutdown_event.is_set():
                        break
                    print(f"  Track [{icao}]: Guiding stable. Capturing sequence...")
                    capture_cfg = CONFIG['capture']
                    base_exposure = float(capture_cfg.get('sequence_exposure_s', 0.5))

                    # Estimate exposure adjustment using the guide frame data
                    # This now uses the "flat+text" image in dry run
                    exposure_factor = _estimate_exposure_adjustment_from_data(guide_data)
                    print(f"    - Auto-exposure factor: {exposure_factor:.2f}")

                    adj_min = float(capture_cfg.get('exposure_adjust_factor_min', 0.2))
                    adj_max = float(capture_cfg.get('exposure_adjust_factor_max', 5.0))
                    exposure_factor = max(adj_min, min(adj_max, exposure_factor))
                    recommended_exp = base_exposure * exposure_factor

                    blur_px_limit = float(capture_cfg.get('target_blur_px', 1.0))
                    t_max_blur = float('inf')
                    if ang_speed > 1e-3 and plate_scale > 1e-3:
                        t_max_blur = (blur_px_limit * plate_scale) / (ang_speed * 3600.0)

                    min_exp_limit = float(capture_cfg.get('exposure_min_s', 0.001))
                    max_exp_limit = float(capture_cfg.get('exposure_max_s', 5.0))

                    final_exp = recommended_exp;
                    limit_reason = "recommended"
                    if final_exp > t_max_blur: final_exp = t_max_blur; limit_reason = f"blur_limit ({t_max_blur:.3f}s)"
                    if final_exp > max_exp_limit:
                        if limit_reason == "recommended" or t_max_blur > max_exp_limit: limit_reason = f"config_max ({max_exp_limit:.3f}s)"
                        final_exp = max_exp_limit
                    if final_exp < min_exp_limit:
                        if limit_reason == "recommended" or t_max_blur < min_exp_limit: limit_reason = f"config_min ({min_exp_limit:.3f}s)"
                        final_exp = min_exp_limit

                    print(f"    - Final sequence exposure: {final_exp:.4f}s (Reason: {limit_reason})")

                    last_seq_ts = time.time() # Set timestamp immediately
                    
                    # Create the log and status payloads here, in the guide thread
                    seq_log = {'type': 'sequence', 'sequence_id': int(last_seq_ts * 1000), 'icao': icao,
                               'timestamp': last_seq_ts,
                               'per_frame_exposure_s': float(final_exp), 'exposure_limit_reason': limit_reason,
                               'guide_offset_px': (float(corrected_dx), float(corrected_dy)),
                               'plate_scale_arcsec_px': float(plate_scale),
                               'predicted_ang_speed_deg_s': float(ang_speed),
                               'target_blur_px_limit': float(blur_px_limit),
                               'last_guide_sharpness': detection.get('sharpness'),
                               'last_guide_confidence': detection.get('confidence'),
                               # n_frames and image_paths will be added by worker
                               }
                    
                    status_payload_base = {
                        "sequence_exposure_s": float(final_exp),
                        "sequence_meta": {"ang_speed_deg_s": float(ang_speed),
                                           "target_blur_px_limit": float(blur_px_limit),
                                           "exposure_limit_reason": limit_reason}
                    }

                    # Do not queue new sequences if shutdown has been signaled
                    if self.shutdown_event.is_set():
                        print(f"  Track [{icao}]: Shutdown signaled, skipping sequence scheduling.")
                        break
                    # Put the job on the queue for the worker
                    job = (icao, final_exp, seq_log, captures_taken, status_payload_base)
                    self.capture_queue.put(job)
                    
                    # Update captures_taken *immediately* in this thread
                    # The worker will add the *real* count later
                    with self.track_lock:
                        self.captures_taken += int(CONFIG['capture'].get('num_sequence_images', 5))
                        captures_taken = self.captures_taken
                    
                    # Write status immediately, worker will update it again
                    status_update = status_payload_base.copy()
                    status_update['captures_taken'] = captures_taken
                    status_update['sequence_count'] = 0 # Worker will set this
                    write_status(status_update) 

                loop_duration = time.time() - loop_start_time
                sleep_time = max(0.01, 0.1 - loop_duration)
                time.sleep(sleep_time)

        finally:
            print(f"  Track [{icao}]: Exiting tracking thread.")
            with self.track_lock:
                was_preempted = self.preempt_requested
                finished_icao = self.current_track_icao
                self.current_track_icao = None
                self.current_track_ev = 0.0
                self.active_thread = None
                self.preempt_requested = False

            was_aborted_by_signal = self.shutdown_event.is_set()

            final_status = {"mode": "idle", "icao": None}
            if was_aborted_by_signal and not was_preempted:
                print(f"  Tracking for {finished_icao} was interrupted.")
                if hasattr(self, 'manual_target_icao') and self.manual_target_icao == finished_icao:
                    final_status["manual_target"] = {"icao": finished_icao, "viability": {"viable": False, "reasons": ["track aborted"]}}
                else:
                    final_status["manual_target"] = None
            elif was_preempted:
                print(f"  Tracking for {finished_icao} preempted. Re-evaluating targets...")
                final_status["manual_target"] = None
            else:
                print(f"  Tracking for {finished_icao} finished normally. System idle.")
                final_status["manual_target"] = None

            write_status(final_status)

            if was_preempted or not was_aborted_by_signal:
                print("  Triggering scheduler to find next target...")
                if not self.is_scheduler_waiting:
                    self._run_scheduler()


if __name__ == "__main__":
    import logging
    import sys
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)

    ensure_log_dir()
    adsb_watch_file = os.path.normpath(os.path.abspath(CONFIG['adsb']['json_file_path']))
    adsb_watch_path = os.path.dirname(adsb_watch_file)
    if not os.path.exists(adsb_watch_path): exit(1)
    if not os.path.exists(adsb_watch_file): print(f"Warning: ADS-B file '{adsb_watch_file}' does not exist yet...")

    event_handler = FileHandler(adsb_watch_file)
    print("Performing initial read of aircraft data...")
    event_handler.process_file()

    # Check for and process command.json at startup
    command_filename = CONFIG['logging']['command_file']
    command_file_path = os.path.normpath(os.path.join(LOG_DIR, command_filename))
    if os.path.exists(command_file_path):
        print(f"Found existing command file '{command_file_path}' at startup. Processing...")
        try:
            with open(command_file_path, 'r') as f:
                command = json.load(f)
            if 'track_icao' in command:
                icao = command['track_icao'].lower().strip()
                print(f"  Manual override to track ICAO: {icao} from startup command.")
                event_handler.manual_target_icao = icao
                # Trigger scheduler to pick up the manual target
                event_handler._run_scheduler()
            os.remove(command_file_path)
            print("  Processed and removed startup command file.")
        except (json.JSONDecodeError, OSError) as e:
            print(f"  Error processing startup command file '{command_file_path}': {e}")
        except Exception as e:
            print(f"  Unexpected error during startup command processing: {e}")

    adsb_observer = Observer()
    adsb_observer.schedule(event_handler, path=adsb_watch_path, recursive=False)
    command_handler = CommandHandler(event_handler)
    command_observer = Observer()
    os.makedirs(LOG_DIR, exist_ok=True)
    command_observer.schedule(command_handler, path=LOG_DIR, recursive=False)

    try:
        adsb_observer.start();
        command_observer.start()
        print(f"Monitoring '{adsb_watch_path}' for ADS-B data...")
        print(f"Monitoring '{LOG_DIR}' for command file '{CONFIG['logging']['command_file']}'...")
        while True:
            if not adsb_observer.is_alive() or not command_observer.is_alive():
                print("Error: Watchdog observer thread died. Exiting.")
                event_handler.shutdown_event.set();
                break
            time.sleep(5)
    except KeyboardInterrupt:
        print("\nShutdown requested (Ctrl+C)...")
        # Signal all threads to stop and prevent new tracks from starting
        event_handler.shutdown_event.set()
        # Notify the stacking orchestrator not to accept any new jobs
        try:
            request_shutdown()
        except Exception:
            pass
        # Cancel any pending scheduler timer to avoid scheduling new track jobs
        if event_handler._scheduler_timer and event_handler._scheduler_timer.is_alive():
            event_handler._scheduler_timer.cancel()
            event_handler.is_scheduler_waiting = False
            print("Cancelled pending scheduler timer.")
        # Wait briefly for the active tracking thread to finish
        active_thread = event_handler.active_thread
        if active_thread and active_thread.is_alive():
            print("Waiting for tracking thread to finish...")
            active_thread.join(timeout=15.0)
            if active_thread.is_alive():
                print("Warning: Tracking thread did not exit cleanly.")
        # Shut down the stack orchestrator early to prevent new jobs from being scheduled
        try:
            shutdown_stack_orchestrator()
        except Exception as e:
            print(f"Warning: Error during stack orchestrator shutdown: {e}")
    except Exception as e:
        print(f"FATAL: Unhandled exception in main loop: {e}")
        event_handler.shutdown_event.set()
        import traceback;
        traceback.print_exc()
    finally:
        print("Stopping observers...")
        if adsb_observer.is_alive(): adsb_observer.stop()
        if command_observer.is_alive(): command_observer.stop()
        if event_handler.indi_controller:
            print("Disconnecting hardware...")
            try:
                event_handler.indi_controller.disconnect()
            except Exception as e:
                print(f"Warning: Error during hardware disconnect: {e}")
        try:
            if adsb_observer.is_alive(): adsb_observer.join(timeout=2.0)
            if command_observer.is_alive(): command_observer.join(timeout=2.0)
        except RuntimeError:
            pass
        print("Shutting down stack orchestrator...")
        try:
            shutdown_stack_orchestrator()
        except Exception as e:
            print(f"Warning: Error during stack orchestrator shutdown: {e}")
        print("Program terminated.")
