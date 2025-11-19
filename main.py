# main.py
"""
Main entry point: Monitors ADS-B data, processes flight paths, and tracks aircraft.
"""

import json
import logging
import math
import os
import queue
import threading
import time
import traceback
from typing import Callable, Optional

import astropy.units as u
import numpy as np
from astropy.coordinates import EarthLocation
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from aircraft_selector import (
    calculate_expected_value,
    evaluate_manual_target_viability,
    select_aircraft,
)
from config_loader import CONFIG, LOG_DIR
from coord_utils import (
    angular_sep_deg,
    angular_speed_deg_s,
    calculate_plate_scale,
    distance_km,
    get_altaz_frame,
    get_sun_azel,
    latlonalt_to_azel,
)
from data_reader import read_aircraft_data
from dead_reckoning import estimate_positions_at_times
from hardware_control import IndiController
from image_analyzer import (
    _detect_aircraft_from_data,
    _estimate_exposure_adjustment_from_data,
    _load_fits_data,
    _save_png_preview_from_data,
)
from logger_config import setup_logging
from stack_orchestrator import request_shutdown
from stack_orchestrator import shutdown as shutdown_stack_orchestrator
from status_writer import write_status
from storage import append_to_json, ensure_log_dir

logger = logging.getLogger(__name__)

# === NEW: Import the modular command classes ===
from commands.track import TrackCommand
from commands.abort import AbortCommand
from commands.park import ParkCommand
from commands.idle import IdleCommand
from commands.auto_track import AutoTrackCommand


def _to_img_url(fs_path: str) -> str:
    """Converts a filesystem image path to its corresponding dashboard URL."""
    return "/images/" + os.path.basename(str(fs_path)) if fs_path else ""


class CommandHandler(FileSystemEventHandler):
    """A simple handler to watch for manual override commands."""

    def __init__(self, main_handler):
        """
        Initializes the CommandHandler.

        Args:
            main_handler: The main FileHandler instance to interact with.
        """
        self.main_handler = main_handler
        command_filename = CONFIG['logging']['command_file']
        self.command_file = os.path.normpath(
            os.path.join(LOG_DIR, command_filename))

    def on_modified(self, event):
        """
        Called by the watchdog observer when a file is modified.

        If the modified file is the command file, this method triggers
        the command processing logic.

        Args:
            event: The file system event object.
        """
        if os.path.normpath(event.src_path) == self.command_file:
            self._process_command()

    def on_created(self, event):
        """
        Called by the watchdog observer when a file is created.

        If the created file is the command file, this method triggers
        the command processing logic.

        Args:
            event: The file system event object.
        """
        if os.path.normpath(event.src_path) == self.command_file:
            self._process_command()

    def on_moved(self, event):
        """
        Called by the watchdog observer when a file is moved or renamed.

        This handles atomic writes (e.g., `mv` or `os.replace`), which
        often trigger a `moved` event. If the destination is the command
        file, it triggers the command processing logic.

        Args:
            event: The file system event object.
        """
        # Watchdog triggers moved event even for atomic replace if dest exists
        if os.path.normpath(event.dest_path) == self.command_file:
            self._process_command()

    def _process_command(self):
        """Reads and processes the command file, handling race conditions."""
        logger.info("--- Manual command detected! ---")
        try:
            # === NEW: Clean, extensible command dispatcher ===
            with open(self.command_file, 'r') as f:
                data = json.load(f)

            # Atomic delete - fail loudly if gone (means another instance already processed it)
            os.remove(self.command_file)
            logger.info("Processed and removed command file.")

            # Command factory - infinitely extensible
            command_map = {
                'track_icao': TrackCommand,
                'abort_track': AbortCommand,
                'park_mount': ParkCommand,
                'idle_monitor': IdleCommand,
                'auto_track': AutoTrackCommand,
            }

            cmd_class = None
            if 'track_icao' in data:
                cmd_class = command_map['track_icao']
            elif data.get('command') in command_map:
                cmd_class = command_map[data.get('command')]

            if not cmd_class:
                logger.warning(f"Unknown command received: {data}")
                return

            success = cmd_class(self.main_handler).execute(data)
            logger.info(f"Command executed → {'SUCCESS' if success else 'FAILED'}")

        except FileNotFoundError:
            logger.info(
                "  Command file not found (likely already processed or deleted). Ignoring.")
        except json.JSONDecodeError as e:
            logger.warning(
                f"  Could not read command file (invalid JSON): {e}")
        except Exception as e:
            # Catch unexpected errors during command processing
            logger.error(f"  Unexpected error processing command: {e}")


class FileHandler(FileSystemEventHandler):
    """Handles file events and orchestrates the tracking process."""

    def __init__(self, watch_file):
        """
        Initializes the FileHandler.

        Args:
            watch_file (str): The path to the aircraft.json file to monitor.
        """
        self.watch_file = os.path.normpath(watch_file)
        self.last_process_time = 0
        self.debounce_seconds = 0.5  # Reduce debounce slightly?
        self.indi_controller = None
        self.current_track_icao = None
        self.current_track_ev = 0.0
        self.track_lock = threading.RLock()  # Protects access to tracking state vars
        obs_cfg = CONFIG['observer']
        # Ensure lat/lon/alt are floats
        try:
            lat = float(obs_cfg['latitude_deg'])
            lon = float(obs_cfg['longitude_deg'])
            alt = float(obs_cfg['altitude_m'])
            self.observer_loc = EarthLocation(
                lat=lat * u.deg, lon=lon * u.deg, height=alt * u.m)
        except (ValueError, KeyError) as e:
            logger.critical(f"FATAL: Invalid observer location in config: {e}")
            raise ValueError(f"Invalid observer location config: {e}") from e

        # Signals that the entire system is shutting down (Ctrl+C, fatal error, etc.)
        self.shutdown_event = threading.Event()
        # Signals that the *current* tracking thread should stop (manual preempt/abort)
        self.track_stop_event = threading.Event()
        self.active_thread = None  # Reference to the current tracking thread
        self.last_file_stat = None  # To detect redundant file events
        self.manual_target_icao = None  # ICAO requested manually
        # Cache last manual viability details for status
        self.manual_viability_info = None
        self.manual_override_active = None  # Keeps track of ongoing manual overrides
        self._scheduler_timer = None  # Timer for delayed scheduling
        self._scheduler_busy = False  # Prevents re-entrant scheduler runs
        self.preempt_requested = False  # Flag for preemptive track switching
        self.is_scheduler_waiting = False
        self.monitor_mode = False  # When True, system monitors but does not auto-track
        self.capture_queue = queue.Queue()
        self.capture_worker_thread = threading.Thread(
            target=self._capture_worker_loop, daemon=True)
        self.capture_worker_thread.start()

        # Shared counter for the number of images captured for the current track.
        # This counter is protected by track_lock and reset whenever a new track begins.
        self.captures_taken = 0


    # === NEW: Nuclear forced reset for zombie/hung tracking threads ===
    def _force_reset_tracking_state(self) -> None:
        """Forcibly clear tracking state when a thread refuses to die (hardware lockup, etc.)."""
        with self.track_lock:
            if self.current_track_icao:
                logger.warning(
                    f"!!! NUCLEAR FORCE RESET: Clearing zombie track state for {self.current_track_icao} !!!"
                )
                self.current_track_icao = None
                self.current_track_ev = 0.0
                self.active_thread = None
                self.preempt_requested = False
                # We leave track_stop_event set so any straggler codepaths will still exit


    def _capture_worker_loop(self):
        """
        Worker thread that runs blocking captures from the queue.
        This frees the main tracking loop from blocking I/O.
        """
        logger.info("  Capture worker thread started.")
        while not self.shutdown_event.is_set():
            try:
                # Wait for a job, but timeout to check shutdown_event
                job = self.capture_queue.get(timeout=1.0)
                if job is None:
                    continue

                (icao, final_exp, seq_log_base,
                 captures_taken_so_far, status_payload_base) = job

                captured_paths = self.indi_controller.capture_sequence(
                    icao, final_exp)

                if captured_paths:
                    # Update log and status *after* capture is complete
                    seq_log = seq_log_base.copy()
                    seq_log['n_frames'] = len(captured_paths)
                    seq_log['image_paths'] = captured_paths
                    append_to_json([seq_log], os.path.join(
                        LOG_DIR, 'captures.json'))

                    captures_taken = captures_taken_so_far + \
                        len(captured_paths)
                    status_update = status_payload_base.copy()
                    status_update["sequence_count"] = len(captured_paths)
                    status_update["captures_taken"] = captures_taken

                    if captured_paths:
                        status_update["last_capture_file"] = os.path.basename(
                            captured_paths[-1])
                        last_raw_png_path = os.path.splitext(
                            captured_paths[-1])[0] + ".png"
                        if os.path.exists(last_raw_png_path):
                            status_update["capture_png"] = _to_img_url(
                                last_raw_png_path)

                    # Update the shared captures_taken counter atomically.
                    with self.track_lock:
                        if self.current_track_icao == icao:
                            self.captures_taken = captures_taken
                    # Write status with updated capture count.
                    write_status(status_update)

            except queue.Empty:
                # This is normal, just loop and check shutdown_event
                continue
            except Exception:
                # Log errors from the capture worker
                logger.error("  Capture worker error:", exc_info=True)

        logger.info("  Capture worker thread shutting down.")

    def process_file(self):
        """
        Main orchestration logic triggered on `aircraft.json` updates.

        This method reads the latest aircraft data, initializes the hardware
        if necessary, selects the best tracking candidates, and manages the
        application's state machine (idle, tracking, etc.). It is responsible
        for deciding when to start a new tracking session via the scheduler.
        """
        # --- File Stat Check & Debounce ---
        try:
            current_stat = os.stat(self.watch_file)
            if self.last_file_stat and (current_stat.st_mtime == self.last_file_stat.st_mtime and
                                        current_stat.st_size == self.last_file_stat.st_size):
                return
            self.last_file_stat = current_stat
        except FileNotFoundError:
            return
        except OSError as e:
            logger.warning(
                f"Warning: Error stating file {self.watch_file}: {e}")
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
                logger.info("Initializing hardware controller...")
                self.indi_controller = IndiController()
                self.indi_controller.shutdown_event = self.track_stop_event
                logger.info("Hardware controller initialized.")
            except Exception as e:
                logger.critical(f"FATAL: Hardware initialization failed: {e}")
                write_status(
                    {"mode": "error", "error_message": f"Hardware init failed: {e}"})
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
            logger.error(f"Error during aircraft selection: {e}")
            candidates = []

        # Check for preemptive switch
        if self.current_track_icao and candidates and self.manual_override_active != self.current_track_icao:
            new_best_target = candidates[0]
            preempt_factor = float(
                CONFIG['selection'].get('preempt_factor', 1.25))
            if new_best_target['icao'] != self.current_track_icao and \
                    new_best_target['ev'] > (self.current_track_ev * preempt_factor):
                logger.info(
                    f"--- PREEMPTIVE SWITCH: New target {new_best_target['icao']} "
                    f"(EV {new_best_target['ev']:.1f}) > Current {self.current_track_icao} "
                    f"(EV {self.current_track_ev:.1f} * {preempt_factor:.2f}). ---"
                )
                with self.track_lock:
                    if self._scheduler_timer and self._scheduler_timer.is_alive():
                        self._scheduler_timer.cancel()
                        self._scheduler_timer = None
                        self.is_scheduler_waiting = False  # Reset flag (#6)
                    self.preempt_requested = True
                    self.track_stop_event.set()

        # --- Build Status Payload ---
        all_aircraft_list = []
        obs_lat = float(CONFIG['observer']['latitude_deg'])
        obs_lon = float(CONFIG['observer']['longitude_deg'])
        for icao, data in aircraft_dict.items():
            lat, lon = data.get('lat'), data.get('lon')
            if lat is None or lon is None:
                continue

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
        all_aircraft_list.sort(
            key=lambda x: x['dist_nm'] if x['dist_nm'] is not None else float('inf'))

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
        if self.current_track_icao:
            status_payload["mode"] = "tracking"
        elif self.monitor_mode:
            status_payload["mode"] = "monitor"
        else:
            status_payload["mode"] = "idle"
        status_payload["monitor_mode"] = self.monitor_mode

        if self.manual_target_icao:
            status_payload["manual_target"] = {
                "icao": self.manual_target_icao,
                "viability": self.manual_viability_info
            }

        # --- Write Status ---
        write_status(status_payload)

        # --- Run Scheduler (if not already tracking and not waiting) ---
        # Do not invoke the scheduler if a shutdown has been requested
        with self.track_lock:
            if not self.shutdown_event.is_set() and not self.current_track_icao and not self.is_scheduler_waiting:
                self._run_scheduler(candidates)

    def on_modified(self, event):
        """
        Called by the watchdog observer when a file is modified.

        If the modified file is the `aircraft.json` file, this method
        triggers the main processing logic.

        Args:
            event: The file system event object.
        """
        if os.path.normpath(event.src_path) == self.watch_file:
            self.process_file()

    def on_moved(self, event):
        """
        Called by the watchdog observer when a file is moved or renamed.

        This handles atomic writes to `aircraft.json` (e.g., from dump1090),
        which often trigger a `moved` event. If the destination is the
        watch file, it triggers the main processing logic.

        Args:
            event: The file system event object.
        """
        if os.path.normpath(event.dest_path) == self.watch_file:
            self.process_file()

    def _predict_target_az_el(self, aircraft_data: dict, when: Optional[float] = None) -> Optional[tuple[float, float]]:
        """
        Predicts the Az/El coordinates of an aircraft at a specific time.

        Args:
            aircraft_data: Dictionary containing the aircraft's state vectors.
            when: The Unix timestamp at which to predict the position.
                  Defaults to the current time.

        Returns:
            A tuple of (azimuth, elevation) in degrees, or None if prediction fails.
        """
        t = when or time.time()
        try:
            pos_list = estimate_positions_at_times(aircraft_data, [t])
            if not pos_list:
                return None
            pos = pos_list[0]
            if not all(k in pos for k in ['est_lat', 'est_lon', 'est_alt']):
                logger.warning(
                    "Warning: Prediction missing required keys (lat/lon/alt).")
                return None
            az, el = latlonalt_to_azel(
                pos['est_lat'], pos['est_lon'], pos['est_alt'], t, self.observer_loc)
            if not (math.isfinite(az) and math.isfinite(el)):
                logger.warning(
                    f"Warning: Non-finite az/el prediction ({az}, {el}).")
                return None
            return (float(az), float(el))
        except Exception as e:
            logger.error(f"Error predicting target az/el: {e}")
            return None

    def _run_scheduler(self, candidates: Optional[list] = None):
        """Decides whether to start a new track or wait."""
        with self.track_lock:
            if self._scheduler_busy:
                logger.debug("  Scheduler: Busy, skipping nested invocation.")
                return
            self._scheduler_busy = True
            try:
                # Do not schedule new tracks during shutdown
                if self.shutdown_event.is_set():
                    # Skip scheduling entirely if the system is shutting down
                    return
                if self.is_scheduler_waiting:
                    return

                # Check again inside the lock to avoid races
                if self.shutdown_event.is_set():
                    return
                if self.current_track_icao:
                    return
                is_manual_override = bool(self.manual_target_icao)
                if self.monitor_mode and not is_manual_override:
                    logger.debug("  Scheduler: Monitor mode active; skipping auto-tracking.")
                    return

                if not self.indi_controller:
                    logger.warning(
                        "  Scheduler: Cannot run, hardware controller not ready.")
                    return
                hs = self.indi_controller.get_hardware_status() or {}

                if candidates is None:
                    aircraft_dict = read_aircraft_data()
                    current_az_el = hs.get("mount_az_el", (0.0, 0.0))
                    try:
                        candidates = select_aircraft(
                            aircraft_dict, current_az_el) or []
                    except Exception as e:
                        logger.error(f"  Scheduler: Error selecting aircraft: {e}")
                        candidates = []
                else:
                    aircraft_dict = read_aircraft_data()

                target_to_consider = None
                is_manual_override = False

                if self.manual_target_icao:
                    icao = self.manual_target_icao
                    logger.info(f"  Scheduler: Evaluating manual target {icao}...")
                    manual_target_in_candidates = next(
                        (c for c in candidates if c['icao'] == icao), None)

                    if manual_target_in_candidates:
                        logger.info(
                            f"  Manual target {icao} is now viable (EV: {manual_target_in_candidates['ev']:.1f}). Selecting.")
                        target_to_consider = manual_target_in_candidates
                        is_manual_override = True
                        self.manual_viability_info = None
                        self.manual_override_active = icao
                        self.manual_target_icao = None
                    else:
                        logger.info(
                            f"  Manual target {icao} not in top candidates. Checking viability...")
                        ok, reasons, details = evaluate_manual_target_viability(
                            icao, aircraft_dict, observer_loc=self.observer_loc
                        )
                        ev_reason_text = None
                        ev_val = None
                        if ok and icao in aircraft_dict:
                            try:
                                current_az_el_ev = hs.get(
                                    "mount_az_el", (0.0, 0.0))  # Use hs here
                                ev_res = calculate_expected_value(
                                    current_az_el_ev, icao, aircraft_dict[icao])
                                ev_val = float(ev_res.get('ev', 0.0))
                                if ev_val <= 0.0:
                                    ev_reason = ev_res.get(
                                        'reason', 'unknown_ev_reason')
                                    reason_map = {
                                        'no_intercept': "cannot slew to intercept within limits",
                                        'late_intercept': "intercept occurs too late (outside horizon)",
                                        'low_quality': "predicted quality below minimum during track",
                                        'no_prediction': "cannot predict flight path",
                                        'prediction_failed': "prediction calculation error"
                                    }
                                    ev_reason_text = reason_map.get(
                                        ev_reason, f"EV too low ({ev_val:.2f})")
                                    if ev_reason_text not in reasons:
                                        reasons.append(ev_reason_text)
                            except Exception as e:
                                logger.warning(
                                    f"  Warning: EV calculation failed for manual target {icao}: {e}")
                                reasons.append("EV calculation failed")

                        self.manual_viability_info = {
                            "viable": bool(ok and ev_val is not None and ev_val > 0.0),
                            "reasons": reasons, "details": details, "ev": ev_val,
                        }
                        reason_str = "; ".join(
                            reasons) if reasons else "basic checks passed, but EV <= 0 or not calculated"
                        retry_s = float(CONFIG.get('selection', {}).get(
                            'manual_retry_interval_s', 5.0))
                        logger.info(
                            f"  Manual target {icao} rejected: {reason_str}. Retrying in {retry_s}s.")
                        write_status(
                            {"manual_target": {"icao": icao, "viability": self.manual_viability_info}})

                        if not self._scheduler_timer or not self._scheduler_timer.is_alive():
                            logger.info(
                                f"  Scheduling retry for manual target in {retry_s}s.")
                            self.is_scheduler_waiting = True
                            self._scheduler_timer = threading.Timer(
                                retry_s, self._run_scheduler_callback)  # Use callback
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
                    logger.warning(
                        f"  Scheduler: Target {target_to_consider['icao']} missing required timing data. Skipping.")
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
                    self._scheduler_timer = threading.Timer(
                        wait_duration, self._run_scheduler_callback)  # Use callback
                    self._scheduler_timer.daemon = True
                    self._scheduler_timer.start()
                    logger.info(
                        f"  Scheduler: Waiting {wait_duration:.1f}s (of {delay_needed:.1f}s total) to start slew for {target_to_consider['icao']}.")
                    return

                # --- Start Tracking ---
                if self._scheduler_timer and self._scheduler_timer.is_alive():
                    self._scheduler_timer.cancel()
                    self._scheduler_timer = None
                self.is_scheduler_waiting = False

                icao_to_track = target_to_consider['icao']
                latest_data = read_aircraft_data().get(icao_to_track)
                if not latest_data:
                    logger.info(
                        f"  Scheduler: No current data for {icao_to_track} just before starting track; deferring.")
                    if not self._scheduler_timer or not self._scheduler_timer.is_alive():
                        self.is_scheduler_waiting = True
                        self._scheduler_timer = threading.Timer(
                            2.0, self._run_scheduler_callback)  # Retry in 2s, use callback
                        self._scheduler_timer.daemon = True
                        self._scheduler_timer.start()
                    return

                self.current_track_icao = icao_to_track
                self.current_track_ev = target_to_consider['ev']
                self.track_stop_event.clear()
                if not is_manual_override:
                    self.manual_override_active = None

                # Reset captures_taken for the new track.  Because we're already under track_lock at
                # the start of this function, we do not acquire it again here.
                self.captures_taken = 0

                write_status({"mode": "acquiring", "icao": icao_to_track})
                
                # --- ENHANCED LOGGING ---
                ev_val = target_to_consider.get('ev', 0.0)
                details = target_to_consider.get('score_details', {})
                d_contrib = details.get('contrib_dist', 0.0)
                c_contrib = details.get('contrib_close', 0.0)
                i_contrib = details.get('contrib_int', 0.0)
                raw_dist = details.get('raw_range_km', 0.0)
                raw_rate = details.get('raw_closure_ms', 0.0)
                raw_slew = details.get('raw_int_s', 0.0)
                
                logger.info(
                    f"  Scheduler: Starting track for {icao_to_track} (Total EV: {ev_val:.3f})\n"
                    f"    > Score Breakdown: Dist: {d_contrib:.2f} + Close: {c_contrib:.2f} + Slew: {i_contrib:.2f}\n"
                    f"    > Flight Stats:    {raw_dist:.1f}km Range | {raw_rate:+.0f}m/s Rate | {raw_slew:.1f}s Slew"
                )

                start_state_info = target_to_consider['start_state'].copy()
                start_state_info['slew_duration_s'] = intercept_duration

                self.active_thread = threading.Thread(
                    target=self.track_aircraft,
                    args=(icao_to_track, latest_data, start_state_info, hs),
                    daemon=True
                )
                self.active_thread.start()
            finally:
                self._scheduler_busy = False

    def _run_scheduler_callback(self):
        """Callback function used by the scheduler's timer."""
        with self.track_lock:
            # If shutdown has been signaled, do not invoke the scheduler again
            self.is_scheduler_waiting = False  # Clear flag before running scheduler again
            if self.shutdown_event.is_set():
                self._scheduler_busy = False
                return
            # Now call the actual scheduler
            self._run_scheduler()

    def track_aircraft(self, icao: str, aircraft_data: dict, start_state: dict, hardware_status: dict):
        """
        Main tracking and guiding loop for a single aircraft.

        This method is executed in a dedicated thread for each tracking session.
        It performs the initial slew, autofocuses, and then enters a continuous
        loop of capturing guide frames, detecting the aircraft's position,
        sending guide pulses to the mount, and capturing science frames when
        the guiding is stable.

        Args:
            icao: The ICAO identifier of the target aircraft.
            aircraft_data: The most recent ADS-B data for the target.
            start_state: Information about the calculated intercept point,
                         including target Az/El and slew time.
            hardware_status: The hardware status at the start of the track.
        """
        try:
            if self.track_stop_event.is_set():
                logger.info(
                    f"  Track [{icao}]: Aborted immediately before slew.")
                return

            target_az = start_state.get('az')
            target_el = start_state.get('el')
            if target_az is None or target_el is None:
                logger.error(
                    f"  Track [{icao}]: Invalid start state coordinates. Aborting.")
                return

            target_az_el = (target_az, target_el)
            write_status(
                {"mode": "slewing", "target_az_el": list(target_az_el), "icao": icao})
            logger.info(
                f"  Track [{icao}]: Slewing to intercept point ({target_az:.2f}, {target_el:.2f})...")

            def _slew_progress(cur_az: float, cur_el: float, state: str):
                live_target_az_el = self._predict_target_az_el(
                    aircraft_data, when=time.time()) or target_az_el
                payload = {
                    "mode": "slewing", "icao": icao,
                    "mount_az_el": [float(cur_az), float(cur_el)],
                    "target_az_el": list(live_target_az_el),
                }
                write_status(payload)

            if not self.indi_controller.slew_to_az_el(target_az, target_el, progress_cb=_slew_progress):
                logger.error(
                    f"  Track [{icao}]: Initial slew failed or aborted. Ending track.")
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

            if self.track_stop_event.is_set():
                logger.info(
                    f"  Track [{icao}]: Aborted after slew completion.")
                return

            current_alt_ft = aircraft_data.get('alt')
            write_status(
                {"mode": "focusing", "autofocus_alt_ft": current_alt_ft, "icao": icao})
            logger.info(f"  Track [{icao}]: Performing autofocus...")
            if not self.indi_controller.autofocus(current_alt_ft):
                logger.error(
                    f"  Track [{icao}]: Autofocus failed. Ending track.")
                write_status({"mode": "idle", "icao": None,
                             "error_message": "Autofocus failed"})
                return

            if self.track_stop_event.is_set():
                logger.info(
                    f"  Track [{icao}]: Aborted after focus completion.")
                return

            logger.info(f"  Track [{icao}]: Starting optical guiding loop.")
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
                timing_error_s = abs(start_state.get(
                    'time', time.time()) - (time.time() + slew_duration_s)) * 0.5
                if timing_error_s > 0:
                    # This logic is moved from image_analyzer._detect_aircraft_from_data
                    # to avoid calling it with a null path. We can create a synthetic
                    # detection result without needing to load a fake image first.
                    error_px = float(timing_error_s) * 20.0
                    max_offset = 150.0
                    dx = min(
                        error_px * np.random.choice([-1, 1]), max_offset * np.sign(error_px))
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
            min_sequence_interval_s = float(
                CONFIG['capture'].get('min_sequence_interval_s', 1.0))

            # --- Main Guide Loop ---
            while True:
                try:
                    latest_full_dict = read_aircraft_data()
                    if icao in latest_full_dict:
                        latest_data = latest_full_dict[icao]
                        update_keys = ['lat', 'lon', 'gs', 'track',
                            'alt', 'vert_rate', 'timestamp', 'age_s']
                        for k in update_keys:
                            if k in latest_data:
                                aircraft_data[k] = latest_data[k]
                    else:
                        logger.info(
                            f"  Track [{icao}]: Target lost from ADS-B feed. Ending track.")
                        break
                    current_age = aircraft_data.get('age_s', float('inf'))
                    if current_age > 60.0:
                        logger.warning(
                            f"  Track [{icao}]: ADS-B data is too old ({current_age:.1f}s > 60s). Aborting track.")
                        break
                except Exception as e:
                    logger.warning(
                        f"  Track [{icao}]: Error refreshing ADS-B data: {e}. Continuing with previous data.")

                iteration += 1
                loop_start_time = time.time()

                if self.track_stop_event.is_set():
                    logger.info(
                        f"  Track [{icao}]: Shutdown signaled, exiting guide loop.")
                    break

                guide_path = None
                guide_data = None
                # In dry run, we still need to "create" the guide image file path and data
                # so that it can be loaded for analysis (sharpness, etc.)
                if CONFIG['development']['dry_run']:
                    try:
                        # Call snap_image, which creates the flat+text FITS
                        guide_path = self.indi_controller.snap_image(
                            f"{icao}_g{iteration}")
                        if guide_path and os.path.exists(guide_path):
                            # Load the FITS file we just created
                            guide_data = _load_fits_data(guide_path)
                        else:
                            # Fallback if snap_image fails or file not found
                            logger.warning(
                                f"  Track [{icao}]: [DRY RUN] snap_image failed or file not found.")
                            sim_h = CONFIG.get('camera_specs', {}).get(
                                'resolution_height_px', 480)
                            sim_w = CONFIG.get('camera_specs', {}).get(
                                'resolution_width_px', 640)
                            # Simple flat fallback
                            guide_data = np.full(
                                (sim_h, sim_w), 1000, np.float32)
                        if guide_data is None:  # Final fallback
                            guide_data = np.full((480, 640), 1000, np.float32)
                    except Exception as e:
                        logger.error(
                            f"  Track [{icao}]: [DRY RUN] Error simulating guide snap: {e}")
                        sim_h = CONFIG.get('camera_specs', {}).get(
                            'resolution_height_px', 480)
                        sim_w = CONFIG.get('camera_specs', {}).get(
                            'resolution_width_px', 640)
                        guide_data = np.full((sim_h, sim_w), 1000, np.float32)
                else:
                    # --- Real Hardware Capture ---
                    try:
                        guide_path = self.indi_controller.snap_image(
                            f"{icao}_g{iteration}")
                        if not guide_path:
                            raise RuntimeError(
                                "snap_image returned None or empty path")
                    except Exception as e:
                        logger.error(
                            f"  Track [{icao}]: Guide frame {iteration} capture failed: {e}")
                        consecutive_losses += 1
                        if consecutive_losses >= max_losses:
                            break
                        time.sleep(1.0)
                        continue
                    guide_data = _load_fits_data(guide_path)
                    if guide_data is None:
                        logger.warning(
                            f"  Track [{icao}]: Failed to load guide image data for {guide_path}. Skipping frame.")
                        consecutive_losses += 1
                        if consecutive_losses >= max_losses:
                            break
                        time.sleep(0.5)
                        continue

                if guide_data is None:  # Handle case where guide_data failed to load/simulate
                    logger.warning(
                        f"  Track [{icao}]: guide_data is None. Skipping frame.")
                    consecutive_losses += 1
                    if consecutive_losses >= max_losses:
                        break
                    time.sleep(0.5)
                    continue

                guide_shape = guide_data.shape

                guide_png_url = ""
                guide_png_path_generated = ""
                if save_guide_png and guide_path:
                    try:
                        guide_png_path_expected = os.path.splitext(guide_path)[
                                                                   0] + ".png"
                        saved_png_path = _save_png_preview_from_data(
                            guide_data, guide_png_path_expected)
                        if saved_png_path and os.path.exists(saved_png_path):
                            guide_png_path_generated = saved_png_path
                            guide_png_url = _to_img_url(
                                guide_png_path_generated)
                        elif not CONFIG['development']['dry_run']:
                            logger.warning(
                                f"  Track [{icao}]: Warning - _save_png_preview_from_data did not return valid path.")
                    except Exception as e:
                        logger.warning(
                            f"  Track [{icao}]: Warning - Error occurred during PNG creation/check: {e}")

                # Use the simulated result on the first iteration, then do real detection
                if iteration == 1 and simulated_detection_result:
                    detection = simulated_detection_result
                else:
                    detection = _detect_aircraft_from_data(
                        guide_data, original_shape=guide_shape)

                if not detection or not detection.get('detected') or not detection.get('center_px'):
                    consecutive_losses += 1
                    reason = (detection or {}).get('reason', 'unknown')
                    sharp = detection.get('sharpness', -1)
                    conf = detection.get('confidence', -1)
                    logger.info(
                        f"  Track [{icao}]: Guide frame {iteration}: Target lost ({consecutive_losses}/{max_losses}). Reason: {reason} (Sharp: {sharp:.1f}, Conf: {conf:.2f})")
                    write_status({"mode": "tracking", "icao": icao, "iteration": iteration,
                                 "stable": False, "last_guide_png": guide_png_url, "guide_offset_px": None})
                    if consecutive_losses >= max_losses:
                        break
                    time.sleep(1.0);
                    continue

                consecutive_losses = 0
                center_px = detection['center_px']
                if not (len(center_px) == 2 and math.isfinite(center_px[0]) and math.isfinite(center_px[1])):
                    logger.warning(
                        f"  Track [{icao}]: Invalid centroid coordinates received: {center_px}. Treating as lost.")
                    consecutive_losses += 1
                    write_status({"mode": "tracking", "icao": icao, "iteration": iteration,
                                 "stable": False, "last_guide_png": guide_png_url})
                    time.sleep(0.5)
                    continue

                offset_px_x = center_px[0] - frame_center[0]
                offset_px_y = center_px[1] - frame_center[1]
                rotation_rad = math.radians(
                    calib_cfg.get('rotation_angle_deg', 0.0))
                if abs(rotation_rad) > 1e-3:
                    cos_rot, sin_rot = math.cos(
                        rotation_rad), math.sin(rotation_rad)
                    corrected_dx = offset_px_x * cos_rot - offset_px_y * sin_rot
                    corrected_dy = offset_px_x * sin_rot + offset_px_y * cos_rot
                else:
                    corrected_dx = offset_px_x
                    corrected_dy = offset_px_y

                guide_error_mag = max(abs(corrected_dx), abs(corrected_dy))
                is_stable_np = guide_error_mag <= deadzone_px
                is_stable = bool(is_stable_np)

                # In dry run, we can now simulate the mount correcting the error
                if CONFIG['development']['dry_run']:
                    # Simulate the mount correcting the error over time
                    new_offset_x = offset_px_x * 0.3 + \
                        (np.random.rand() - 0.5) * 2.0
                    new_offset_y = offset_px_y * 0.3 + \
                        (np.random.rand() - 0.5) * 2.0
                    # Update the 'simulated_detection_result' for the *next* iteration
                    simulated_detection_result = detection.copy()
                    simulated_detection_result['center_px'] = (
                        frame_center[0] + new_offset_x, frame_center[1] + new_offset_y)
                    # The first frame might not be stable, but subsequent ones should be
                    if iteration > 1:
                        is_stable = True

                now = time.time()
                try:
                    sun_az, sun_el = get_sun_azel(now, self.observer_loc)
                    est_list = estimate_positions_at_times(
                        aircraft_data, [now, now + 1.0])
                    if len(est_list) < 2:
                        logger.error(
                            f"  Track [{icao}]: Could not predict motion for quality check. Ending track.")
                        break
                    current_pos_est, next_sec_pos_est = est_list[0], est_list[1]
                    current_az, current_el = latlonalt_to_azel(
                        current_pos_est['est_lat'], current_pos_est['est_lon'], current_pos_est['est_alt'], now, self.observer_loc)
                    next_sec_az, next_sec_el = latlonalt_to_azel(
                        next_sec_pos_est['est_lat'], next_sec_pos_est['est_lon'], next_sec_pos_est['est_alt'], now + 1.0, self.observer_loc)
                    ang_speed = angular_speed_deg_s(
                        (current_az, current_el), (next_sec_az, next_sec_el), 1.0, frame)
                    # === NEW HARD SAFETY FILTERS (matches the hybrid selector exactly) ===
                    current_range_km = distance_km(
                        self.observer_loc.lat.deg, self.observer_loc.lon.deg,
                        current_pos_est['est_lat'], current_pos_est['est_lon'])
                    current_sun_sep = angular_sep_deg(
                        (current_az, current_el), (sun_az, sun_el), frame)

                    sel_cfg = CONFIG['selection']
                    if (current_el < float(sel_cfg.get('min_elevation_deg', 5.0)) or
                        current_range_km > float(sel_cfg.get('max_range_km', 100.0)) or
                        current_sun_sep < float(sel_cfg.get('min_sun_separation_deg', 15.0))):
                        logger.info(
                            f" Track [{icao}]: Safety filter violated during track "
                            f"(el={current_el:.1f}° range={current_range_km:.1f}km sun_sep={current_sun_sep:.1f}°). Ending track.")
                        break
                except Exception as e:
                    logger.error(f" Track [{icao}]: Error during live safety checks / prediction: {e}. Ending track.", exc_info=True)
                    break

                status_payload = {
                    "mode": "tracking",
                    "icao": icao,
                    "iteration": iteration,
                    "guide_offset_px": {"dx": float(corrected_dx), "dy": float(corrected_dy)},
                    "stable": is_stable,
                    "last_guide_png": guide_png_url,
                    "last_guide_file": os.path.basename(guide_path) if guide_path else "N/A",
                    "target_details": {
                        "lat": current_pos_est.get('est_lat'),
                        "lon": current_pos_est.get('est_lon'),
                        "alt": current_pos_est.get('est_alt'),
                        "flight": aircraft_data.get('flight', '').strip() or 'N/A',
                        "gs": aircraft_data.get('gs'),
                        "track": aircraft_data.get('track'),
                    },
                    "sun_pos": {"az": sun_az, "el": sun_el},
                    "sun_sep_target_deg": current_sun_sep,
                    "sun_sep_mount_deg": angular_sep_deg(
                        hardware_status.get(
                            "mount_az_el", (0, 0)), (sun_az, sun_el), frame
                    ),
                    "image_vitals": {
                        "sharpness": detection.get('sharpness'),
                        "confidence": detection.get('confidence'),
                    },
                    "current_range_km": round(current_range_km, 1) if np.isfinite(current_range_km) else None,
                    "captures_taken": captures_taken,
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
                    self.indi_controller.guide_pulse(
                        'N' if el_sign > 0 else 'S', pulse_ms_y)
                    pulse_sent = True
                elif corrected_dy < -deadzone_px:
                    pulse_ms_y = calculate_pulse(corrected_dy)
                    self.indi_controller.guide_pulse(
                        'S' if el_sign > 0 else 'N', pulse_ms_y)
                    pulse_sent = True
                if corrected_dx > deadzone_px:
                    pulse_ms_x = calculate_pulse(corrected_dx)
                    self.indi_controller.guide_pulse(
                        'E' if az_sign > 0 else 'W', pulse_ms_x)
                    pulse_sent = True
                elif corrected_dx < -deadzone_px:
                    pulse_ms_x = calculate_pulse(corrected_dx)
                    self.indi_controller.guide_pulse(
                        'W' if az_sign > 0 else 'E', pulse_ms_x)
                    pulse_sent = True

                if pulse_sent:
                    # if pulse_ms_x > 0 or pulse_ms_y > 0: logger.info(f"    - Guide Pulses: X={pulse_ms_x}ms, Y={pulse_ms_y}ms")
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
                    if self.track_stop_event.is_set():
                        break
                    logger.info(
                        f"  Track [{icao}]: Guiding stable. Capturing sequence...")
                    capture_cfg = CONFIG['capture']
                    base_exposure = float(
                        capture_cfg.get('sequence_exposure_s', 0.5))

                    # Estimate exposure adjustment using the guide frame data
                    # This now uses the "flat+text" image in dry run
                    exposure_factor = _estimate_exposure_adjustment_from_data(
                        guide_data)
                    logger.info(
                        f"    - Auto-exposure factor: {exposure_factor:.2f}")

                    adj_min = float(capture_cfg.get(
                        'exposure_adjust_factor_min', 0.2))
                    adj_max = float(capture_cfg.get(
                        'exposure_adjust_factor_max', 5.0))
                    exposure_factor = max(
                        adj_min, min(adj_max, exposure_factor))
                    recommended_exp = base_exposure * exposure_factor

                    blur_px_limit = float(
                        capture_cfg.get('target_blur_px', 1.0))
                    t_max_blur = float('inf')
                    if ang_speed > 1e-3 and plate_scale > 1e-3:
                        t_max_blur = (blur_px_limit * plate_scale) / \
                                      (ang_speed * 3600.0)

                    min_exp_limit = float(
                        capture_cfg.get('exposure_min_s', 0.001))
                    max_exp_limit = float(
                        capture_cfg.get('exposure_max_s', 5.0))

                    final_exp = recommended_exp
                    limit_reason = "recommended"
                    if final_exp > t_max_blur:
                        final_exp = t_max_blur
                        limit_reason = f"blur_limit ({t_max_blur:.3f}s)"
                    if final_exp > max_exp_limit:
                        if limit_reason == "recommended" or t_max_blur > max_exp_limit:
                            limit_reason = f"config_max ({max_exp_limit:.3f}s)"
                        final_exp = max_exp_limit
                    if final_exp < min_exp_limit:
                        if limit_reason == "recommended" or t_max_blur < min_exp_limit:
                            limit_reason = f"config_min ({min_exp_limit:.3f}s)"
                        final_exp = min_exp_limit

                    logger.info(
                        f"    - Final sequence exposure: {final_exp:.4f}s (Reason: {limit_reason})")

                    last_seq_ts = time.time()  # Set timestamp immediately

                    # Create the log and status payloads here, in the guide thread
                    seq_log = {
                        'type': 'sequence',
                        'sequence_id': int(last_seq_ts * 1000),
                        'icao': icao,
                        'timestamp': last_seq_ts,
                        'per_frame_exposure_s': float(final_exp),
                        'exposure_limit_reason': limit_reason,
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
                    if self.track_stop_event.is_set():
                        logger.info(
                            f"  Track [{icao}]: Shutdown signaled, skipping sequence scheduling.")
                        break
                    # Put the job on the queue for the worker
                    job = (icao, final_exp, seq_log,
                           captures_taken, status_payload_base)
                    self.capture_queue.put(job)

                    # Update captures_taken *immediately* in this thread
                    # The worker will add the *real* count later
                    with self.track_lock:
                        self.captures_taken += int(
                            CONFIG['capture'].get('num_sequence_images', 5))
                        captures_taken = self.captures_taken

                    # Write status immediately, worker will update it again
                    status_update = status_payload_base.copy()
                    status_update['captures_taken'] = captures_taken
                    status_update['sequence_count'] = 0  # Worker will set this
                    write_status(status_update)

                loop_duration = time.time() - loop_start_time
                sleep_time = max(0.01, 0.1 - loop_duration)
                time.sleep(sleep_time)

        finally:
            logger.info(f"  Track [{icao}]: Exiting tracking thread.")
            with self.track_lock:
                was_preempted = self.preempt_requested
                finished_icao = self.current_track_icao
                self.current_track_icao = None
                self.current_track_ev = 0.0
                self.active_thread = None
                self.preempt_requested = False
                if self.manual_override_active == finished_icao:
                    self.manual_override_active = None

            was_aborted_by_signal = self.track_stop_event.is_set()

            if self.monitor_mode:
                final_mode = "monitor"
            else:
                final_mode = "idle"
            final_status = {"mode": final_mode, "icao": None}
            if was_aborted_by_signal and not was_preempted:
                logger.info(f"  Tracking for {finished_icao} was interrupted.")
                if hasattr(self, 'manual_target_icao') and self.manual_target_icao == finished_icao:
                    final_status["manual_target"] = {"icao": finished_icao, "viability": {
                        "viable": False, "reasons": ["track aborted"]}}
                else:
                    final_status["manual_target"] = None
            elif was_preempted:
                logger.info(
                    f"  Tracking for {finished_icao} preempted. Re-evaluating targets...")
                final_status["manual_target"] = None
            else:
                logger.info(
                    f"  Tracking for {finished_icao} finished normally. System idle.")
                final_status["manual_target"] = None

            write_status(final_status)

            if was_preempted or not was_aborted_by_signal:
                logger.info("  Triggering scheduler to find next target...")
                with self.track_lock:
                    if not self.is_scheduler_waiting:
                        self._run_scheduler()


if __name__ == "__main__":
    setup_logging()
    ensure_log_dir()
    adsb_watch_file = os.path.normpath(
        os.path.abspath(CONFIG['adsb']['json_file_path']))
    adsb_watch_path = os.path.dirname(adsb_watch_file)
    if not os.path.exists(adsb_watch_path):
        exit(1)
    if not os.path.exists(adsb_watch_file): logger.warning(
        f"Warning: ADS-B file '{adsb_watch_file}' does not exist yet...")

    event_handler = FileHandler(adsb_watch_file)
    logger.info("Performing initial read of aircraft data...")
    event_handler.process_file()

    # Check for and process command.json at startup
    command_filename = CONFIG['logging']['command_file']
    command_file_path = os.path.normpath(
        os.path.join(LOG_DIR, command_filename))
    if os.path.exists(command_file_path):
        logger.info(
            f"Found existing command file '{command_file_path}' at startup. Processing...")
        try:
            with open(command_file_path, 'r') as f:
                command = json.load(f)
            if 'track_icao' in command:
                icao = command['track_icao'].lower().strip()
                logger.info(
                    f"  Manual override to track ICAO: {icao} from startup command.")
                event_handler.manual_target_icao = icao
                # Trigger scheduler to pick up the manual target
                event_handler._run_scheduler()
            os.remove(command_file_path)
            logger.info("  Processed and removed startup command file.")
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(
                f"  Error processing startup command file '{command_file_path}': {e}")
        except Exception as e:
            logger.error(
                f"  Unexpected error during startup command processing: {e}")

    adsb_observer = Observer()
    adsb_observer.schedule(
        event_handler, path=adsb_watch_path, recursive=False)
    command_handler = CommandHandler(event_handler)
    command_observer = Observer()
    os.makedirs(LOG_DIR, exist_ok=True)
    command_observer.schedule(command_handler, path=LOG_DIR, recursive=False)

    try:
        adsb_observer.start()
        command_observer.start()
        logger.info(f"Monitoring '{adsb_watch_path}' for ADS-B data...")
        logger.info(
            f"Monitoring '{LOG_DIR}' for command file '{CONFIG['logging']['command_file']}'...")
        while True:
            if not adsb_observer.is_alive() or not command_observer.is_alive():
                logger.critical(
                    "Error: Watchdog observer thread died. Exiting.")
                event_handler.shutdown_event.set()
                event_handler.track_stop_event.set()
                break
            time.sleep(2)
    except KeyboardInterrupt:
        logger.info("\nShutdown requested (Ctrl+C)...")
        # Signal all threads to stop and prevent new tracks from starting
        event_handler.shutdown_event.set()
        event_handler.track_stop_event.set()
        # Notify the stacking orchestrator not to accept any new jobs
        try:
            request_shutdown()
        except Exception as e:
            logger.warning(f"Warning: Error during stack orchestrator request_shutdown: {e}")
        # Cancel any pending scheduler timer to avoid scheduling new track jobs
        if event_handler._scheduler_timer and event_handler._scheduler_timer.is_alive():
            event_handler._scheduler_timer.cancel()
            event_handler.is_scheduler_waiting = False
            logger.info("Cancelled pending scheduler timer.")
        # Wait briefly for the active tracking thread to finish
        active_thread = event_handler.active_thread
        if active_thread and active_thread.is_alive():
            logger.info("Waiting for tracking thread to finish...")
            active_thread.join(timeout=15.0)
            if active_thread.is_alive():
                logger.warning(
                    "Warning: Tracking thread did not exit cleanly.")
        # Shut down the stack orchestrator early to prevent new jobs from being scheduled
        try:
            shutdown_stack_orchestrator()
        except Exception as e:
            logger.warning(
                f"Warning: Error during stack orchestrator shutdown: {e}")
    except Exception as e:
        logger.critical(f"FATAL: Unhandled exception in main loop: {e}")
        event_handler.shutdown_event.set()
        event_handler.track_stop_event.set()
        traceback.print_exc()
    finally:
        logger.info("Stopping observers...")
        if adsb_observer.is_alive():
            adsb_observer.stop()
        if command_observer.is_alive():
            command_observer.stop()
        if event_handler.indi_controller:
            logger.info("Disconnecting hardware...")
            try:
                event_handler.indi_controller.disconnect()
            except Exception:
                logger.warning(
                    "Warning: Error during hardware disconnect:", exc_info=True)
        try:
            if adsb_observer.is_alive():
                adsb_observer.join(timeout=2.0)
            if command_observer.is_alive():
                command_observer.join(timeout=2.0)
        except RuntimeError as e:
            logger.warning(f"Warning: Error joining observer threads: {e}")
        logger.info("Shutting down stack orchestrator...")
        try:
            shutdown_stack_orchestrator()
        except RuntimeError as e:
            logger.warning(
                f"Warning: Error during stack orchestrator shutdown: {e}")
        logger.info("Program terminated.")
