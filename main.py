# main.py
"""
Main entry point: Monitors ADS-B data, processes flight paths, and tracks aircraft.
"""

import json
import logging
import os
import queue
import subprocess
import threading
import time
import traceback
import sys
from typing import Optional

import astropy.units as u
import numpy as np
from astropy.coordinates import EarthLocation
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from adsb.selector import (
    select_aircraft,
)
from config.loader import CONFIG, LOG_DIR
from astro.coords import (
    distance_km,
)
from adsb.data_reader import read_aircraft_data
from hardware_control import IndiController
from utils.logger_config import setup_logging
from imaging.stacking.orchestrator import request_shutdown
from imaging.stacking.orchestrator import shutdown as shutdown_stack_orchestrator
from utils.status_writer import write_status
from utils.storage import ensure_log_dir
from tracking.command_watcher import CommandHandler
from tracking.capture_worker import capture_worker_loop
from tracking.scheduler import run_scheduler, scheduler_callback
from tracking.state import TrackingState, TrackingFSMState

logger = logging.getLogger(__name__)




class FileHandler(FileSystemEventHandler):
    """
    Handles file events and orchestrates the tracking process.
    Owns the long-lived threads/events (scheduler, tracker, capture worker) and uses locks/events
    to coordinate mount control, captures, and shutdown/preempt signals.
    """

    def __init__(self, watch_file, adsb_observer: Observer, command_observer: Observer, fake_adsb_proc: Optional[subprocess.Popen] = None):
        """
        Initializes the FileHandler.

        Args:
            watch_file (str): The path to the aircraft.json file to monitor.
            adsb_observer (Observer): The watchdog observer for ADS-B data.
            command_observer (Observer): The watchdog observer for command files.
            fake_adsb_proc (Optional[subprocess.Popen]): The subprocess running fake_dump1090_1hz.py if in dry_run mode.
        """
        self.watch_file = os.path.normpath(watch_file)
        self.last_process_time = 0
        self.debounce_seconds = 0.5  # Reduce debounce slightly?
        self.state = TrackingState()
        self.indi_controller = None
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
        self.active_thread = None  # Reference to the current tracking thread
        self.last_file_stat = None  # To detect redundant file events
        self.capture_queue = queue.Queue()
        self.capture_worker_thread = threading.Thread(
            target=capture_worker_loop, args=(self,), daemon=True)
        self.capture_worker_thread.start()


        # Store observers and fake_adsb_proc
        self.adsb_observer = adsb_observer
        self.command_observer = command_observer
        self.fake_adsb_proc = fake_adsb_proc





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
            if (
                self.last_file_stat
                and (current_stat.st_mtime == self.last_file_stat.st_mtime)
                and (current_stat.st_size == self.last_file_stat.st_size)
            ):
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
                # Provide a stop/shutdown handle instead of exposing raw events
                self.indi_controller.shutdown_handle = self.state.get_stop_handle()
                logger.info("Hardware controller initialized.")
            except Exception as e:
                logger.critical(f"FATAL: Hardware initialization failed: {e}")
                write_status(
                    {"mode": "error", "error_message": f"Hardware init failed: {e}"})
                self.indi_controller = None
                return

        # --- Get Current Hardware Status ---
        hardware_status = self.indi_controller.get_hardware_status() or {}
        status_payload = hardware_status.copy()
        snapshot = self.state.get_snapshot()
        manual_icao = snapshot.manual_target_icao
        manual_viability = snapshot.manual_viability
        current_track_icao = snapshot.current_track_icao
        current_track_ev = snapshot.current_track_ev
        manual_override_active = snapshot.manual_override_active
        monitor_mode = snapshot.monitor_mode
        scheduler_waiting = snapshot.scheduler_waiting
        shutdown_requested = snapshot.shutdown

        # --- Handle Empty Airspace ---
        if not aircraft_dict:
            status_payload["mode"] = "tracking" if current_track_icao else "idle"
        if manual_icao:
            status_payload["manual_target"] = {
                "icao": manual_icao,
                "viability": manual_viability or {
                    "viable": False,
                    "reasons": ["no ADS-B data available"],
                },
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
        is_tracking_icao = current_track_icao is not None
        is_actively_tracking = snapshot.fsm_state == TrackingFSMState.TRACKING
        has_candidates = bool(candidates)
        is_manual_override_active = (
            manual_override_active
            and manual_override_active != current_track_icao
        )

        # Only consider preemption when actively tracking (skip during slews/focus)
        if is_tracking_icao and is_actively_tracking and has_candidates and not is_manual_override_active:
            new_best_target = candidates[0]
            preempt_factor = float(
                CONFIG['selection'].get('preempt_factor', 1.25))
            
            is_new_target_different = new_best_target['icao'] != current_track_icao
            is_new_target_better = new_best_target['ev'] > (current_track_ev * preempt_factor)
            
            if is_new_target_different and is_new_target_better:
                logger.info(
                    f"--- PREEMPTIVE SWITCH: New target {new_best_target['icao']} "
                    f"(EV {new_best_target['ev']:.1f}) > Current {current_track_icao} "
                    f"(EV {current_track_ev:.1f} * {preempt_factor:.2f}). ---"
                )
                with self.state.lock:
                    self.state.cancel_scheduler_timer()
                    self.state.mark_preempt(True)
                    self.state.signal_track_stop()

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
            "queue": candidates[1:6] if current_track_icao else candidates[:5],
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
        status_payload["mode"] = snapshot.mode
        status_payload["monitor_mode"] = monitor_mode

        if manual_icao:
            status_payload["manual_target"] = {
                "icao": manual_icao,
                "viability": manual_viability
            }

        # --- Write Status ---
        write_status(status_payload)

        # --- Run Scheduler (if not already tracking and not waiting) ---
        # Do not invoke the scheduler if a shutdown has been requested
        is_waiting = scheduler_waiting
        has_track = bool(current_track_icao)
        if not shutdown_requested and not has_track and not is_waiting:
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

    def _run_scheduler(self, candidates: Optional[list] = None):
        run_scheduler(self, candidates)

    def _run_scheduler_callback(self):
        scheduler_callback(self)

    def _force_reset_tracking_state(self):
        """
        Emergency reset used when a tracking thread fails to exit.
        Leaves the global shutdown signal untouched but clears per-track state so
        subsequent commands can proceed.
        """
        logger.warning("Force-stop asserted due to unresponsive track thread.")
        self.state.force_stop_tracking(reason="unresponsive track thread")
        with self.state.lock:
            self.active_thread = None

    def reset_captures(self) -> int:
        """Reset the shared capture counter."""
        return self.state.reset_captures()

    def increment_captures(self, n: int) -> int:
        """Increment the shared capture counter and return the new value."""
        return self.state.increment_captures(n)

    def set_captures_if_current(self, icao: str, captures: int) -> int:
        """
        Set captures_taken if the provided ICAO matches the current track.
        Returns the resulting captures_taken value.
        """
        return self.state.set_captures_if_current(icao, captures)

    def get_captures(self) -> int:
        """Return the current shared capture counter."""
        return self.state.get_captures()


    def shutdown(self):
        """
        Performs a graceful shutdown of all active components, including observers,
        hardware, the stacking orchestrator, and any simulated ADS-B processes.
        """
        logger.info("\nShutdown initiated...")
        # Signal all threads to stop and prevent new tracks from starting
        self.state.signal_shutdown()
        self.state.signal_track_stop()

        # Notify the stacking orchestrator not to accept any new jobs
        try:
            request_shutdown()
        except Exception as e:
            logger.warning(f"Warning: Error during stack orchestrator request_shutdown: {e}")

        # Cancel any pending scheduler timer to avoid scheduling new track jobs
        had_live_timer = self.state.has_live_scheduler_timer()
        self.state.cancel_scheduler_timer()
        if had_live_timer:
            logger.info("Cancelled pending scheduler timer.")

        # Wait briefly for the active tracking thread to finish
        active_thread = self.active_thread
        if active_thread and active_thread.is_alive():
            logger.info("Waiting for tracking thread to finish...")
            active_thread.join(timeout=15.0)
            if active_thread.is_alive():
                logger.warning("Warning: Tracking thread did not exit cleanly.")

        self._stop_stack_orchestrator()
        self._stop_observers()
        self._stop_hardware()
        self._stop_fake_adsb()
        logger.info("Program terminated.")

    def _stop_stack_orchestrator(self):
        """Shut down the stack orchestrator to prevent new jobs and stop workers."""
        try:
            shutdown_stack_orchestrator()
        except Exception as e:
            logger.warning(f"Warning: Error during stack orchestrator shutdown: {e}")

    def _stop_observers(self):
        """Stop and join file observers."""
        logger.info("Stopping observers...")
        if self.adsb_observer and self.adsb_observer.is_alive():
            self.adsb_observer.stop()
        if self.command_observer and self.command_observer.is_alive():
            self.command_observer.stop()

        try:
            if self.adsb_observer and self.adsb_observer.is_alive():
                self.adsb_observer.join(timeout=2.0)
            if self.command_observer and self.command_observer.is_alive():
                self.command_observer.join(timeout=2.0)
        except RuntimeError as e:
            logger.warning(f"Warning: Error joining observer threads: {e}")

    def _stop_hardware(self):
        """Disconnect hardware cleanly (if initialized)."""
        if not self.indi_controller:
            return
        logger.info("Disconnecting hardware...")
        try:
            self.indi_controller.disconnect()
        except Exception:
            logger.warning("Warning: Error during hardware disconnect:", exc_info=True)

    def _stop_fake_adsb(self):
        """Terminate the fake ADS-B process if running."""
        if not self.fake_adsb_proc:
            return
        logger.info("Stopping fake_dump1090_1hz.py...")
        self.fake_adsb_proc.terminate()
        try:
            self.fake_adsb_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning("fake_dump1090_1hz.py did not exit; killing.")
            self.fake_adsb_proc.kill()

def process_startup_command(event_handler):
    """
    Reads and processes any existing command file at startup to apply manual overrides.
    """
    command_filename = CONFIG['logging']['command_file']
    command_file_path = os.path.normpath(os.path.join(LOG_DIR, command_filename))
    if not os.path.exists(command_file_path):
        return

    logger.info(f"Found existing command file '{command_file_path}' at startup. Processing...")
    try:
        with open(command_file_path, 'r') as f:
            command = json.load(f)
        if 'track_icao' in command:
            icao = command['track_icao'].lower().strip()
            logger.info(f"  Manual override to track ICAO: {icao} from startup command.")
            event_handler.state.set_manual_target(icao)
            # Trigger scheduler to pick up the manual target
            event_handler._run_scheduler()
        os.remove(command_file_path)
        logger.info("  Processed and removed startup command file.")
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"  Error processing startup command file '{command_file_path}': {e}")
    except Exception as e:
        logger.error(f"  Unexpected error during startup command processing: {e}")


def start_fake_adsb():
    """
    Launches the fake ADS-B generator in dry_run mode.

    Returns:
        subprocess.Popen | None: The process handle if started, otherwise None.
    """
    if not CONFIG['development'].get('dry_run'):
        return None

    fake_adsb_script = os.path.join(
        os.path.dirname(__file__), "scripts", "fake_dump1090_1hz.py")
    out_arg = CONFIG['adsb']['json_file_path']
    try:
        proc = subprocess.Popen(
            [sys.executable, fake_adsb_script, "--out", out_arg],
            cwd=os.path.dirname(fake_adsb_script),
        )
        logger.info(
            f"Started fake_dump1090_1hz.py (pid {proc.pid}) writing to {out_arg}")
        return proc
    except Exception as e:
        logger.error(f"Failed to start fake_dump1090_1hz.py: {e}")
        return None


if __name__ == "__main__":
    setup_logging()
    ensure_log_dir()

    # Initialize observers and fake_adsb_proc early to pass to FileHandler
    adsb_observer = Observer()
    command_observer = Observer()
    fake_adsb_proc: Optional[subprocess.Popen] = start_fake_adsb()

    adsb_watch_file = os.path.normpath(
        os.path.abspath(CONFIG['adsb']['json_file_path']))
    adsb_watch_path = os.path.dirname(adsb_watch_file)
    if not os.path.exists(adsb_watch_path):
        exit(1)
    if not os.path.exists(adsb_watch_file):
        logger.warning(
            f"Warning: ADS-B file '{adsb_watch_file}' does not exist yet...")

    event_handler = FileHandler(adsb_watch_file, adsb_observer, command_observer, fake_adsb_proc)
    logger.info("Performing initial read of aircraft data...")
    event_handler.process_file()

    # Check for and process command.json at startup
    process_startup_command(event_handler)

    adsb_observer.schedule(
        event_handler, path=adsb_watch_path, recursive=False)
    command_handler = CommandHandler(event_handler)
    os.makedirs(LOG_DIR, exist_ok=True)
    command_observer.schedule(command_handler, path=LOG_DIR, recursive=False)

    try:
        adsb_observer.start()
        command_observer.start()
        logger.info(f"Monitoring '{adsb_watch_path}' for ADS-B data...")
        logger.info(
            f"Monitoring '{LOG_DIR}' for command file '{CONFIG['logging']['command_file']}'...")
        while True:
            adsb_observer_alive = adsb_observer.is_alive()
            command_observer_alive = command_observer.is_alive()

            if not adsb_observer_alive or not command_observer_alive:
                logger.critical(
                    "Error: Watchdog observer thread died. Exiting.")
                event_handler.state.signal_shutdown()
                event_handler.state.signal_track_stop()
                break
            time.sleep(2)
    except KeyboardInterrupt:
        logger.info("\nShutdown requested (Ctrl+C)...")
        event_handler.shutdown()
    except Exception as e:
        logger.critical(f"FATAL: Unhandled exception in main loop: {e}")
        # In case of an unhandled exception, ensure a shutdown is attempted.
        event_handler.state.signal_shutdown()
        event_handler.state.signal_track_stop()
        traceback.print_exc()
    finally:
        # Ensure final cleanup even if an unexpected error occurred before calling shutdown.
        # The shutdown method itself will log 'Program terminated.'
        if not event_handler.state.is_shutdown(): # Only call if not already in shutdown state
            event_handler.shutdown()
