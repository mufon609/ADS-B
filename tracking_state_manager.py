import traceback
from typing import Dict, List, Optional, Tuple

import numpy as np

from config_loader import CONFIG, LOG_DIR
from storage import append_to_json, ensure_log_dir
from image_analyzer import _to_img_url
from status_writer import write_status

logger = logging.getLogger(__name__)

class TrackingStateManager:
    """
    Manages all tracking-related state variables and provides thread-safe access
    and modification methods. This centralizes the state and its concurrency
    management, decoupling it from the FileHandler's event processing.
    """

    def __init__(self):
        """
        Initializes the TrackingStateManager with all necessary state variables
        and starts the capture worker thread.
        """
        self.current_track_icao: Optional[str] = None
        self.current_track_ev: float = 0.0
        self.track_lock = threading.RLock()  # Protects access to tracking state vars
        self.shutdown_event = threading.Event()  # Signals tracking thread to stop
        self.active_thread: Optional[threading.Thread] = None  # Reference to the current tracking thread
        self.manual_target_icao: Optional[str] = None  # ICAO requested manually
        self.manual_viability_info: Optional[Dict] = None  # Cache last manual viability details for status
        self._scheduler_timer: Optional[threading.Timer] = None  # Timer for delayed scheduling
        self.preempt_requested: bool = False  # Flag for preemptive track switching
        self.is_scheduler_waiting: bool = False # Flag to indicate if scheduler is waiting for a timer
        self.capture_queue: queue.Queue = queue.Queue()
        self.capture_worker_thread = threading.Thread(
            target=self._capture_worker_loop, daemon=True)
        self.capture_worker_thread.start()

        # Shared counter for the number of images captured for the current track.
        # This counter is protected by track_lock and reset whenever a new track begins.
        self.captures_taken: int = 0

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
                 captures_taken_so_far, status_payload_base, indi_controller_ref) = job

                # Use the passed indi_controller_ref
                captured_paths = indi_controller_ref.capture_sequence(
                    icao, final_exp)

                if captured_paths:
                    # Update log and status *after* capture is complete
                    seq_log = seq_log_base.copy()
                    seq_log['n_frames'] = len(captured_paths)
                    seq_log['image_paths'] = captured_paths
                    append_to_json([seq_log], os.path.join(
                        LOG_DIR, 'captures.json'))

                    captures_taken = captures_taken_so_far +
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
            except Exception as e:
                # Log errors from the capture worker
                logger.error(f"  Capture worker error: {e}")
                traceback.print_exc()

        logger.info("  Capture worker thread shutting down.")

    # --- Getter/Setter methods for state variables (example) ---
    def get_current_track_icao(self) -> Optional[str]:
        with self.track_lock:
            return self.current_track_icao

    def set_current_track_icao(self, icao: Optional[str]):
        with self.track_lock:
            self.current_track_icao = icao

    def get_current_track_ev(self) -> float:
        with self.track_lock:
            return self.current_track_ev

    def set_current_track_ev(self, ev: float):
        with self.track_lock:
            self.current_track_ev = ev

    def get_manual_target_icao(self) -> Optional[str]:
        with self.track_lock:
            return self.manual_target_icao

    def set_manual_target_icao(self, icao: Optional[str]):
        with self.track_lock:
            self.manual_target_icao = icao

    def get_manual_viability_info(self) -> Optional[Dict]:
        with self.track_lock:
            return self.manual_viability_info

    def set_manual_viability_info(self, info: Optional[Dict]):
        with self.track_lock:
            self.manual_viability_info = info

    def get_scheduler_timer(self) -> Optional[threading.Timer]:
        with self.track_lock:
            return self._scheduler_timer

    def set_scheduler_timer(self, timer: Optional[threading.Timer]):
        with self.track_lock:
            self._scheduler_timer = timer

    def get_is_scheduler_waiting(self) -> bool:
        with self.track_lock:
            return self.is_scheduler_waiting

    def set_is_scheduler_waiting(self, waiting: bool):
        with self.track_lock:
            self.is_scheduler_waiting = waiting

    def get_preempt_requested(self) -> bool:
        with self.track_lock:
            return self.preempt_requested

    def set_preempt_requested(self, requested: bool):
        with self.track_lock:
            self.preempt_requested = requested

    def get_captures_taken(self) -> int:
        with self.track_lock:
            return self.captures_taken

    def set_captures_taken(self, count: int):
        with self.track_lock:
            self.captures_taken = count

    def increment_captures_taken(self, increment: int):
        with self.track_lock:
            self.captures_taken += increment

    def get_active_thread(self) -> Optional[threading.Thread]:
        with self.track_lock:
            return self.active_thread

    def set_active_thread(self, thread: Optional[threading.Thread]):
        with self.track_lock:
            self.active_thread = thread

    def get_shutdown_event(self) -> threading.Event:
        # Shutdown event does not need to be protected by track_lock for direct access
        return self.shutdown_event

    def put_capture_job(self, job: Tuple):
        self.capture_queue.put(job)

    def cancel_scheduler_timer(self):
        with self.track_lock:
            if self._scheduler_timer and self._scheduler_timer.is_alive():
                self._scheduler_timer.cancel()
                self._scheduler_timer = None
                self.is_scheduler_waiting = False
                logger.info("  Cancelled pending scheduler timer.")

    def signal_shutdown(self):
        self.shutdown_event.set()

    def clear_shutdown_event(self):
        self.shutdown_event.clear()
