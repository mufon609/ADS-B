import logging
import os
import queue
import time

from config.loader import LOG_DIR
from imaging.analysis import _load_fits_data, _save_png_preview_from_data
from utils.status_writer import write_status
from utils.storage import append_to_json
from utils.capture_types import CaptureJob
from utils.image_utils import to_img_url

logger = logging.getLogger(__name__)


def capture_worker_loop(handler):
    """
    Worker thread that runs blocking captures from the queue.
    This frees the main tracking loop from blocking I/O.
    """
    logger.info("  Capture worker thread started.")
    while not handler.state.is_shutdown():
        try:
            # If a track stop was requested, drop any pending jobs for the old track
            if handler.state.should_stop_tracking():
                try:
                    while True:
                        handler.capture_queue.get_nowait()
                except queue.Empty:
                    pass
                time.sleep(0.1)
                continue

            # Wait for a job, but timeout to check shutdown_event
            job: CaptureJob = handler.capture_queue.get(timeout=1.0)
            if job is None:
                continue

            icao = job.icao
            final_exp = job.exposure
            seq_log_base = job.seq_log
            captures_taken_so_far = job.captures_taken
            status_payload_base = job.status_payload_base

            # Skip work if the track was preempted/aborted after enqueuing
            if handler.state.should_stop_tracking():
                continue

            captured_paths = handler.indi_controller.capture_sequence(
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
                        status_update["capture_png"] = to_img_url(
                            last_raw_png_path)

                # Update the shared captures_taken counter atomically.
                handler.set_captures_if_current(icao, captures_taken)
                # Write status with updated capture count.
                write_status(status_update)

        except queue.Empty:
            # This is normal, just loop and check shutdown_event
            continue
        except Exception:
            # Log errors from the capture worker
            logger.error("  Capture worker error:", exc_info=True)

    logger.info("  Capture worker thread shutting down.")
