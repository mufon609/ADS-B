# commands/park.py
import logging
import threading
from typing import Any, Dict
from .base import Command

logger = logging.getLogger(__name__)

class ParkCommand(Command):
    """Stops tracking and parks the mount. Fully bulletproof."""

    def execute(self, params: Dict[str, Any]) -> bool:
        logger.info("=== PARK COMMAND RECEIVED ===")

        old_thread = None

        with self.context.track_lock:
            # Cancel any pending scheduler actions
            if (
                self.context._scheduler_timer
                and self.context._scheduler_timer.is_alive()
            ):
                self.context._scheduler_timer.cancel()
                self.context._scheduler_timer = None
                self.context.is_scheduler_waiting = False

            # Clear manual/preempt flags and enter monitor mode
            self.context.manual_target_icao = None
            self.context.manual_viability_info = None
            self.context.manual_override_active = None
            self.context.preempt_requested = False
            self.context.monitor_mode = True

            # Stop current track if running
            if self.context.current_track_icao:
                logger.info(f"  Stopping current track {self.context.current_track_icao} before parking")
                self.context.track_stop_event.set()
                old_thread = self.context.active_thread

        # Wait for tracking thread to die (outside lock)
        if old_thread and old_thread.is_alive():
            logger.info("  Waiting for tracking thread to terminate before parking...")
            old_thread.join(timeout=10.0)
            if old_thread.is_alive():
                logger.warning("  Tracking thread did not exit cleanly — forcing state reset")
                self.context._force_reset_tracking_state()

        # Ensure we are truly idle and ready for park
        if not self.context.shutdown_event.is_set():
            self.context.track_stop_event.clear()
        logger.info("  Track stop signal cleared after park preparation (monitor mode) ")

        # Send park command (non-blocking)
        if self.context.indi_controller:
            logger.info("  Sending park command to mount...")
            threading.Thread(
                target=self.context.indi_controller.park_mount,
                daemon=True
            ).start()
            return True
        else:
            logger.error("  Park failed: no hardware controller")
            return False
