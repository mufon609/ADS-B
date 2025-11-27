# commands/park.py
import logging
import threading
from typing import Any, Dict
from tracking.state import TrackingIntent
from .base import Command

logger = logging.getLogger(__name__)

class ParkCommand(Command):
    """Stops tracking and parks the mount. Fully bulletproof."""

    def execute(self, params: Dict[str, Any]) -> bool:
        """
        Cancel tracking and scheduler activity, then issue a non-blocking park command.

        Args:
            params: Unused; present for interface parity.

        Returns:
            True if the park command was issued to the hardware controller; False otherwise.
        """
        logger.info("=== PARK COMMAND RECEIVED ===")

        old_thread = None

        with self.context.state.lock:
            # Cancel any pending scheduler actions
            if self.context.state.has_live_scheduler_timer():
                self.context.state.cancel_scheduler_timer()

            # Clear manual/preempt flags and enter monitor mode
            self.context.state.apply_intent(intent=TrackingIntent.ENTER_MONITOR)

            # Stop current track if running
            current_track = self.context.state.current_track_icao
            if current_track:
                logger.info(f"  Stopping current track {current_track} before parking")
                self.context.state.signal_track_stop()
                old_thread = self.context.active_thread

        # Wait for tracking thread to die (outside lock)
        if old_thread and old_thread.is_alive():
            logger.info("  Waiting for tracking thread to terminate before parking...")
            old_thread.join(timeout=10.0)
            if old_thread.is_alive():
                logger.warning("  Tracking thread did not exit cleanly — asserting force-stop")
                self.context._force_reset_tracking_state()

        # Ensure we are truly idle and ready for park
        if self.context.state.is_force_stop_active():
            logger.warning("  Force-stop asserted; keeping stop signal set after park attempt.")
            return False

        self.context.state.clear_track_stop_if_running()
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
