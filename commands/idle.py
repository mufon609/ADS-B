# commands/idle.py
import logging
import threading
from typing import Any, Dict, Optional

from tracking.state import TrackingIntent
from .base import Command

logger = logging.getLogger(__name__)


class IdleCommand(Command):
    """Stops tracking and keeps the hardware in a monitoring/idle state."""

    def execute(self, params: Dict[str, Any]) -> bool:
        """
        Stop any active tracking, cancel scheduled tasks, and enter monitor mode.

        Args:
            params: Optional ``reason`` string for logging/UX.

        Returns:
            True once tracking threads are stopped and monitor mode is set.
        """
        reason = params.get("reason", "manual")
        logger.info(f"=== IDLE / MONITOR COMMAND RECEIVED ({reason}) ===")

        old_thread: Optional[threading.Thread] = None

        with self.context.state.lock:
            if self.context.state.has_live_scheduler_timer():
                self.context.state.cancel_scheduler_timer()

            self.context.state.apply_intent(intent=TrackingIntent.ENTER_MONITOR)

            current_track = self.context.state.current_track_icao
            if current_track:
                logger.info(
                    f"  Stopping current track {current_track} for idle mode")
                self.context.state.signal_track_stop()
                old_thread = self.context.active_thread

        if old_thread and old_thread.is_alive():
            logger.info("  Waiting for tracking thread to terminate...")
            old_thread.join(timeout=10.0)
            if old_thread.is_alive():
                logger.warning(
                    "  Tracking thread did not exit within timeout — asserting force-stop")
                self.context._force_reset_tracking_state()

        controller = getattr(self.context, "indi_controller", None)
        if controller:
            try:
                controller.enter_idle_mode()
            except Exception:
                logger.warning("  Error while entering hardware idle mode:",
                               exc_info=True)

        if self.context.state.is_force_stop_active():
            logger.warning("  Force-stop asserted; stop signal remains set for operator intervention.")
            return False

        self.context.state.clear_track_stop_if_running()

        logger.info("  System is now in monitor/idle mode (no auto-tracking).")
        return True
