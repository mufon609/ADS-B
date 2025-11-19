# commands/idle.py
import logging
import threading
from typing import Any, Dict, Optional

from .base import Command

logger = logging.getLogger(__name__)


class IdleCommand(Command):
    """Stops tracking and keeps the hardware in a monitoring/idle state."""

    def execute(self, params: Dict[str, Any]) -> bool:
        reason = params.get("reason", "manual")
        logger.info(f"=== IDLE / MONITOR COMMAND RECEIVED ({reason}) ===")

        old_thread: Optional[threading.Thread] = None

        with self.context.track_lock:
            if (self.context._scheduler_timer and
                    self.context._scheduler_timer.is_alive()):
                self.context._scheduler_timer.cancel()
                self.context._scheduler_timer = None
                self.context.is_scheduler_waiting = False

            self.context.manual_target_icao = None
            self.context.manual_viability_info = None
            self.context.preempt_requested = False
            self.context.manual_override_active = None
            self.context.monitor_mode = True

            if self.context.current_track_icao:
                logger.info(
                    f"  Stopping current track {self.context.current_track_icao} for idle mode")
                self.context.track_stop_event.set()
                old_thread = self.context.active_thread

        if old_thread and old_thread.is_alive():
            logger.info("  Waiting for tracking thread to terminate...")
            old_thread.join(timeout=10.0)
            if old_thread.is_alive():
                logger.warning(
                    "  Tracking thread did not exit within timeout — forcing state reset")
                self.context._force_reset_tracking_state()

        controller = getattr(self.context, "indi_controller", None)
        if controller:
            try:
                controller.enter_idle_mode()
            except Exception:
                logger.warning("  Error while entering hardware idle mode:",
                               exc_info=True)

        if not self.context.shutdown_event.is_set():
            self.context.track_stop_event.clear()

        logger.info("  System is now in monitor/idle mode (no auto-tracking).")
        return True
