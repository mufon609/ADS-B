# commands/auto_track.py
import logging
from typing import Any, Dict

from tracking.state import TrackingIntent
from .base import Command

logger = logging.getLogger(__name__)


class AutoTrackCommand(Command):
    """Re-enables automatic tracking of the best candidate."""

    def execute(self, params: Dict[str, Any]) -> bool:
        """
        Exit manual/monitor modes and restart the scheduler-driven auto-tracking loop.

        Args:
            params: Unused; accepted for interface consistency.

        Returns:
            True once auto-tracking is re-enabled.
        """
        logger.info("=== AUTO TRACK COMMAND RECEIVED ===")

        # If a force-stop interlock is active, clear it only when no tracking thread is alive.
        with self.context.state.lock:
            if self.context.state.is_force_stop_active():
                active_thread = getattr(self.context, "active_thread", None)
                if active_thread and active_thread.is_alive():
                    logger.error("  Force-stop interlock active and tracking thread still alive; cannot resume auto-tracking.")
                    return False
                logger.warning("  Clearing force-stop interlock before resuming auto-tracking.")
                self.context.state.clear_force_stop()

        with self.context.state.lock:
            self.context.state.apply_intent(
                intent=TrackingIntent.EXIT_MONITOR
            )
            self.context.state.clear_manual_state()

        self.context.state.clear_track_stop_if_running()

        logger.info("  Auto-tracking enabled. Scheduler will resume.")
        self.context._run_scheduler()
        return True
