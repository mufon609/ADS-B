# commands/auto_track.py
import logging
from typing import Any, Dict

from .base import Command

logger = logging.getLogger(__name__)


class AutoTrackCommand(Command):
    """Re-enables automatic tracking of the best candidate."""

    def execute(self, params: Dict[str, Any]) -> bool:
        logger.info("=== AUTO TRACK COMMAND RECEIVED ===")

        with self.context.track_lock:
            self.context.monitor_mode = False
            self.context.manual_target_icao = None
            self.context.manual_viability_info = None
            self.context.manual_override_active = None

        if not self.context.shutdown_event.is_set():
            self.context.track_stop_event.clear()

        logger.info("  Auto-tracking enabled. Scheduler will resume.")
        self.context._run_scheduler()
        return True
