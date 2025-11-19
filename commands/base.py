# commands/base.py

from abc import ABC, abstractmethod
from typing import Any, Dict
import logging

logger = logging.getLogger(__name__)

class Command(ABC):
    """Base class for all system commands."""

    def __init__(self, context: Any):
        self.context = context

    @abstractmethod
    def execute(self, params: Dict[str, Any]) -> bool:
        """
        Execute the command.
        Returns True if command completed successfully and system is in expected state.
        """
        pass

    def _force_reset_tracking_state(self) -> None:
        """Nuclear option: forcibly clear tracking state if thread is dead/hung."""
        with self.context.track_lock:
            if self.context.current_track_icao:
                logger.warning(
                    f"!!! FORCIBLY clearing tracking state for stuck thread (ICAO: {self.context.current_track_icao}) !!!"
                )
                self.context.current_track_icao = None
                self.context.current_track_ev = 0.0
                self.context.active_thread = None
                self.context.preempt_requested = False
                # Do NOT clear track_stop_event — let it stay set so any stragglers die
