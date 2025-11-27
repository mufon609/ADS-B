# commands/abort.py
import logging
from typing import Any, Dict

from .base import Command
from .idle import IdleCommand

logger = logging.getLogger(__name__)


class AbortCommand(Command):
    """Compatibility wrapper that invokes Idle/Monitor mode."""

    def execute(self, params: Dict[str, Any]) -> bool:
        """
        Enter idle/monitor mode as a synonym for abort.

        Args:
            params: Optional ``reason`` for logging.

        Returns:
            True when the system transitions to idle.
        """
        logger.info("=== ABORT COMMAND RECEIVED ===")
        return IdleCommand(self.context).execute({"reason": "abort"})
