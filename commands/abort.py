# commands/abort.py
import logging
from typing import Any, Dict

from .base import Command
from .idle import IdleCommand

logger = logging.getLogger(__name__)


class AbortCommand(Command):
    """Compatibility wrapper that invokes Idle/Monitor mode."""

    def execute(self, params: Dict[str, Any]) -> bool:
        logger.info("=== ABORT COMMAND RECEIVED ===")
        return IdleCommand(self.context).execute({"reason": "abort"})
