# commands/base.py

from abc import ABC, abstractmethod
from typing import Any, Dict
import logging

logger = logging.getLogger(__name__)

class Command(ABC):
    """Base class for all system commands."""

    def __init__(self, context: Any):
        """Inject shared runtime state (scheduler, locks, hardware controller, etc.)."""
        self.context = context

    @abstractmethod
    def execute(self, params: Dict[str, Any]) -> bool:
        """
        Run the command with the provided parameters.

        Args:
            params: Command-specific options/flags (may be empty).

        Returns:
            True when the command completed and the system reached the intended state;
            False otherwise.
        """
        pass
