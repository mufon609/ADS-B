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
