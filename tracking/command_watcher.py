import json
import logging
import os
from watchdog.events import FileSystemEventHandler

from config.loader import CONFIG, LOG_DIR
from commands.track import TrackCommand
from commands.abort import AbortCommand
from commands.park import ParkCommand
from commands.idle import IdleCommand
from commands.auto_track import AutoTrackCommand

logger = logging.getLogger(__name__)


class CommandHandler(FileSystemEventHandler):
    """A simple handler to watch for manual override commands."""

    def __init__(self, main_handler):
        """
        Initializes the CommandHandler.

        Args:
            main_handler: The main FileHandler instance to interact with.
        """
        self.main_handler = main_handler
        command_filename = CONFIG['logging']['command_file']
        self.command_file = os.path.normpath(
            os.path.join(LOG_DIR, command_filename))

    def on_modified(self, event):
        """
        Called by the watchdog observer when a file is modified.

        If the modified file is the command file, this method triggers
        the command processing logic.

        Args:
            event: The file system event object.
        """
        if os.path.normpath(event.src_path) == self.command_file:
            self._process_command()

    def on_created(self, event):
        """
        Called by the watchdog observer when a file is created.

        If the created file is the command file, this method triggers
        the command processing logic.

        Args:
            event: The file system event object.
        """
        if os.path.normpath(event.src_path) == self.command_file:
            self._process_command()

    def on_moved(self, event):
        """
        Called by the watchdog observer when a file is moved or renamed.

        This handles atomic writes (e.g., `mv` or `os.replace`), which
        often trigger a `moved` event. If the destination is the command
        file, it triggers the command processing logic.

        Args:
            event: The file system event object.
        """
        # Watchdog triggers moved event even for atomic replace if dest exists
        if os.path.normpath(event.dest_path) == self.command_file:
            self._process_command()

    def _resolve_command_class(self, data):
        """
        Returns the appropriate command class for the given payload, or None if unknown.
        """
        command_map = {
            'track_icao': TrackCommand,
            'abort_track': AbortCommand,
            'park_mount': ParkCommand,
            'idle_monitor': IdleCommand,
            'auto_track': AutoTrackCommand,
        }
        if 'track_icao' in data:
            return command_map['track_icao']
        cmd_key = data.get('command')
        if cmd_key in command_map:
            return command_map[cmd_key]
        return None

    def _process_command(self):
        """Reads and processes the command file, handling race conditions."""
        logger.info("--- Manual command detected! ---")
        try:
            # === NEW: Clean, extensible command dispatcher ===
            with open(self.command_file, 'r') as f:
                data = json.load(f)

            # Atomic delete - fail loudly if gone (means another instance already processed it)
            os.remove(self.command_file)
            logger.info("Processed and removed command file.")

            cmd_class = self._resolve_command_class(data)

            if not cmd_class:
                logger.warning(f"Unknown command received: {data}")
                return

            success = cmd_class(self.main_handler).execute(data)
            logger.info(f"Command executed → {'SUCCESS' if success else 'FAILED'}")

        except FileNotFoundError:
            logger.info(
                "  Command file not found (likely already processed or deleted). Ignoring.")
        except json.JSONDecodeError as e:
            logger.warning(
                f"  Could not read command file (invalid JSON): {e}")
        except Exception as e:
            # Catch unexpected errors during command processing
            logger.error(f"  Unexpected error processing command: {e}")
