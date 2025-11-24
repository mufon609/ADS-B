# storage.py
"""
Module for logging data to JSON files with atomic writes and log rotation.
"""
import json
import logging
import os
import threading
import time
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional

from config_loader import CONFIG, LOG_DIR

logger = logging.getLogger(__name__)

_file_locks = defaultdict(threading.Lock)


def ensure_log_dir():
    """Creates log directory and images subdir if they don't exist, using absolute paths."""
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(os.path.join(LOG_DIR, 'images'), exist_ok=True)


def rel_to_logs_url(path: str) -> Optional[str]:
    """Map an absolute path under LOG_DIR to a /logs/... URL for UI links."""
    if not path:
        return None
    try:
        abs_path = os.path.abspath(path)
        abs_root = os.path.abspath(LOG_DIR)
        if (
            os.path.commonpath([abs_path, abs_root]) != abs_root
        ):
            return None
        rel = os.path.relpath(abs_path, abs_root).replace(os.sep, "/")
        return f"/logs/{rel}"
    except Exception:
        logger.exception("Error mapping absolute path to /logs/ URL")
        return None


def _fsync_dir(dir_path: str):
    """Fsync the containing directory to make a rename operation durable on disk."""
    # Skip fsync on Windows as it's not reliably available/needed in the same way
    if os.name == 'nt':
        return
    try:
        # Open directory using O_RDONLY. On some systems O_DIRECTORY is needed/helpful.
        dir_fd = os.open(dir_path, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except Exception as e:
        logger.warning(f"Failed to fsync directory {dir_path}: {e}")


def _read_log_object(file_path: str) -> Dict[str, Any]:
    """Reads a JSON log file, safely handling errors and always returning a valid structure."""
    def _blank():
        return {"metadata": {"epoch": "unix_utc", "version": "1.0"}, "data": []}

    if not os.path.exists(file_path):
        return _blank()
    try:
        # Ensure reading with UTF-8 encoding
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # Handle empty files gracefully
            if not content:
                return _blank()
            obj = json.loads(content)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as e:
        logger.warning(f"Error reading log file {file_path}: {e}")
        return _blank()

    # Basic structure validation and migration for old list-based format
    if isinstance(obj, list):
        obj = {
            "metadata": {"epoch": "unix_utc", "version": "1.0", "migrated": True},
            "data": obj
        }
    if 'metadata' not in obj or not isinstance(obj.get('metadata'), dict):
        obj['metadata'] = {"epoch": "unix_utc", "version": "1.0"}
    if 'data' not in obj or not isinstance(obj.get('data'), list):
        obj['data'] = []  # Ensure 'data' is always a list
    return obj


def _rotate(file_path: str) -> str:
    """Rotates the log file by renaming it with a timestamp and PID."""
    # Create backup path
    backup_path = f"{file_path}.{int(time.time())}.{os.getpid()}.bak"
    try:
        os.rename(file_path, backup_path)
        _fsync_dir(os.path.dirname(file_path))  # Sync directory after rename
    except OSError as e:
        logger.error(f"Error rotating log file {file_path}: {e}")
        # Return original path if rotation fails? Or backup path even if rename failed?
        # Returning backup_path for consistency, though it might not exist.
    return backup_path


def _should_rotate(file_path: str, new_records: Iterable[Dict[str, Any]], threshold_bytes: int) -> bool:
    """Determines if a log file should be rotated before adding new content."""
    try:
        current_size = os.path.getsize(
            file_path) if os.path.exists(file_path) else 0
    except OSError:
        current_size = 0  # Assume 0 if cannot get size

    # Rotate if current size already exceeds threshold
    if current_size >= threshold_bytes:
        return True

    # Estimate size of new records without excessive overhead
    # Use a sample if many records? For now, serialize all.
    # Use separators for compact JSON estimate.
    try:
        # Convert iterable to list for json.dumps
        records_list = list(new_records)
        # Add estimate for list brackets and commas
        added_bytes_estimate = len(json.dumps(
            records_list,
            ensure_ascii=False,
            separators=(',', ':')
        ).encode('utf-8'))
    except Exception:
        # Fallback to a reasonable estimate if serialization fails
        added_bytes_estimate = 2048 * \
            len(records_list) if isinstance(records_list, list) else 2048

    # Check if adding estimated size exceeds threshold
    return (current_size + added_bytes_estimate) > threshold_bytes


def append_to_json(datas: List[Dict[str, Any]], file_path: str):
    """
    Appends records to a JSON log file atomically, with pre-emptive rotation.
    Thread-safe via per-file locks; writes go through a temp file + rename and will rotate
    before exceeding the configured size.
    """
    # Ensure datas is a list and not empty
    if not isinstance(datas, list) or not datas or not all(isinstance(x, dict) for x in datas):
        return
    
    # Ensure datas is a list for consistent handling, especially if it was an iterable
    datas_list = list(datas)

    # Ensure target directory exists
    dir_path = os.path.dirname(file_path)
    try:
        os.makedirs(dir_path, exist_ok=True)
    except OSError as e:
        logger.error(
            f"Error: Could not create directory {dir_path} for log file {file_path}: {e}")
        return  # Cannot proceed if directory creation fails

    # Get rotation threshold from config
    threshold_mb = int(CONFIG.get('logging', {}).get('log_max_size_mb', 25))
    # Ensure at least 1MB threshold
    threshold_bytes = max(1024 * 1024, threshold_mb * 1024 * 1024)

    with _file_locks[file_path]:
        # Check if rotation is needed before reading the log
        if _should_rotate(file_path, datas_list, threshold_bytes):
            if os.path.exists(file_path):
                _rotate(file_path)
                # After rotation, the current file path effectively doesn't exist for reading,
                # so _read_log_object will correctly return a blank structure.

        # Read existing log content (will return blank if file doesn't exist or after rotation)
        log_object = _read_log_object(file_path)
        # Append new data records
        log_object['data'].extend(datas_list)

        # Prepare for atomic write using a temporary file in the same directory
        base_name = os.path.basename(file_path)
        # Use simpler temp file naming convention if needed
        tmp_path = os.path.join(
            dir_path,
            f".{base_name}.{os.getpid()}.{int(time.time_ns())}.tmp"
        )

        try:
            # Write the updated object to the temporary file
            with open(tmp_path, 'w', encoding='utf-8') as f:
                # Use compact separators for potentially smaller file size? Indent=2 is more readable.
                json.dump(
                    log_object, f, indent=2,
                    ensure_ascii=False, allow_nan=False
                )
                # Ensure data is written to OS buffer
                f.flush()
                # Ensure data is written physically to disk
                os.fsync(f.fileno())

            # Atomically replace the original file with the temporary file
            os.replace(tmp_path, file_path)
            # Sync the directory to make the rename persistent
            _fsync_dir(dir_path)
        except Exception as e:
            # Log error if writing or replacing fails
            logger.error(f"Error writing log file {file_path}: {e}")
        finally:
            # Clean up the temporary file if it still exists (e.g., after an error)
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    # Ignore errors during cleanup
                    pass
