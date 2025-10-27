"""
Module for logging data to JSON files with atomic writes and log rotation.
"""
import json
import os
import time
import threading
from typing import List, Dict, Any, Iterable
from config_loader import CONFIG, LOG_DIR

_log_lock = threading.Lock()

def ensure_log_dir():
    """Creates log directory and images subdir if they don't exist, using absolute paths."""
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(os.path.join(LOG_DIR, 'images'), exist_ok=True)

def _fsync_dir(dir_path: str):
    """Fsync the containing directory to make a rename operation durable on disk."""
    try:
        dir_fd = os.open(dir_path, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except Exception:
        pass

def _read_log_object(file_path: str) -> Dict[str, Any]:
    """Reads a JSON log file, safely handling errors and always returning a valid structure."""
    def _blank():
        return {"metadata": {"epoch": "unix_utc", "version": "1.0"}, "data": []}

    if not os.path.exists(file_path):
        return _blank()
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            if not content:
                return _blank()
            obj = json.loads(content)
    except (OSError, json.JSONDecodeError):
        return _blank()

    if isinstance(obj, list):
        obj = {
            "metadata": {"epoch": "unix_utc", "version": "1.0", "migrated": True},
            "data": obj
        }
    if 'metadata' not in obj or not isinstance(obj.get('metadata'), dict):
        obj['metadata'] = {"epoch": "unix_utc", "version": "1.0"}
    if 'data' not in obj or not isinstance(obj.get('data'), list):
        obj['data'] = []
    return obj

def _rotate(file_path: str) -> str:
    """Rotates the log file by renaming it with a timestamp and PID."""
    backup_path = f"{file_path}.{int(time.time())}.{os.getpid()}.bak"
    try:
        os.rename(file_path, backup_path)
        print(f"Log file {file_path} rotated to {backup_path}")
        _fsync_dir(os.path.dirname(file_path))
    except OSError as e:
        print(f"Error rotating log file {file_path}: {e}")
    return backup_path

def _should_rotate(file_path: str, new_records: Iterable[Dict[str, Any]], threshold_bytes: int) -> bool:
    """Determines if a log file should be rotated before adding new content."""
    try:
        current_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
    except OSError:
        current_size = 0

    if current_size >= threshold_bytes:
        return True

    try:
        added_bytes = len(json.dumps(list(new_records), ensure_ascii=False, separators=(',', ':')).encode('utf-8'))
    except Exception:
        added_bytes = 2048

    return (current_size + added_bytes) > threshold_bytes

def append_to_json(datas: List[Dict[str, Any]], file_path: str):
    """Appends records to a JSON log file atomically, with pre-emptive rotation."""
    if not isinstance(datas, list) or not all(isinstance(x, dict) for x in datas):
        return

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    threshold_mb = int(CONFIG.get('logging', {}).get('log_max_size_mb', 25))
    threshold_bytes = threshold_mb * 1024 * 1024

    with _log_lock:
        if _should_rotate(file_path, datas, threshold_bytes):
            if os.path.exists(file_path):
                _rotate(file_path)

        log_object = _read_log_object(file_path)
        log_object['data'].extend(datas)

        dir_path = os.path.dirname(file_path)
        base_name = os.path.basename(file_path)
        tmp_path = os.path.join(dir_path, f".{base_name}.{os.getpid()}.tmp")

        try:
            with open(tmp_path, 'w', encoding='utf-8') as f:
                json.dump(log_object, f, indent=2, ensure_ascii=False, allow_nan=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, file_path)
            _fsync_dir(dir_path)
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass