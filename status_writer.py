"""
Module for atomically writing the application status to a JSON file.
"""
import json
import os
import threading
import time
import tempfile
from config_loader import CONFIG, LOG_DIR

_status_lock = threading.Lock()

def write_status(status: dict):
    """Atomically write a single JSON object the dashboard can read."""
    with _status_lock:
        path = os.path.join(LOG_DIR, 'status.json')
        os.makedirs(os.path.dirname(path), exist_ok=True)

        current_status = {}
        try:
            with open(path, 'r') as f:
                current_status = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            pass

        current_status.update(status)
        current_status['updated_at'] = time.time()

        fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".status.", text=True)
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(current_status, f, indent=2)
            os.replace(tmp_path, path)
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass