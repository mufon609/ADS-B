#status_writer.py
"""
Module for atomically writing the application status to a JSON file.
"""
import json
import logging
import os
import tempfile
import threading
import time

import numpy as np

from config_loader import CONFIG, LOG_DIR

logger = logging.getLogger(__name__)

_status_lock = threading.Lock()
_current_status_cache: dict = {"mode": "initializing"}

def convert_numpy_types(obj):
    """Recursively converts NumPy types in a dict/list to standard Python types for JSON."""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    # Handle various NumPy integer types using the abstract base class
    elif isinstance(obj, (np.integer, np.floating)):
        if isinstance(obj, np.floating) and (np.isnan(obj) or np.isinf(obj)):
            return None if np.isnan(obj) else str(obj)
        return int(obj) if isinstance(obj, np.integer) else float(obj)
    # Handle complex numbers (store as dict)
    elif isinstance(obj, np.complexfloating): # More general check for complex
        return {'real': obj.real, 'imag': obj.imag}
    # Handle NumPy arrays (convert to list)
    elif isinstance(obj, np.ndarray):
        return convert_numpy_types(obj.tolist()) # Convert arrays to lists recursively
    # Handle NumPy boolean
    elif isinstance(obj, np.bool_):
        return bool(obj)
    # Handle NumPy void type (often from structured arrays)
    elif isinstance(obj, np.void):
        return None # Represent void as null or handle appropriately
    # Return object unchanged if not a NumPy type or container
    return obj

def write_status(status: dict):
    """
    Updates the in-memory status cache and atomically writes the full status
    to a JSON file the dashboard can read. Disk I/O happens outside the lock.
    """
    global _current_status_cache # Explicitly modify global cache

    status_to_write = None
    with _status_lock:
        try:
            # Deep update to handle nested dictionaries safely
            # Convert input status dict numpy types *before* merging
            safe_status_update = convert_numpy_types(status)
            # Merge update into cache
            for key, value in safe_status_update.items():
                 _current_status_cache[key] = value # Simple update replaces/adds keys
            _current_status_cache['updated_at'] = time.time()
            # Create a copy for writing (conversion already happened)
            status_to_write = _current_status_cache.copy()
        except Exception as e:
            logger.error(f"Error updating status cache: {e}")
            status_to_write = _current_status_cache.copy() if _current_status_cache else None

    if status_to_write: # Ensure we have data to write
        path = os.path.join(LOG_DIR, 'status.json')
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)

            fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".status.", suffix=".json", text=True)
            try:
                with os.fdopen(fd, "w", encoding='utf-8') as f: # Ensure UTF-8
                    # Ensure conversion just before dumping
                    json_compatible_status = convert_numpy_types(status_to_write)
                    json.dump(json_compatible_status, f, indent=2, allow_nan=False) # Ensure allow_nan=False
                    f.flush()
                    os.fsync(f.fileno()) # Ensure data is on disk
                os.replace(tmp_path, path) # Atomic rename
            finally:
                try:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                except OSError: pass # Ignore cleanup errors
        except Exception as e:
            # Print the specific error related to writing
            logger.error(f"Error writing status file '{path}': {e}")
             # Optionally print the problematic dictionary for debugging
             # import sys
             # logger.error(f"Problematic status dict: {status_to_write}", file=sys.stderr)