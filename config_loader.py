#config_loader.py
"""
Loads and provides the application configuration from a YAML file.

This module mirrors the original project's ``config_loader`` to simplify
configuration management for the local ADS‑B tracker implementation. It
exposes a global ``CONFIG`` dictionary and a ``LOG_DIR`` string. The
configuration is loaded from ``config.yaml`` in the project root by
default, but can be overridden via the ``ADSB_CONFIG_FILE`` environment
variable.  After loading, a few paths are expanded to absolute paths and
the logging directories are created if necessary.
"""

import logging
import os
import sys

import yaml


class ConfigError(RuntimeError):
    """Raised when the configuration file cannot be loaded or is invalid."""
    pass

# --- Load Configuration ---
try:
    CONFIG_FILE = os.environ.get('ADSB_CONFIG_FILE', 'config.yaml')
    if not os.path.exists(CONFIG_FILE):
        raise ConfigError(f"Config file '{CONFIG_FILE}' not found.")

    with open(CONFIG_FILE, 'r') as f:
        CONFIG = yaml.safe_load(f)

    # --- Validate/normalize selection range key ---
    sel_cfg = CONFIG.get('selection', {})
    if 'max_range_nm' in sel_cfg and 'max_range_km' not in sel_cfg:
        raise ConfigError(
            "Config uses deprecated 'selection.max_range_nm'. Please rename to 'selection.max_range_km' (in kilometers).")

    # --- Path Expansions and Directory Creation ---
    log_dir_rel = CONFIG.get('logging', {}).get('log_dir', 'logs')
    log_dir_abs = os.path.abspath(os.path.join(os.path.dirname(CONFIG_FILE), log_dir_rel))
    os.makedirs(log_dir_abs, exist_ok=True)
    os.makedirs(os.path.join(log_dir_abs, 'images'), exist_ok=True)

    # Update config with absolute path for log_dir
    CONFIG['logging']['log_dir'] = log_dir_abs

    # Expand ADS-B JSON file path relative to config file
    adsb_json_path_rel = CONFIG.get('adsb', {}).get('json_file_path', 'logs/aircraft.json')
    adsb_json_path_abs = os.path.abspath(os.path.join(os.path.dirname(CONFIG_FILE), adsb_json_path_rel))
    CONFIG['adsb']['json_file_path'] = adsb_json_path_abs
    os.makedirs(os.path.dirname(adsb_json_path_abs), exist_ok=True)

    # Warn on deprecated guiding key
    guide_cfg = CONFIG.get('capture', {}).get('guiding', {})
    if 'pulse_duration_ms' in guide_cfg:
        logging.warning(
            "Config key 'capture.guiding.pulse_duration_ms' is deprecated and not used. "
            "Use proportional_gain / min_pulse_ms / max_pulse_ms instead.")

    # Export absolute log directory for convenience
    LOG_DIR: str = log_dir_abs

except ConfigError as e:
    logging.critical(f"FATAL: {e}")
    sys.exit(1)
except Exception as e:
    logging.critical(f"FATAL: An unexpected error occurred while loading the configuration: {e}")
    sys.exit(1)
