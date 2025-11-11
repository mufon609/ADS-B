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

import os
import sys
import yaml


class ConfigError(RuntimeError):
    """Raised when the configuration file cannot be loaded or is invalid."""


def _expand_path(p: str, base: str | None = None) -> str:
    """Expand ~ and $VARS, and make absolute (relative to ``base`` if provided)."""
    if p is None:
        return None
    p = os.path.expandvars(os.path.expanduser(str(p)))
    if base and not os.path.isabs(p):
        p = os.path.join(base, p)
    return os.path.abspath(p)


def load_config(path: str | None = None) -> dict:
    """Loads the YAML configuration file and returns it as a dict."""
    # Environment override takes precedence
    env_path = os.environ.get("ADSB_CONFIG_FILE")
    if env_path:
        path = env_path

    if path is None:
        # Default to config.yaml next to this file
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")

    try:
        with open(path, "r") as f:
            cfg = yaml.safe_load(f) or {}
        if not isinstance(cfg, dict):
            raise ConfigError("Top‑level YAML must be a mapping (dict).")
        return cfg
    except (FileNotFoundError, yaml.YAMLError) as e:
        raise ConfigError(f"Could not load configuration file '{path}': {e}") from e


try:
    # Load the configuration
    CONFIG: dict = load_config()

    # Project root is this module's directory
    ROOT = os.path.dirname(os.path.abspath(__file__))
    CONFIG['project_root'] = ROOT

    # Ensure logging section exists and expand log directory to absolute path
    CONFIG.setdefault('logging', {})
    log_dir_raw = CONFIG['logging'].get('log_dir', 'logs')
    log_dir_abs = _expand_path(log_dir_raw, base=ROOT)
    CONFIG['logging']['log_dir_abs'] = log_dir_abs

    # Ensure ADS‑B JSON file path is absolute
    adsb_cfg = CONFIG.setdefault('adsb', {})
    json_path = adsb_cfg.get('json_file_path')
    if not json_path:
        raise ConfigError("Missing required config: adsb.json_file_path")
    adsb_cfg['json_file_path'] = _expand_path(json_path, base=ROOT)

    # Provide default for selection.max_range_nm if missing
    selection_cfg = CONFIG.setdefault('selection', {})
    if 'max_range_nm' not in selection_cfg:
        selection_cfg['max_range_nm'] = 100

    # Create log directories up front
    for d in (log_dir_abs, os.path.join(log_dir_abs, 'images')):
        try:
            os.makedirs(d, exist_ok=True)
        except OSError as e:
            raise ConfigError(f"Unable to create log directory '{d}': {e}") from e

    # Export absolute log directory for convenience
    LOG_DIR: str = log_dir_abs

    # Optionally print the Bonus flag for debugging
    print(f"DEBUG: Bonus visibility change: {CONFIG.get('selection', {}).get('Bonus')}")

except ConfigError as e:
    print(f"FATAL: {e}")
    sys.exit(1)