#config_loader.py
"""
Loads and provides the application configuration from config.yaml.
"""
import os
import sys
import yaml

class ConfigError(RuntimeError):
    """Custom exception for configuration errors."""
    pass

def _expand_path(p: str, base: str = None) -> str:
    """Expand ~ and $VARS, and make absolute (relative to base if provided)."""
    if p is None:
        return None
    p = os.path.expandvars(os.path.expanduser(str(p)))
    if base and not os.path.isabs(p):
        p = os.path.join(base, p)
    return os.path.abspath(p)

def load_config(path: str = None) -> dict:
    """Loads the YAML configuration file (anchored to this module by default)."""
    env_path = os.environ.get("ADSB_CONFIG_FILE")
    if env_path:
        path = env_path

    try:
        if path is None:
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml')
        with open(path, 'r') as f:
            cfg = yaml.safe_load(f)
        if cfg is None:
            cfg = {}
        if not isinstance(cfg, dict):
            raise ConfigError("Top-level YAML must be a mapping (dict).")
        return cfg
    except (FileNotFoundError, yaml.YAMLError) as e:
        raise ConfigError(f"Could not load configuration file '{path}': {e}") from e

try:
    CONFIG = load_config()

    # --- Define and inject absolute paths ---

    # Project root
    ROOT = os.path.dirname(os.path.abspath(__file__))
    CONFIG['project_root'] = ROOT

    # Logging section + absolute log dir
    CONFIG.setdefault('logging', {})
    log_dir_raw = CONFIG['logging'].get('log_dir', 'logs')
    log_dir_abs = _expand_path(log_dir_raw, base=ROOT)
    CONFIG['logging']['log_dir_abs'] = log_dir_abs

    # ADS-B file path normalization (fail fast if missing)
    adsb_cfg = CONFIG.setdefault('adsb', {})
    json_path = adsb_cfg.get('json_file_path')
    if not json_path:
        raise ConfigError("Missing required config: adsb.json_file_path")
    adsb_cfg['json_file_path'] = _expand_path(json_path, base=ROOT)

    # Ensure 'max_range_nm' is present for consistency
    selection_cfg = CONFIG.setdefault('selection', {})
    if 'max_range_nm' not in selection_cfg:
        selection_cfg['max_range_nm'] = 100  # Default value in nautical miles

    # Create the log directories (with clear error if not writable)
    for d in (CONFIG['logging']['log_dir_abs'], os.path.join(CONFIG['logging']['log_dir_abs'], 'images')):
        try:
            os.makedirs(d, exist_ok=True)
        except OSError as e:
            raise ConfigError(f"Unable to create log directory '{d}': {e}") from e

    # Export absolute log dir for convenience
    LOG_DIR = CONFIG['logging']['log_dir_abs']

    print(f"DEBUG: Bonus visibility change: {CONFIG.get('selection', {}).get('Bonus')}")

except ConfigError as e:
    print(f"FATAL: {e}")
    sys.exit(1)
