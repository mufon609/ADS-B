# logger_config.py
"""
Centralized logging configuration for the application.
"""
import logging
import sys
from logging.handlers import RotatingFileHandler
import os
from config_loader import CONFIG, LOG_DIR

def setup_logging():
    """
    Configures the root logger for the application.
    """
    log_cfg = CONFIG.get('logging', {})
    log_level_str = log_cfg.get('level', 'INFO').upper()
    log_level = getattr(logging, log_level_str, logging.INFO)

    # Create a detailed format for file logging
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create a simplified format for console output
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )

    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Remove any existing handlers to avoid duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File Handler
    log_file_path = os.path.join(LOG_DIR, 'gemini.log')
    file_handler = RotatingFileHandler(
        log_file_path,
        maxBytes=log_cfg.get('log_max_size_mb', 25) * 1024 * 1024,
        backupCount=log_cfg.get('log_backup_count', 5)
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Redirect uncaught exceptions to the logger
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

    sys.excepthook = handle_exception
