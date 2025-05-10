import logging
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime

def setup_logger(name, log_file='creatorguard.log'):
    """Set up a logger with both file and console handlers."""
    # Create logs directory if it doesn't exist
    log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s'
    )

    # File handler (rotating to keep log size manageable)
    file_handler = RotatingFileHandler(
        log_path, 
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def log_error(logger, error, context=None):
    """Log an error with optional context."""
    error_msg = f"Error: {str(error)}"
    if context:
        error_msg = f"{error_msg}\nContext: {context}"
    logger.error(error_msg, exc_info=True)

def log_warning(logger, message, context=None):
    """Log a warning with optional context."""
    warning_msg = message
    if context:
        warning_msg = f"{warning_msg}\nContext: {context}"
    logger.warning(warning_msg)

def log_info(logger, message):
    """Log an info message."""
    logger.info(message)
    return message
