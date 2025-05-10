"""Logging utility functions."""

import logging
import traceback
import sys
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime

# Create logs directory if it doesn't exist
logs_dir = 'logs'
os.makedirs(logs_dir, exist_ok=True)

def setup_logger(name, log_file='creatorguard.log'):
    """Set up a logger with both file and console handlers."""
    logger = logging.getLogger(name)
    
    # If logger already has handlers, return it
    if logger.handlers:
        return logger
        
    logger.setLevel(logging.DEBUG)
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s'
    )
    
    # File handler (rotating to keep log size manageable)
    file_handler = RotatingFileHandler(
        os.path.join(logs_dir, log_file),
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

def get_logger(name):
    """Get a logger instance with the given name."""
    return setup_logger(name)

def log_error(logger=None, error=None, message="An error occurred", context=None):
    """Log an error with stack trace."""
    if logger is None:
        logger = logging.getLogger('creatorguard')
    
    error_msg = f"{message}: {str(error)}" if error else message
    if context:
        error_msg = f"{error_msg}\nContext: {context}"
    
    logger.error(error_msg)
    
    if error:
        logger.error("Stack trace:")
        logger.error(traceback.format_exc())

def log_warning(logger=None, message="", context=None):
    """Log a warning message."""
    if logger is None:
        logger = logging.getLogger('creatorguard')
    
    warning_msg = message
    if context:
        warning_msg = f"{warning_msg}\nContext: {context}"
    
    logger.warning(warning_msg)

def log_info(logger=None, message=""):
    """Log an informational message."""
    if logger is None:
        logger = logging.getLogger('creatorguard')
    
    logger.info(message)
