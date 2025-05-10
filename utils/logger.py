"""Logging utility functions."""

import logging
import traceback
import sys
from datetime import datetime
import os

# Create logs directory if it doesn't exist
logs_dir = 'logs'
os.makedirs(logs_dir, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(logs_dir, 'creatorguard.log')),
        logging.StreamHandler(sys.stdout)
    ]
)

def get_logger(name):
    """Get a logger instance with the given name."""
    logger = logging.getLogger(name)
    return logger

def log_error(logger=None, error=None, message="An error occurred"):
    """Log an error with stack trace."""
    if logger is None:
        logger = logging.getLogger('creatorguard')
    
    error_msg = f"{message}: {str(error)}" if error else message
    logger.error(error_msg)
    
    if error:
        logger.error("Stack trace:")
        logger.error(traceback.format_exc())

def log_info(logger=None, message=""):
    """Log an informational message."""
    if logger is None:
        logger = logging.getLogger('creatorguard')
    
    logger.info(message)
