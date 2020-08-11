import logging
import os
import sys
from logging.handlers import TimedRotatingFileHandler

import commonvoice

PACKAGE_ROOT = os.path.dirname(commonvoice.__file__)

FORMATTER = logging.Formatter(
    "%(asctime)s — %(name)s — %(levelname)s —" "%(funcName)s:%(lineno)d — %(message)s"
)

LOG_DIR = os.path.join(PACKAGE_ROOT, "logs")

# LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, "ml_api.log")

# UPLOAD_FOLDER = PACKAGE_ROOT / 'uploads'
# UPLOAD_FOLDER.mkdir(exist_ok=True)



def get_console_handler():
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(FORMATTER)
    return console_handler


def get_file_handler():
    file_handler = TimedRotatingFileHandler(LOG_FILE, when="midnight")
    file_handler.setFormatter(FORMATTER)
    file_handler.setLevel(logging.WARNING)
    return file_handler


def get_logger(*, logger_name):
    """Get logger with prepared handlers."""

    logger = logging.getLogger(logger_name)

    logger.setLevel(logging.INFO)

    logger.addHandler(get_console_handler())
    logger.addHandler(get_file_handler())
    logger.propagate = False

    return logger


class Config:
    DEBUG = False
    TESTING = False
    CSRF_ENABLED = True
    SERVER_PORT = 5000


#    UPLOAD_FOLDER = UPLOAD_FOLDER


class DevelopmentConfig(Config):
    DEVELOPMENT = True
    DEBUG = True


class TestingConfig(Config):
    TESTING = True
