import os
import logging

from model.config import logging_config

logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging_config.get_console_handler())
logger.propagate = False


with open(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "VERSION")
) as version_file:
    __version__ = version_file.read().strip()
