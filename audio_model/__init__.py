import logging
import os

from audio_model.audio_model.config import logging_config


logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging_config.get_console_handler())
# logger.addHandler(logging.FileHandler('common-voice.log'))
logger.propagate = False


with open(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "VERSION.txt")
) as version_file:
    __version__ = version_file.read().strip()


