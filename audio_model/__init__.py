import logging
import os

from audio_model import config
from audio_model.config import logging_config

with open(os.path.join(config.PACKAGE_ROOT, "VERSION.txt")) as version_file:
    __version__ = version_file.read().strip()

from .models import AudioCNN, AudioLSTM
from audio_model import utils
from audio_model import pipeline_mananger
from audio_model import model_manager
from audio_model import preprocessing
from audio_model import trained_model

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging_config.get_console_handler())
# logger.addHandler(logging.FileHandler('common-voice.log'))
logger.propagate = False
