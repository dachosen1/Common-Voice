import logging
import os
import warnings

import librosa
import numpy as np
import pandas as pd

from audio_model.audio_model.config.config import FRAME, DO_NOT_INCLUDE
from audio_model.audio_model.utils import audio_melspectrogram, remove_silence

warnings.filterwarnings("ignore")

_logger = logging.getLogger("audio_model")


def check_dir(path):
    if os.path.exists(path) is False:
        os.mkdir(path)


def data_labels(data_path, label):
    """
    Filters files with the appropriate label from the data csv file
    :param data_path:
    :param label: data label default values is gender: possible values includes [gender, age, country]
    :return:
    """
    label_data = pd.read_csv(os.path.join(data_path, "data.csv"))
    label_data = label_data[label_data[label].notna()]
    label_data = label_data[label_data[label] != "other"]
    return label_data



class Mp3parser:
    def __init__(self, data_path, clips_dir, document_path, data_label, model):
        """

        """
        self.data_path = data_path
        self.clips_dir = clips_dir
        self.data_label = data_label
        self.label_data = data_labels(self.data_path, label=self.data_label)
        self.document_path = document_path
        self.model = model

        check_dir(os.path.join(self.document_path, self.data_label))

        self.remove_count = 0
        self.add_count = 0
        self.SAMPLE_RATE = FRAME['SAMPLE_RATE']

    def convert_to_wav(self, index) -> None:
        clips_name = self.label_data.path.values[index]
        path = os.path.join(self.clips_dir, clips_name)
        sample_length_in_seconds = 1

        try:
            signal, _ = librosa.load(path=path, sr=self.SAMPLE_RATE)
            signal = remove_silence(signal=signal)

            duration = len(signal) // self.SAMPLE_RATE

            start = 0
            step = int(sample_length_in_seconds * self.SAMPLE_RATE)

            for i in range(1, duration + 1):
                data = signal[start: start + step]

                training_mfcc = audio_melspectrogram(data)

                assert training_mfcc.shape[0] == self.model.PARAM["INPUT_SIZE"]
                assert training_mfcc.shape[1] == 32

                clip_name = "{}".format(clips_name.split(".")[0])
                label_name = self.label_data[self.label_data.path == clips_name][self.data_label].values[0]

                if label_name in DO_NOT_INCLUDE:
                    break

                train_test_choice = np.random.choice(
                    ["train_data", "val_data", "test_data"], p=[0.7, 0.2, 0.1]
                )

                dir_path = os.path.join(self.document_path, self.data_label, train_test_choice)
                check_dir(dir_path)

                dir_path = os.path.join(self.document_path, self.data_label, train_test_choice, label_name)

                check_dir(dir_path)
                save_path = os.path.join(dir_path, clip_name + "_" + str(i))
                np.save(save_path, training_mfcc.T)
                start = self.SAMPLE_RATE * i
                self.add_count += 1

        except FileNotFoundError:
            _logger.info("Can't find the file {}".format(clips_name))
            self.remove_count += 1

        except FileExistsError:
            _logger.warning("Error in creating folder that's already created")
