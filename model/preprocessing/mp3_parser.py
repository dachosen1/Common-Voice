import logging
import os
import warnings

import numpy as np
import pandas as pd
from pydub import AudioSegment

from model.config import config
from utlis import envelope, audio_mfcc

warnings.filterwarnings("ignore")

_logger = logging.getLogger("model")


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


def remove_silence(*, signal, sample_rate, threshold):
    """
    strip out dead audio space
    :param signal: Audio sample signal
    :param sample_rate: Audio Sample rate
    :param threshold: silence threshold
    :return:

    """

    signal = signal[np.abs(signal) > threshold]
    wrap = envelope(
        y=signal, signal_rate=sample_rate, threshold=config.FRAME["MASK_THRESHOLD"]
    )
    signal = signal[wrap]
    return signal


class Mp3parser:
    def __init__(
            self, data_path, clips_dir, document_path, data_label="gender",
    ):
        """

        """
        self.hop_length = 512
        self.data_path = data_path
        self.clips_dir = clips_dir
        self.data_label = data_label
        self.label_data = data_labels(self.data_path, label=self.data_label)
        self.document_path = document_path

        check_dir(os.path.join(self.document_path, self.data_label))

    def convert_to_wav(self, clips_name: set) -> None:
        path = os.path.join(self.clips_dir, clips_name)
        sample_length_in_seconds = 1

        try:
            audio_mp3 = AudioSegment.from_mp3(file=path).set_frame_rate(
                frame_rate=config.FRAME["SAMPLE_RATE"]
            )
            signal = (
                    np.array(audio_mp3.normalize().get_array_of_samples(), dtype="int32")
                    / 100000
            )

            duration = len(signal) // config.FRAME["SAMPLE_RATE"]

            # Strip out moments of silence
            signal = remove_silence(
                signal=signal, sample_rate=config.FRAME["SAMPLE_RATE"], threshold=0.02
            )

            start = 0
            step = int(sample_length_in_seconds * config.FRAME["SAMPLE_RATE"])

            for i in range(1, duration + 1):
                data = signal[start: start + step]

                melspectrogram_DB = audio_mfcc(data)

                assert melspectrogram_DB.shape[0] == config.MODEL_PARAM["INPUT_SIZE"]

                clip_name = "{}".format(clips_name.split(".")[0])
                label_name = self.label_data[self.label_data.name == clip_name][
                    self.data_label
                ].values[0]
                train_test_choice = np.random.choice(
                    ["train_data", "val_data", "test_data"], p=[0.7, 0.2, 0.1]
                )
                dir_path = os.path.join(
                    self.document_path, self.data_label, train_test_choice
                )
                check_dir(dir_path)
                dir_path = os.path.join(
                    self.document_path, self.data_label, train_test_choice, label_name
                )
                check_dir(dir_path)
                save_path = os.path.join(dir_path, clip_name + "_" + str(i) + ".csv")
                np.savetxt(save_path, melspectrogram_DB.T, delimiter=",")
                start = step * i

        except IndexError:
            _logger.info(
                " The {} label for {} is NA ".format(self.data_label, clip_name)
            )

        except ValueError:
            _logger.info(" The MP3 for {} is too short".format(clip_name))

        except RuntimeError:
            _logger.info("The MP3 for {} is corrupt, can't open it".format(clip_name))
