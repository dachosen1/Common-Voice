import os
import warnings

import librosa
import numpy as np
import pandas as pd
import logging
from model.config import config

warnings.filterwarnings("ignore")

_logger = logging.getLogger(__name__)

def _set_frame_rate(wav_file, frame_rate=config.FRAME['FRAME_RATE']):
    wav_file.set_frame_rate(frame_rate=frame_rate)
    return wav_file


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


class MP3_Parser:
    def __init__(
            self,
            data_path,
            clips_dir,
            document_path,
            data_label="gender",
    ):
        """

        """
        self.hop_length = 512
        self.data_path = data_path
        self.clips_dir = clips_dir
        self.data_label = data_label
        self.label_data = data_labels(self.data_path, label=self.data_label)
        self.document_path = document_path

    def convert_to_mfcc(self, clips_name: set) -> None:
        path = os.path.join(self.clips_dir, clips_name)

        sample_length_in_seconds = 1

        try:
            signal, sample_rate = librosa.load(path)

            # Strip out moments of silence
            signal = signal[np.abs(signal) > 0.02]

            duration = len(signal) // sample_rate
            start = 0
            step = int(sample_length_in_seconds * sample_rate)

            for i in range(1, duration+1):
                mfcc = librosa.feature.mfcc(y=signal[start:start+step], sr=sample_rate, hop_length=self.hop_length,
                                            n_mfcc=13)

                assert mfcc.shape[1] == config.MODEL_PARAM['INPUT_SIZE']

                clip_name = f'{clips_name.split(".")[0]}'
                label_name = self.label_data[self.label_data.name == clip_name][self.data_label].values[0]
                train_test_choice = np.random.choice(["train_data", "val_data", "test_data"], p=[0.7, 0.2, 0.1])
                save_path = os.path.join(self.document_path, "gender", train_test_choice, label_name, clip_name + '_'
                                         + str(i) + '.csv')
                np.savetxt(save_path, mfcc, delimiter=',')
                start = step * i

        except IndexError:
            _logger.info(f" The label for {clip_name} is NA ")

        except ValueError:
            _logger.info(f" The MP3 for {clip_name} is too short")

        except RuntimeError:
            _logger.info(f" The MP3 for {clip_name} is corrupt, can't open it")
