import os

import numpy as np
import pandas as pd
from pydub import AudioSegment
from python_speech_features import mfcc

import librosa
from model.config import config
from utlis import envelope


import warnings

warnings.filterwarnings("ignore")

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
            mel_seq_count=512,
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
        self.mel_seq_count = mel_seq_count
        self.timeseries_length = 128

    def convert_to_wav(self, clips_name: set) -> None:
        path = os.path.join(self.clips_dir, clips_name)

        try:
            y, sr = librosa.load(path)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=self.hop_length, n_mfcc=13)

            spectral_center = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=self.hop_length)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=self.hop_length)
            spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=self.hop_length)
            data = np.zeros((self.timeseries_length, 33), dtype=np.float64)
            data[:, 0:13] = mfcc.T[0:self.timeseries_length, :]
            data[:, 13:14] = spectral_center.T[0:self.timeseries_length, :]
            data[:, 14:26] = chroma.T[0:self.timeseries_length, :]
            data[:, 26:33] = spectral_contrast.T[0:self.timeseries_length, :]

            clip_name = f'{clips_name.split(".")[0]}'
            label_name = self.label_data[self.label_data.name == clip_name][self.data_label].values[0]
            train_test_choice = np.random.choice(["train_data", "val_data", "test_data"], p=[0.7, 0.2, 0.1])
            save_path = os.path.join(self.document_path, "gender", train_test_choice, label_name, clip_name + '.csv')
            np.savetxt(save_path, data, delimiter=',')

        except IndexError:
            print(" The label for {} is NA "
                  ".......".format(clip_name))
        except ValueError:
            print(f" The MP3 for {clip_name} is too short")
