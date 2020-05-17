import os

import numpy as np
import pandas as pd
from pydub import AudioSegment
from python_speech_features import mfcc

from model.config import config
from utlis import envelope


def _set_frame_rate(wav_file, frame_rate=config.FRAME_RATE):
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
        self.data_path = data_path
        self.clips_dir = clips_dir
        self.data_label = data_label
        self.label_data = data_labels(self.data_path, label=self.data_label)
        self.document_path = document_path
        self.mel_seq_count = mel_seq_count

    def convert_to_wav(self, clips_name: set) -> None:
        path = os.path.join(self.clips_dir, clips_name)
        wav = AudioSegment.from_mp3(path)
        clip_name = f'{clips_name.split(".")[0]}'
        signal = np.array(wav.get_array_of_samples())
        rate = wav.frame_rate
        try:
            label_name = self.label_data[self.label_data.name == clip_name][
                self.data_label
            ].values[0]
            mask = envelope(y=signal, signal_rate=rate, threshold=100)
            signal = signal[mask]
            mel = mfcc(signal, rate, numcep=13, nfilt=128, nfft=1500).flatten()
            self.clean_data(
                filename=clip_name,
                rate=self.mel_seq_count,
                signal=mel,
                label=label_name,
            )
        except IndexError:
            print(" The label for {} is NA   "
                  ".......".format(clip_name))

    def clean_data(
            self, filename: str, rate: int, signal: np.ndarray, label: object
    ) -> None:
        """
        Split an audio signal into windows
        :param filename:
        :param rate:
        :param signal:
        :param label:
        :return:
        """
        seq_count = signal.shape[0] // self.mel_seq_count

        try:
            flatten_data = signal[0: seq_count * rate].reshape(seq_count, -1)
            flatten_data = pd.DataFrame(flatten_data)

            for row in range(flatten_data.shape[0]):
                train_test_choice = np.random.choice(
                    ["train_data", "val_data", "test_data"], p=[0.7, 0.2, 0.1]
                )
                save_path = os.path.join(
                    self.document_path, "gender", train_test_choice, label
                )
                flatten_data.iloc[row].to_csv(
                    "{}/{}-{}.csv".format(save_path, filename, row),
                    header=False,
                    index=False,
                )
        except ValueError:
            print(" Skipped {} ......file is corrupt".format(filename))
