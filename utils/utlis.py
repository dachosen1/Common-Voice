from __future__ import unicode_literals

import os

import numpy as np
import pandas as pd
import torch
from pydub import AudioSegment
from scipy.io import wavfile


def csv_loader(path: str) -> torch.Tensor:
    """

    :param path:
    :return:
    """
    data = np.array(pd.read_csv(path, header=None))

    sample = torch.from_numpy(data)
    return sample


def mp3_loader(path):
    file = AudioSegment.from_mp3(path)
    return file


def envelope(*, y, rate, threshold):
    mask = []

    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate / 10), min_periods=1, center=True).mean()

    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)

    return mask


def clean_data(filename, rate, signal, new_file_path):
    seq_count = signal.shape[0] // rate

    flatten_data = signal[0 : seq_count * rate].reshape(seq_count, -1)

    flatten_data = pd.DataFrame(flatten_data)

    for row in range(flatten_data.shape[0]):
        flatten_data.iloc[row].to_csv(
            f"{new_file_path}{filename}_{row}.csv", header=False, index=False
        )


def convert_mp3_to_wav(mp3):
    file = AudioSegment.from_mp3(mp3)


def convert_mp3_to_wav():
    """
    Converts all mp3  in the clips folder to wav and removes them from the folder
    :return: None
    """

    for mp3 in os.listdir():
        path = os.path.join(os.getcwd(), mp3)
        file = AudioSegment.from_mp3(mp3)
        new_file_name = f'{mp3.split(".")[0]}.wav'
        wav_path = os.path.join(os.getcwd(), new_file_name)
        file.export(wav_path, format="wav")
        os.remove(path)

    # todo: needs to be dynamic: search for director for mp3 files also function needs to save document in s3
    # todo: Add the option to specify a path
    # todo: add the options to upload model to the cloud


class Wav_parse:
    def __init__(self, path):
        self.path = path
        self.wav = []  # todo: convert path to wav

        self.rate, self.signal = wavfile.read(path)

        pass

    def play_wav_file(self):
        # todo: function to be able to play wav name
        pass

    def remove_background_noise(self):
        # todo: function to remove any background noise
        pass

    def voice_window(self, window):
        # todo: sliding window that isolates a window in a time of voice
        pass

    def normalize_voice(self):
        # todo: function to normalize voice pace
        pass

    def visualize_wav(self):
        # todo: visualize differnt voice patterns
        pass

    def voice_content(self):
        # todo: function to fetch the meta data for wav name.
        pass


class Wav_model:
    """
    Module to format get wav ready for modeling

    """

    def format_wav(self):
        # todo: function to format WAV files to for modeling. Maaybe a matrix? do more research on it
        pass

    def voice_batch(self):
        # todo: select a batch of voices to train RNN on
        pass


def calc_fft(*, y, rate):
    n = len(y)
    freq = np.fft.rfftfreq(n, d=1 / rate)
    Y = abs(np.fft.rfft(y) / n)

    return Y, freq


if __name__ == "__main__":
    convert_mp3_to_wav()
    AudioSegment.ffmpeg = r"C:\Users\ander\Anaconda3\Lib\ffmpeg\bin"
