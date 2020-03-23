from __future__ import unicode_literals

import os

import numpy as np
import pandas as pd
import torch
import youtube_dl
from matplotlib import pyplot as plt
from pydub import AudioSegment
from scipy.io import wavfile

AudioSegment.ffmpeg = r"C:\Users\ander\Anaconda3\Lib\ffmpeg\bin"


def csv_loader(path):
    data = np.array(pd.read_csv(path))

    sample = torch.from_numpy(data)
    return sample


def envelope(*, y, rate, threshold):
    mask = []

    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window = int(rate / 10), min_periods = 1, center = True).mean()

    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)

    return mask


def clean_data(filename, rate, signal, new_file_path):
    seq_count = signal.shape[0] // rate

    flatten_data = signal[0: seq_count * rate].reshape(seq_count, -1)

    flatten_data = pd.DataFrame(flatten_data)

    for row in range(flatten_data.shape[0]):
        flatten_data.iloc[row].to_csv(
            f"{new_file_path}{filename}_{row}.csv", header = False, index = False
        )


# todo: add all data to an s3 bucket


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
        file.export(wav_path, format = "wav")
        os.remove(path)

    # todo: needs to be dynamic: search for director for mp3 files also function needs to save document in s3
    # todo: Add the option to specify a path
    # todo: add the options to upload RNN_Model to the cloud


class Wav_parse:
    def __init__(self, path):
        self.path = path
        self.wav = []  # todo: convert path to wav

        self.rate, self.signal = wavfile.read(path)

        pass

    def play_wav_file(self):
        # todo: function to be able to play wav file
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
        # todo: function to fetch the meta data for wav file.
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
    freq = np.fft.rfftfreq(n, d = 1 / rate)
    Y = abs(np.fft.rfft(y) / n)

    return Y, freq


def plot_signals(signals):
    fig, axes = plt.subplots(
        nrows = len(signals) // 5, ncols = 5, sharex = False, sharey = True, figsize = (20, 5)
    )
    fig.suptitle("Time Series", size = 16)
    i = 0
    for x in range(len(signals) // 5):
        for y in range(5):
            axes[x, y].set_title(list(signals.keys())[i])
            axes[x, y].plot(list(signals.values())[i])
            axes[x, y].get_xaxis().set_visible(False)
            axes[x, y].get_yaxis().set_visible(False)
            i += 1


def plot_fft(signals, fft):
    fig, axes = plt.subplots(
        nrows = len(signals) // 5, ncols = 5, sharex = False, sharey = True, figsize = (20, 5)
    )
    fig.suptitle("Fourier Transforms", size = 16)
    i = 0
    for x in range(len(signals) // 5):
        for y in range(2):
            data = list(fft.values())[i]
            Y, freq = data[0], data[1]
            axes[x, y].set_title(list(fft.keys())[i])
            axes[x, y].plot(freq, Y)
            axes[x, y].get_xaxis().set_visible(False)
            axes[x, y].get_yaxis().set_visible(False)
            i += 1


def plot_fbank(signals, fbank):
    fig, axes = plt.subplots(
        nrows = len(signals) // 5, ncols = 5, sharex = False, sharey = True, figsize = (20, 5)
    )
    fig.suptitle("Filter Bank Coefficients", size = 16)
    i = 0
    for x in range(len(signals) // 5):
        for y in range(5):
            axes[x, y].set_title(list(fbank.keys())[i])
            axes[x, y].imshow(
                list(fbank.values())[i], cmap = "hot", interpolation = "nearest"
            )
            axes[x, y].get_xaxis().set_visible(False)
            axes[x, y].get_yaxis().set_visible(False)
            i += 1


def plot_mfccs(signals, mfccs):
    fig, axes = plt.subplots(
        nrows = len(signals) // 5, ncols = 5, sharex = False, sharey = True, figsize = (20, 5)
    )
    fig.suptitle("Mel Frequency Cepstrum Coefficients", size = 16)
    i = 0
    for x in range(2):
        for y in range(5):
            axes[x, y].set_title(list(mfccs.keys())[i])
            axes[x, y].imshow(
                list(mfccs.values())[i], cmap = "hot", interpolation = "nearest"
            )
            axes[x, y].get_xaxis().set_visible(False)
            axes[x, y].get_yaxis().set_visible(False)
            i += 1


ydl_opts = {}
with youtube_dl.YoutubeDL(ydl_opts) as ydl:
    ydl.download(["https://www.youtube.com/watch?v=BaW_jenozKc"])

if __name__ == "__main__":
    convert_mp3_to_wav()
