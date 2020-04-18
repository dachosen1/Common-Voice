from __future__ import unicode_literals

import itertools
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy
import numpy as np
import pandas as pd
import torch
from pydub import AudioSegment


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


def envelope(*, y: object, signal_rate: object, threshold: object):
    signal_clean = []
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(
        window=int(signal_rate / 1000), min_periods=1, center=True
    ).mean()

    for mean in y_mean:
        if mean > threshold:
            signal_clean.append(True)
        else:
            signal_clean.append(False)
    return signal_clean


def clean_data(filename, rate, signal, new_file_path):
    seq_count = signal.shape[0] // rate
    flatten_data = signal[0 : seq_count * rate].reshape(seq_count, -1)
    flatten_data = pd.DataFrame(flatten_data)
    for row in range(flatten_data.shape[0]):
        flatten_data.iloc[row].to_csv(
            "{}{}_{}.csv".format(new_file_path, filename, row),
            header=False,
            index=False,
        )


def convert_mp3_to_wav():
    """
    Converts all mp3  in the clips folder to wav and removes them from the folder
    :return: None
    """

    for mp3 in os.listdir():
        path = os.path.join(os.getcwd(), mp3)
        file = AudioSegment.from_mp3(mp3)
        new_file_name = "{}.wav".format(mp3.split(".")[0])
        wav_path = os.path.join(os.getcwd(), new_file_name)
        file.export(wav_path, format="wav")
        os.remove(path)


def calc_fft(*, y, rate):
    n = len(y)
    freq = np.fft.rfftfreq(n, d=1 / rate)
    Y = abs(np.fft.rfft(y) / n)

    return Y, freq


def plot_confusion_matrix(
    cm: numpy.ndarray, class_names: list()
) -> matplotlib.figure.Figure:
    """
  Returns a matplotlib figure containing the plotted confusion matrix.

  Args:
    cm (array, shape = [n, n]): a confusion matrix of integer classes
    class_names (array, shape = [n]): String names of the integer classes
  """
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype("float") / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    return figure
