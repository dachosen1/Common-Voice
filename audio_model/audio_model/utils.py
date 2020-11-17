from __future__ import unicode_literals

import concurrent.futures
import itertools
import logging
import os
import shutil
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import wandb
from librosa import power_to_db
from librosa import stft
from pydub import AudioSegment
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)
from torch.utils.data import WeightedRandomSampler
from tqdm import tqdm

from audio_model.audio_model.config.config import FRAME, DataDirectory

_logger = logging.getLogger("audio_model")

warnings.filterwarnings("ignore")
import librosa


def npy_loader(path: str) -> torch.Tensor:
    """
    :param path:
    :return:
    """
    data = np.array(np.load(path))
    sample = torch.from_numpy(data)
    return sample


def mp3_loader(path):
    file = AudioSegment.from_mp3(path)
    return file


def csv_loader(path: str) -> torch.Tensor:
    """
    :param path:
    :return:
    """
    data = np.array(pd.read_csv(path, header=None))
    sample = torch.from_numpy(data)
    return sample

def envelope(y: int, signal_rate: int, threshold: float):
    signal_clean = []
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(
        window=int(signal_rate / 10), min_periods=1, center=True
    ).mean()

    for mean in y_mean:
        if mean > threshold:
            signal_clean.append(True)
        else:
            signal_clean.append(False)
    return signal_clean


def remove_un_label_files(clips_names: list) -> None:
    """
    Remove a list of list of files that do not contain any labels
    :param clips_names: list of of mp3 names

    """
    data = pd.read_csv("Development/data.csv")
    data_path = set(data.path)
    clips_path = DataDirectory.CLIPS_DIR

    delete_path = r"C:\Users\ander\Documents\delete"

    for mp3 in tqdm(clips_names):
        if mp3 not in data_path:
            shutil.move(os.path.join(clips_path, mp3), os.path.join(delete_path, mp3))


def calc_fft(*, y, rate):
    n = len(y)
    freq = np.fft.rfftfreq(n, d=1 / rate)
    Y = abs(np.fft.rfft(y) / n)

    return Y, freq


def plot_confusion_matrix(
        cm: np.ndarray, class_names: list
) -> matplotlib.figure.Figure:
    """
    Generates a Matplotlib figure containing the plotted confusion matrix.

    :param cm: cm (array, shape = [n, n]): a confusion matrix of integer classes
    :param class_names: class_names (array, shape = [n]): String names of the integer classes
    :return:
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


def sample_weight(data_folder):
    """
    Return sample weight for stratistifed random sampling
    :param data_folder: Dataset folder object
    :return:
    """
    class_sample_count = np.array(
        [len([i for i in data_folder.targets if i == t]) for t in range(0, len(data_folder.classes))])
    weight = 1 / class_sample_count
    samples_weight = np.array([weight[t] for t in data_folder.targets])
    samples_weight = torch.from_numpy(samples_weight)
    samples_weight = samples_weight.double()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    return sampler


def envelop_mask(signal):
    mask = envelope(signal, signal_rate=FRAME['SAMPLE_RATE'], threshold=FRAME['MASK_THRESHOLD'])
    return signal[mask]


def audio_melspectrogram(signal):
    specto = librosa.feature.melspectrogram(y=signal, sr=FRAME['SAMPLE_RATE'], n_mels=FRAME['N_MELS'], fmax=FRAME['FMAX'])
    spec_to_db = power_to_db(specto, ref=np.max)
    return spec_to_db


def audio_sfft(signal):
    sftf_signal = np.abs(stft(signal))
    spec_to_db = power_to_db(sftf_signal, ref=np.max)
    return normalize(spec_to_db)


def audio_mfcc(signal):
    signal_mfcc_ = librosa.feature.mfcc(signal, sr=FRAME['SAMPLE_RATE'], n_mfcc=FRAME['NUMCEP'])
    return normalize(signal_mfcc_.T)


def generate_pred(mel, model, label, model_name):
    """
    Generates audio prediction and label
    :param model_name:
    :param mel: decibel (dB) units
    :param model: torch audio_model
    :param label: label dictionary
    :return: prints prediction label and probability
    """
    mel = torch.from_numpy(mel).reshape(1, -1, model_name.PARAM["INPUT_SIZE"]).float()

    if torch.cuda.is_available():
        model.cuda()
        mel = mel.cuda()

    out = model(mel)
    prob = torch.topk(out, k=1).values
    pred = torch.topk(out, k=1).indices
    label_name = label[int(pred.cpu().data.numpy())]

    _logger.info(
        "Prediction: {}, Probability: {}".format(
            label_name, round(float(prob.flatten()[0]), 5)
        )
    )

    return label_name, round(float(prob.flatten()[0]), 5)


def _metric_summary(pred: np.ndarray, label: np.ndarray):
    acc = accuracy_score(y_true=label, y_pred=pred)
    f1 = f1_score(y_true=label, y_pred=pred)
    pc = precision_score(y_true=label, y_pred=pred)
    rs = recall_score(y_true=label, y_pred=pred)
    return acc, f1, pc, rs

def log_scalar(name, value, step):
    """Log a scalar value to both MLflow and TensorBoard"""
    wandb.log({name: value}, step=step)


max_workers = concurrent.futures.ProcessPoolExecutor()._max_workers


def run_thread_pool(function, my_iter):
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        tqdm(executor.map(function, my_iter), total=len(my_iter))


def run_process_pool(function, my_iter):
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        tqdm(executor.map(function, my_iter), total=len(my_iter))


def linear_to_mel(spectrogram):
    return librosa.feature.melspectrogram(
        S=spectrogram, sr=FRAME['SAMPLE_RATE'], n_fft=FRAME['NFFT'], n_mels=FRAME['N_MELS'], fmin=FRAME['TOP_DB'])


'''
def build_mel_basis():
    return librosa.filters.mel(hp.sample_rate, hp.n_fft, n_mels=hp.num_mels, fmin=hp.fmin)
'''


def normalize(signal):
    mean = np.mean(signal)
    sd = np.std(signal)
    signal_normalized = (signal - mean) / sd
    return signal_normalized


def amp_to_db(x):
    return 20 * np.log10(np.maximum(1e-5, x))


def db_to_amp(x):
    return np.power(10.0, x * 0.05)


def spectrogram(y):
    D = stft(y)
    S = amp_to_db(np.abs(D)) - FRAME['REF_LEVEL_DB']
    return normalize(S)


def melspectrogram(y):
    D = stft(y)
    S = amp_to_db(linear_to_mel(np.abs(D)))
    return normalize(S)

def remove_silence(*, signal, sample_rate, threshold):
    """
    strip out dead audio space
    :param signal: Audio sample signal
    :param sample_rate: Audio Sample rate
    :param threshold: silence threshold
    :return:
    """

    signal = signal[np.abs(signal) > threshold]
    wrap = envelope(y=signal, signal_rate=sample_rate, threshold=FRAME['MASK_THRESHOLD'])
    signal = signal[wrap]
    return signal


def stft(y):
    return librosa.stft(
        y=y,
        n_fft=FRAME['NFFT'], hop_length=FRAME['HOP_LENGTH'], win_length=FRAME['WIN_LENGTH'])

