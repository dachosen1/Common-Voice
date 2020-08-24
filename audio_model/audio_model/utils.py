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
from librosa.feature import melspectrogram
from pydub import AudioSegment
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import WeightedRandomSampler
from tqdm import tqdm

from audio_model.audio_model.config.config import CommonVoiceModels, DataDirectory

_logger = logging.getLogger("audio_model")

warnings.filterwarnings("ignore")


def csv_loader(path: str) -> torch.Tensor:
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


def envelope(*, y: int, signal_rate: object, threshold: object):
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
    Return sample weight for stratified random sampling
    :param data_folder: Dataset folder object
    :return: WeightedRandomSampler class
    """
    class_sample_count = np.array(
        [
            len([i for i in data_folder.targets if i == t])
            for t in range(0, len(data_folder.classes))
        ]
    )
    weight = 1 / class_sample_count
    samples_weight = np.array([weight[t] for t in data_folder.targets])
    samples_weight = torch.from_numpy(samples_weight)
    samples_weight = samples_weight.double()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    return sampler


def audio_melspectrogram(signal, sample_rate=CommonVoiceModels.Frame.FRAME['SAMPLE_RATE'],
               n_mels=CommonVoiceModels.Frame.FRAME['N_MELS'],fmax=CommonVoiceModels.Frame.FRAME['FMAX']):

    specto = melspectrogram(y=signal, sr=sample_rate, n_mels=n_mels,
                            fmax=fmax)
    spec_to_db = power_to_db(specto, ref=np.max)

    return spec_to_db


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
    pc, rc, _, _ = precision_recall_fscore_support(
        y_true=label, y_pred=pred, average="weighted"
    )
    return acc, pc, rc


def log_scalar(name, value, step):
    """Log a scalar value to both MLflow and TensorBoard"""
    wandb.log({name: value}, step=step)


def run_thread_pool(function, my_iter):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        tqdm(executor.map(function, my_iter), total=len(my_iter))


def run_process_pool(function, my_iter):
    with concurrent.futures.ProcessPoolExecutor() as executor:
        tqdm(executor.map(function, my_iter), total=len(my_iter))
