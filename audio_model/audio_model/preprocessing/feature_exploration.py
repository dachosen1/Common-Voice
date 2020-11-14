import os
import warnings

import librosa
import librosa.display
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib import pyplot as plt

from audio_model.audio_model.config.config import DO_NOT_INCLUDE

warnings.filterwarnings('ignore')

ROOT_DIR = r'C:\Users\ander\Documents\common-voice-data'
DATA = pd.read_csv(os.path.join(ROOT_DIR, 'data.csv'))
CLIPS = os.path.join(ROOT_DIR, 'clips')


def generate_label_dict(data, unique_labels, mp3_label, label_count=5, ):
    label_dict = {}

    for label in unique_labels:
        count = 0
        label_list = []
        for i in range(100):
            index = np.random.randint(0, 1000)
            if os.path.exists(os.path.join(CLIPS, data[data[mp3_label] == label]['path'].values[index])):
                label_list.append(data[data[mp3_label] == label]['path'].values[index])
                count += 1
                if count == label_count:
                    label_dict[label] = label_list
                    break

    return label_dict


def plot_signals(target_list, target_dict, n_rows=4, top_db=60):
    n_cols = len(target_list)

    fig, axs = plt.subplots(ncols=n_cols, nrows=n_rows, figsize=(25, 12))

    for count, value in enumerate(target_list):
        axs[0, count].set_title(f"Output: {value}, top DB: {top_db}", loc='center', y=1.1)

        for j in range(n_rows):
            signal, sr = librosa.load(os.path.join(CLIPS, target_dict[value][j]))
            signal, _ = librosa.effects.trim(signal, top_db=top_db)
            axs[j, count].plot(signal)

    plt.tight_layout()
    plt.show()


def feature_power_to_db(signal, sr, n_mels=128, fmax=8000, seconds=2):
    signal = signal[:sr * seconds]
    S = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=n_mels, fmax=fmax)
    S_dB = librosa.power_to_db(S, ref=np.max)
    return S_dB.T


def feature_mfcc(signal, sr, seconds):
    signal = signal[:sr * seconds]
    S = librosa.feature.mfcc(y=signal, sr=sr)
    return S


def plot_feature(audio_transformation, target_list, target_dict, n_rows=4, top_db=60, **kwargs):
    n_cols = len(target_list)

    fig, axs = plt.subplots(ncols=n_cols, nrows=n_rows, figsize=(25, 12))

    for count, value in enumerate(target_list):
        axs[0, count].set_title(f"Output: {value}, top DB: {top_db}", loc='center', y=1.1)

        for j in range(n_rows):
            signal, sr = librosa.load(os.path.join(CLIPS, target_dict[value][j]))
            signal, _ = librosa.effects.trim(signal, top_db=top_db)
            audio_feature = audio_transformation(signal=signal, **kwargs)
            axs[j, count].imshow(audio_feature, cmap=cm.inferno)

    plt.tight_layout()
    plt.show()


accent = DATA[['name', 'path', 'accent']].dropna()
unique_accent = [i for i in accent['accent'].unique() if i not in DO_NOT_INCLUDE]

accent_dict = generate_label_dict(data=accent, unique_labels=unique_accent, mp3_label='accent')

# plot_power_db(unique_accent, accent_dict, top_db=20)
# plot_power_db(unique_accent, accent_dict, top_db=60)

plot_feature(feature_power_to_db, unique_accent, accent_dict, top_db=30, n_rows=5, sr=22500, seconds=3)
