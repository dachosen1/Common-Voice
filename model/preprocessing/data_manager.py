import os

import numpy as np
import pandas as pd
from python_speech_features import mfcc
from scipy.io import wavfile
from tqdm import tqdm

from utils.utlis import envelope


def clean_data(
    document_path, filename: str, rate: int, signal: np.ndarray, label: object
) -> None:
    seq_count = signal.shape[0] // rate

    try:
        flatten_data = signal[0 : seq_count * rate].reshape(seq_count, -1)
        flatten_data = pd.DataFrame(flatten_data)

        for row in range(flatten_data.shape[0]):
            train_test_choice = np.random.choice(
                ["train_data", "val_data", "test_data"], p=[0.7, 0.1, 0.1]
            )
            save_path = os.path.join(document_path, "gender", train_test_choice, label)
            flatten_data.iloc[row].to_csv(
                "{}\{}-{}.csv".format(save_path, filename, row),
                header=False,
                index=False,
            )

    except ValueError:
        print(" Skipped {} ......".format(filename))


def label_split(
    wav_path: str, data_path: str, mel_seq_count: int, document_path
) -> None:
    clip_wav = os.listdir(wav_path)
    label_data = pd.read_csv(os.path.join(data_path, "data.csv"))
    label_data = label_data[label_data["gender"].notna()]
    label_data = label_data[label_data["gender"] != "other"]

    for wav in tqdm(clip_wav):
        rate, signal = wavfile.read(os.path.join(wav_path, wav))

        try:
            label_name = label_data[label_data.name == wav.split(".")[0]].gender.values[
                0
            ]
            mask = envelope(y=signal, signal_rate=rate, threshold=100)
            signal = signal[mask]
            wav_name = wav.split(".")[0]
            mel = mfcc(signal, rate, numcep=13, nfilt=26, nfft=1500).flatten()
            clean_data(document_path, wav_name, mel_seq_count, mel, label_name)
        except IndexError:
            print(" The label for {} is not in compatible".format(wav))
