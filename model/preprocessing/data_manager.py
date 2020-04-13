import os

import numpy as np
import pandas as pd
from python_speech_features import mfcc
from scipy.io import wavfile
from tqdm import tqdm

from model.config.config import Model, Storage


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


def clean_data(
    document_path, filename: str, rate: int, signal: np.ndarray, label: object
) -> None:
    seq_count = signal.shape[0] // rate

    try:
        flatten_data = signal[0 : seq_count * rate].reshape(seq_count, -1)
        flatten_data = pd.DataFrame(flatten_data)

        for row in range(flatten_data.shape[0]):
            train_test_choice = np.random.choice(
                ["train_data", "val_data"], p=[0.8, 0.2]
            )
            save_path = os.path.join(document_path, "gender", train_test_choice, label)
            flatten_data.iloc[row].to_csv(
                f"{save_path}\{filename}-{row}.csv", header=False, index=False
            )

            assert flatten_data.iloc[row].shape[0] == rate

    except ValueError:
        print(f" Skipped {filename} ......")


def gender_split(
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
            print(f" The label for {wav} is not in compatible")


if __name__ == "__main__":
    gender_split(
        wav_path=Storage.WAV_PATH,
        data_path=Storage.RAW_DATA_PATH,
        mel_seq_count=Model.INPUT_SIZE,
        document_path=Storage.PARENT_FOLDER_PATH,
    )
