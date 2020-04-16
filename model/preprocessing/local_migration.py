import os
import shutil

import pandas as pd
from pydub import AudioSegment
from tqdm import tqdm

from model.config.config import Storage

AudioSegment.ffmpeg = r"C:\Users\ander\ffmpeg-4.2.2-win64-static\bin"


def convert_to_wav(clips_name: set) -> None:
    clips_dir = r"C:\Users\ander\Documents\common-voice-data\clips"
    wav_dir = r"C:\Users\ander\Documents\common-voice-data\wav"

    path = os.path.join(clips_dir, clips_name)
    file = AudioSegment.from_mp3(path)
    new_file_name = f'{clips_name.split(".")[0]}.wav'
    wav_path = os.path.join(wav_dir, new_file_name)
    file.export(wav_path, format="wav")
    print("exported {}".format(clips_name))


def remove_un_label_files(clips_names):
    data = pd.read_csv("Development/data.csv")
    data_path = set(data.path)
    clips_path = Storage.DATA_CLIPS_PATH

    delete_path = r"C:\Users\ander\Documents\delete"

    for mp3 in tqdm(clips_names):
        if mp3 not in data_path:
            shutil.move(os.path.join(clips_path, mp3), os.path.join(delete_path, mp3))
