import os
import shutil

import pandas as pd
from pydub import AudioSegment
from tqdm import tqdm

AudioSegment.ffmpeg = r"C:\Users\ander\ffmpeg-4.2.2-win64-static\bin"


def convert_to_wav(clips_name: set) -> None:
    clips_dir = r"C:\Users\ander\Documents"

    for mp3 in tqdm(clips_name):
        path = os.path.join(clips_dir + "\clips", mp3)
        file = AudioSegment.from_mp3(path)
        new_file_name = f'{mp3.split(".")[0]}.wav'
        wav_path = os.path.join(clips_dir + "\wav", new_file_name)
        file.export(wav_path, format="wav")


def remove_unlabel_files(clips_names):
    data = pd.read_csv("Development/data.csv")
    data_path = set(data.path)

    delete_path = r"C:\Users\ander\Documents\delete"

    for mp3 in tqdm(clips_names):
        if mp3 not in data_path:
            shutil.move(os.path.join(clips_path, mp3), os.path.join(delete_path, mp3))


if __name__ == "__main__":
    clips_path = r"C:\Users\ander\Documents\clips"
    mp3_list = os.listdir(clips_path)[2900:]
    mp3_list = set(mp3_list)
    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     executor.map(convert_to_wav, mp3_list)

    convert_to_wav(mp3_list)
