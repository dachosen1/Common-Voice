import os
from concurrent import futures

from tqdm import tqdm

from model.config.config import Model, Storage
from model.preprocessing.data_manager import label_split
from model.preprocessing.local_migration import convert_to_wav

clips_path = Storage.DATA_CLIPS_PATH
mp3_list = os.listdir(clips_path)
mp3_list = set(mp3_list)

with futures.ThreadPoolExecutor() as executor:
    tqdm(executor.map(convert_to_wav, mp3_list))

label_split(
    data_path=Storage.RAW_DATA_PATH,
    wav_path=Storage.WAV_PATH,
    mel_seq_count=Model.INPUT_SIZE,
    document_path=Storage.PARENT_FOLDER_PATH,
)
