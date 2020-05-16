import logging
import os
from concurrent import futures

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder
from tqdm import tqdm

from model import __version__
from model.LSTM import AudioLSTM
from model.config import config
from model.model_manager import train
from model.preprocessing.mp3_parser import MP3_Parser
from utlis import csv_loader

_logger = logging.getLogger(__name__)

torch.manual_seed(0)


def run_training(model: type, train_dir: str, val_dir: str) -> None:
    train_dataset = DatasetFolder(root=train_dir, loader=csv_loader, extensions=".csv")
    val_dataset = DatasetFolder(root=val_dir, loader=csv_loader, extensions=".csv")

    train_data_loader = DataLoader(
        train_dataset, batch_size=config.Model.BATCH_SIZE, shuffle=True, num_workers=4
    )

    val_data_loader = DataLoader(
        val_dataset, batch_size=config.Model.BATCH_SIZE, shuffle=True, num_workers=4
    )

    model = model(
        num_layer=config.Model.NUM_LAYERS,
        input_size=config.Model.INPUT_SIZE,
        hidden_size=config.Model.HIDDEN_DIM,
        output_size=config.Model.OUTPUT_SIZE,
        dropout=config.Model.DROPOUT,
    )

    trained_model = train(
        model, train_data_loader, val_data_loader, early_stopping=False
    )

    PATH = os.path.join(
        config.TRAINED_MODEL_DIR, config.GENDER_MODEL_NAME + __version__ + ".pt"
    )

    _logger.info("Save RNN_TYPE in directory")
    torch.save(trained_model.state_dict(), PATH)


def generate_training_data(method="dev"):
    clips_path = config.LocalStorage.CLIPS_DIR
    mp3_list = os.listdir(clips_path)

    if method == "dev":
        mp3_list = mp3_list[0: round(len(mp3_list) * 0.01)]

        mp3_list = set(mp3_list)

        parser = MP3_Parser(
            data_path=config.LocalStorage.ROOT_DIR,
            clips_dir=config.LocalStorage.CLIPS_DIR,
            document_path=config.LocalStorage.DEV_DIR,
        )

        with futures.ThreadPoolExecutor() as executor:
            tqdm(executor.map(parser.convert_to_wav, mp3_list))

    elif method == "train":
        mp3_list = set(mp3_list)
        parser = MP3_Parser(
            data_path=config.LocalStorage.ROOT_DIR,
            clips_dir=config.LocalStorage.CLIPS_DIR,
            document_path=config.LocalStorage.TRAIN_DIR,
        )

        with futures.ThreadPoolExecutor() as executor:
            tqdm(executor.map(parser.convert_to_wav, mp3_list))

    else:
        return print("Skipping developing pipeline")

    print("Done Uploading Data for training")


if __name__ == "__main__":
    generate_training_data(method="None")

    run_training(
        model=AudioLSTM,
        train_dir=config.LocalTrainPipeline.TRAIN_DIR,
        val_dir=config.LocalTrainPipeline.VAL_DIR,
    )
