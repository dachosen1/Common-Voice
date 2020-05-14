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
from model.config.config import MODEL_NAME, Model, TRAINED_MODEL_DIR, Train_Pipeline
from model.model_manager import train
from model.preprocessing.mp3_converter import MP3_Parser
from utils.utlis import csv_loader

_logger = logging.getLogger(__name__)

torch.manual_seed(0)


def run_training(model: type, train_dir: str, val_dir: str) -> None:
    train_dataset = DatasetFolder(root=train_dir, loader=csv_loader, extensions=".csv")
    val_dataset = DatasetFolder(root=val_dir, loader=csv_loader, extensions=".csv")

    train_data_loader = DataLoader(
        train_dataset, batch_size=Model.BATCH_SIZE, shuffle=True, drop_last=True
    )

    val_data_loader = DataLoader(
        val_dataset, batch_size=Model.BATCH_SIZE, shuffle=True, drop_last=True
    )

    model = model(
        num_layer=Model.NUM_LAYERS,
        input_size=Model.INPUT_SIZE,
        hidden_size=Model.HIDDEN_DIM,
        output_size=Model.OUTPUT_SIZE,
        dropout=Model.DROPOUT,
    )

    trained_model = train(
        model, train_data_loader, val_data_loader, early_stopping=False
    )
    _logger.info("Save RNN_TYPE in directory")
    torch.save(
        trained_model.state_dict(),
        os.path.join(TRAINED_MODEL_DIR, MODEL_NAME + __version__),
    )


if __name__ == "__main__":
    clips_path = config.Local_Storage.DATA_CLIPS_PATH
    mp3_list = os.listdir(clips_path)
    mp3_list = set(mp3_list)

    parser = MP3_Parser(
        data_path=config.Local_Storage.RAW_DATA_PATH,
        clips_dir=config.Local_Storage.DATA_CLIPS_PATH,
        document_path=config.Local_Storage.PARENT_FOLDER_PATH,
    )

    with futures.ThreadPoolExecutor() as executor:
        tqdm(executor.map(parser.convert_to_wav, mp3_list))

    run_training(
        model=AudioLSTM,
        train_dir=Train_Pipeline.TRAIN_DIR,
        val_dir=Train_Pipeline.VAL_DIR,
    )
