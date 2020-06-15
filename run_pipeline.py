import logging
import os
import warnings
from concurrent import futures

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder
from tqdm import tqdm

from model import __version__
from model import predict
from model.LSTM import AudioLSTM
from model.config import config, logging_config
from model.model_manager import train
from model.preprocessing.mp3_parser import MP3_Parser
from utlis import csv_loader, sample_weight

warnings.filterwarnings("ignore")

logging_config.get_logger()
_logger = logging.getLogger(__name__)

torch.manual_seed(0)


def run_training(model: type, train_dir: str, val_dir: str, RNN_TYPE) -> None:
    train_dataset = DatasetFolder(root=train_dir, loader=csv_loader, extensions=".csv")
    val_dataset = DatasetFolder(root=val_dir, loader=csv_loader, extensions=".csv")

    train_sample_weight = sample_weight(train_dataset)
    val_sample_weight = sample_weight(val_dataset)

    train_data_loader = DataLoader(
        train_dataset, batch_size=config.MODEL_PARAM['BATCH_SIZE'], sampler=train_sample_weight, num_workers=4,
        drop_last=True
    )

    val_data_loader = DataLoader(
        val_dataset, batch_size=config.MODEL_PARAM['BATCH_SIZE'], sampler=val_sample_weight, num_workers=4,
        drop_last=True
    )

    _logger.info('Successfully Uploaded Train and Validation data........ ')

    model = model(
        num_layer=config.MODEL_PARAM['NUM_LAYERS'],
        input_size=config.MODEL_PARAM['INPUT_SIZE'],
        hidden_size=config.MODEL_PARAM['HIDDEN_DIM'],
        output_size=config.MODEL_PARAM['OUTPUT_SIZE'],
        dropout=config.MODEL_PARAM['DROPOUT'],
        RNN_TYPE=RNN_TYPE,
        batch_size=config.MODEL_PARAM['BATCH_SIZE']
    )

    _logger.info('LSTM Model has been initialized')

    trained_model = train(
        model, train_data_loader, val_data_loader, early_stopping=False
    )

    trained_model_path = os.path.join(
        config.TRAINED_MODEL_DIR, config.GENDER_MODEL_NAME + __version__ + ".pt"
    )

    _logger.info(f"Save Model version {__version__} in directory")
    torch.save(trained_model.state_dict(), trained_model_path)


def generate_training_data(method, percentage):
    clips_path = config.LocalStorage.CLIPS_DIR
    mp3_list = os.listdir(clips_path)

    if method == "dev":
        mp3_list = mp3_list[0: round(len(mp3_list) * percentage)]
        mp3_list = set(mp3_list)

        parser = MP3_Parser(
            data_path=config.LocalStorage.ROOT_DIR,
            clips_dir=config.LocalStorage.CLIPS_DIR,
            document_path=config.LocalStorage.DEV_DIR,
        )

        with futures.ThreadPoolExecutor() as executor:
            tqdm(executor.map(parser.convert_to_mfcc, mp3_list))

    elif method == "train":
        mp3_list = set(mp3_list)
        parser = MP3_Parser(
            data_path=config.LocalStorage.ROOT_DIR,
            clips_dir=config.LocalStorage.CLIPS_DIR,
            document_path=config.LocalStorage.TRAIN_DIR,
        )

        with futures.ThreadPoolExecutor() as executor:
            tqdm(executor.map(parser.convert_to_mfcc, mp3_list))

    else:
        _logger.info("Skipping MP3 feature engineering. Will use existing mfcc data for training")
    _logger.info("Done Uploading Data for training")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    generate_training_data(method="none", percentage=0.01)

    run_training(
        model=AudioLSTM,
        train_dir=config.LocalTrainPipeline.TRAIN_DIR,
        val_dir=config.LocalTrainPipeline.VAL_DIR,
        RNN_TYPE='LSTM'
    )

    predict.predict(r'C:\Users\ander\Documents\common-voice-dev\gender\test_data\female')
