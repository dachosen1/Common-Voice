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
from model.config import config
from model.model_manager import train
from model.preprocessing.mp3_parser import Mp3parser
from utlis import csv_loader, sample_weight

warnings.filterwarnings("ignore")

_logger = logging.getLogger('model')

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
    _logger.info(f"Uploaded training data to {train_dir} using {config.MODEL_PARAM['BATCH_SIZE']} batch sizes")

    val_data_loader = DataLoader(
        val_dataset, batch_size=config.MODEL_PARAM['BATCH_SIZE'], sampler=val_sample_weight, num_workers=4,
        drop_last=True
    )
    _logger.info(f"Uploaded validation data to {val_dir} using {config.MODEL_PARAM['BATCH_SIZE']} batch sizes")



    model = model(
        num_layer=config.MODEL_PARAM['NUM_LAYERS'],
        input_size=config.MODEL_PARAM['INPUT_SIZE'],
        hidden_size=config.MODEL_PARAM['HIDDEN_DIM'],
        output_size=config.MODEL_PARAM['OUTPUT_SIZE'],
        dropout=config.MODEL_PARAM['DROPOUT'],
        RNN_TYPE=RNN_TYPE,
        batch_size=config.MODEL_PARAM['BATCH_SIZE']
    )

    _logger.info(f"LSTM Model has been initialized with {config.MODEL_PARAM['NUM_LAYERS']}  layers, "
                 f" {config.MODEL_PARAM['HIDDEN_DIM']} hidden dimension,"
                 f"{config.MODEL_PARAM['INPUT_SIZE']} Input size "
                 f"{config.MODEL_PARAM['OUTPUT_SIZE']} output size "
                 f"{config.MODEL_PARAM['BATCH_SIZE']} batch size"
                 f"{config.MODEL_PARAM['DROPOUT']} dropout")

    trained_model = train(
        model, train_data_loader, val_data_loader, early_stopping=False
    )

    trained_model_path = os.path.join(
        config.TRAINED_MODEL_DIR, config.GENDER_MODEL_NAME + __version__ + ".pt"
    )
    _logger.info(f"Saved {config.GENDER_MODEL_NAME} version {__version__} in {config.TRAINED_MODEL_DIR}")

    torch.save(trained_model.state_dict(), trained_model_path)


def generate_training_data(method, percentage):
    clips_path = config.Storage.CLIPS_DIR
    mp3_list = os.listdir(clips_path)

    if method == "dev":
        mp3_list = mp3_list[0: round(len(mp3_list) * percentage)]
        mp3_list = set(mp3_list)

        parser = Mp3parser(
            data_path=config.Storage.ROOT_DIR,
            clips_dir=config.Storage.CLIPS_DIR,
            document_path=config.Storage.DEV_DIR,
        )

        _logger.info(f'Uploaded {len(mp3_list)} MP3 files for trainings')

        with futures.ThreadPoolExecutor() as executor:
            tqdm(executor.map(parser.convert_to_wav, mp3_list))

    elif method == "train":
        mp3_list = set(mp3_list)
        parser = Mp3parser(
            data_path=config.Storage.ROOT_DIR,
            clips_dir=config.Storage.CLIPS_DIR,
            document_path=config.Storage.TRAIN_DIR,
        )

        with futures.ThreadPoolExecutor() as executor:
            tqdm(executor.map(parser.convert_to_wav, mp3_list))

        _logger.info(f'Uploaded {len(mp3_list)} MP3 files for trainings')
    else:
        _logger.info("Skipping MP3 feature engineering. Will use existing mfcc data for training")



if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    generate_training_data(method="none", percentage=0.01)

    run_training(
        model=AudioLSTM,
        train_dir=config.Pipeline.TRAIN_DIR,
        val_dir=config.Pipeline.VAL_DIR,
        RNN_TYPE='LSTM'
    )

    # predict.directory_predict(r'C:\Users\ander\Documents\common-voice-dev\gender\test_data\female')
