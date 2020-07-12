import logging
import os
import warnings
from concurrent import futures

import mlflow
import torch
import wandb
from model import __version__
from model.LSTM import AudioLSTM
from model.config.config import Common_voice_models, DataDirectory, TrainingTestingSplitDirectory, TRAINED_MODEL_DIR
from model.model_manager import train
from model.preprocessing.mp3_parser import Mp3parser
from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder
from tqdm import tqdm

from utlis import csv_loader, sample_weight

warnings.filterwarnings("ignore")

_logger = logging.getLogger("model")

torch.manual_seed(0)


class Run:
    def __init__(self, model_name):

        self.model_name = model_name
        ALL_PARAM = {'Train': self.model_name.TRAIN_PARAM, 'Model': self.model_name.PARAM,
                     'Frame': self.model_name.FRAME}

        self.label = model_name.LABEL
        self.name = model_name.NAME
        self.train_dir = os.path.join(self.label, DataDirectory.DEV_DIR, TrainingTestingSplitDirectory.TRAIN_DIR)
        self.val_dir = os.path.join(
            DataDirectory.DEV_DIR,
            self.label,
            TrainingTestingSplitDirectory.VAL_DIR,
        )
        self.test_dir = os.path.join(
            DataDirectory.DEV_DIR,
            self.label,
            TrainingTestingSplitDirectory.TEST_DIR,
        )
        self.output_size = model_name.PARAM['OUTPUT_SIZE']

        wandb.init('Common-Voice', config=ALL_PARAM)

    def train_model(self, model: type, RNN_TYPE) -> None:
        train_dataset = DatasetFolder(
            root=self.train_dir, loader=csv_loader, extensions=".csv"
        )
        val_dataset = DatasetFolder(
            root=self.val_dir, loader=csv_loader, extensions=".csv"
        )

        train_sample_weight = sample_weight(train_dataset)
        val_sample_weight = sample_weight(val_dataset)

        train_data_loader = DataLoader(
            train_dataset,
            batch_size=self.name.MODEL_PARAM["BATCH_SIZE"],
            sampler=train_sample_weight,
            num_workers=4,
            drop_last=True,
        )

        _logger.info(
            "Uploaded training data to {} using {} batch sizes".format(
                self.train_dir, model.MODEL_PARAM["BATCH_SIZE"]
            )
        )

        val_data_loader = DataLoader(
            val_dataset,
            batch_size=model.MODEL_PARAM["BATCH_SIZE"],
            sampler=val_sample_weight,
            num_workers=4,
            drop_last=True,
        )

        _logger.info(
            "Uploaded validation data to {} using {} batch sizes".format(
                self.val_dir, model.MODEL_PARAM["BATCH_SIZE"]
            )
        )

        model = model(
            num_layer=self.model_name.MODEL_PARAM["NUM_LAYERS"],
            input_size=self.model_name.MODEL_PARAM["INPUT_SIZE"],
            hidden_size=self.model_name.MODEL_PARAM["HIDDEN_DIM"],
            output_size=self.model_name.MODEL_PARAM["OUTPUT_SIZE"],
            dropout=self.model_name.MODEL_PARAM["DROPOUT"],
            RNN_TYPE=RNN_TYPE,
            batch_size=self.model_name.MODEL_PARAM["BATCH_SIZE"],
        )

        _logger.info(
            "LSTM Model has been initialized with {} "
            " layers, {} hidden dimension, "
            "{} Input size, "
            "{} output size, "
            "{} batch size, "
            "{} dropout".format(
                self.model_name.MODEL_PARAM["NUM_LAYERS"],
                self.model_name.MODEL_PARAM["HIDDEN_DIM"],
                self.model_name.MODEL_PARAM["INPUT_SIZE"],
                self.model_name.MODEL_PARAM["OUTPUT_SIZE"],
                self.model_name.MODEL_PARAM["BATCH_SIZE"],
                self.model_name.MODEL_PARAM["DROPOUT"],
            )
        )

        trained_model = train(model, train_data_loader, val_data_loader,
                              learning_rate=self.model_name.TRAIN_PARAM['LEARNING_RATE'],
                              epoch=self.model_name.TRAIN_PARAM['EPOCH'],
                              gradient_clip=self.model_name.TRAIN_PARAM['GRADIENT_CLIP'],
                              early_stopping=True
                              )

        trained_model_path = os.path.join(
            TRAINED_MODEL_DIR, self.name + __version__ + ".pt"
        )
        _logger.info(
            "Saved {} version {} in {}".format(
                self.name, __version__, TRAINED_MODEL_DIR
            )
        )

        torch.save(trained_model.state_dict(), trained_model_path)

    def load_data(self, method, percentage):
        clips_path = DataDirectory.CLIPS_DIR
        mp3_list = os.listdir(clips_path)

        if method == "train":
            mp3_list = mp3_list[0: round(len(mp3_list) * percentage)]
            mp3_list = set(mp3_list)

            _logger.info("Uploaded {} MP3 files for trainings".format(len(mp3_list)))

            parser = Mp3parser(
                data_path=DataDirectory.ROOT_DIR,
                clips_dir=DataDirectory.CLIPS_DIR,
                document_path=DataDirectory.DEV_DIR,
                data_label=self.label,
            )

            with futures.ThreadPoolExecutor() as executor:
                tqdm(executor.map(parser.convert_to_wav, mp3_list))

            _logger.info("Added {} total training examples.".format(parser.add_count))
            _logger.info(
                "Removed {} total training examples.".format(parser.remove_count)
            )

        else:
            _logger.info(
                "Skipping MP3 feature engineering. Will use existing mfcc data for training"
            )


if __name__ == "__main__":
    with mlflow.start_run():
        run = Run(Common_voice_models.Gender)
        run.load_data(method="train", percentage=0.05)
        run.train_model(model=AudioLSTM, RNN_TYPE="LSTM")

    # TODO: Automate Model Labels and output

    # predict.directory_predict(r'C:\Users\ander\Documents\common-voice-dev\gender\test_data\female')
