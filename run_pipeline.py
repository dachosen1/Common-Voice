import logging
import os
import warnings

import torch
import wandb
from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder

from audio_model import __version__
from audio_model.LSTM import AudioLSTM
from audio_model.config.config import (
    CommonVoiceModels,
    DataDirectory,
    TrainingTestingSplitDirectory,
    TRAINED_MODEL_DIR
)
from audio_model.model_manager import train
from audio_model.preprocessing.mp3_parser import Mp3parser
from utlis import csv_loader, sample_weight, run_thread_pool

warnings.filterwarnings("ignore")

_logger = logging.getLogger("audio_model")

torch.manual_seed(0)


class Run:
    def __init__(self, model_name):

        self.model_name = model_name
        ALL_PARAM = {
            "Model": self.model_name.PARAM,
            "Frame": self.model_name.FRAME,
        }

        self.label = model_name.LABEL
        self.name = model_name.NAME
        self.train_dir = os.path.join(
            DataDirectory.DEV_DIR, self.label, TrainingTestingSplitDirectory.TRAIN_DIR
        )
        self.val_dir = os.path.join(
            DataDirectory.DEV_DIR, self.label, TrainingTestingSplitDirectory.VAL_DIR,
        )
        self.test_dir = os.path.join(
            DataDirectory.DEV_DIR, self.label, TrainingTestingSplitDirectory.TEST_DIR,
        )
        self.output_size = model_name.PARAM["OUTPUT_SIZE"]

        wandb.init(project="Common-Voice", tags=self.label, config=ALL_PARAM)

    def train_model(self, model: type, RNN_TYPE) -> None:
        train_dataset = DatasetFolder(
            root=self.train_dir, loader=csv_loader, extensions=".npy"
        )
        val_dataset = DatasetFolder(
            root=self.val_dir, loader=csv_loader, extensions=".npy"
        )

        train_sample_weight = sample_weight(train_dataset)
        val_sample_weight = sample_weight(val_dataset)

        train_data_loader = DataLoader(
            train_dataset,
            batch_size=self.model_name.PARAM["BATCH_SIZE"],
            sampler=train_sample_weight,
            num_workers=4,
            drop_last=True,
        )

        _logger.info(
            "Uploaded training data to {} using {} batch sizes".format(
                self.train_dir, self.model_name.PARAM["BATCH_SIZE"]
            )
        )

        val_data_loader = DataLoader(
            val_dataset,
            batch_size=self.model_name.PARAM["BATCH_SIZE"],
            sampler=val_sample_weight,
            num_workers=4,
            drop_last=True,
        )

        _logger.info(
            "Uploaded validation data to {} using {} batch sizes".format(
                self.val_dir, self.model_name.PARAM["BATCH_SIZE"]
            )
        )

        model = model(
            num_layer=self.model_name.PARAM["NUM_LAYERS"],
            input_size=self.model_name.PARAM["INPUT_SIZE"],
            hidden_size=self.model_name.PARAM["HIDDEN_DIM"],
            output_size=self.model_name.PARAM["OUTPUT_SIZE"],
            dropout=self.model_name.PARAM["DROPOUT"],
            RNN_TYPE=RNN_TYPE,
            batch_size=self.model_name.PARAM["BATCH_SIZE"],
        )

        _logger.info(
            "LSTM Model has been initialized with {} "
            "layers, {} hidden dimension, "
            "{} Input size, "
            "{} output size, "
            "{} batch size, "
            "{} dropout".format(
                self.model_name.PARAM["NUM_LAYERS"],
                self.model_name.PARAM["HIDDEN_DIM"],
                self.model_name.PARAM["INPUT_SIZE"],
                self.model_name.PARAM["OUTPUT_SIZE"],
                self.model_name.PARAM["BATCH_SIZE"],
                self.model_name.PARAM["DROPOUT"],
            )
        )

        trained_model = train(
            model=model,
            epoch=self.model_name.PARAM["EPOCH"],
            gradient_clip=self.model_name.PARAM["GRADIENT_CLIP"],
            learning_rate=self.model_name.PARAM["LEARNING_RATE"],
            train_loader=train_data_loader,
            valid_loader=val_data_loader,
            early_stopping=True,
        )

        trained_model_path = os.path.join(TRAINED_MODEL_DIR, self.name + __version__ + ".pt")
        _logger.info("Saved {} version {} in {}".format(self.name, __version__, TRAINED_MODEL_DIR))

        torch.save(trained_model.state_dict(), trained_model_path)

    def load_data(self, method, percentage):
        if method == "train":
            parser = Mp3parser(
                data_path=DataDirectory.DATA_DIR,
                clips_dir=DataDirectory.CLIPS_DIR,
                document_path=DataDirectory.DEV_DIR,
                data_label=self.label,
                model=self.model_name,
            )

            path_list = parser.label_data.path
            path_list = path_list[0:round(len(path_list) * percentage)]

            mp3_list = range(len(path_list))
            _logger.info("Uploaded {} MP3 files for trainings".format(len(mp3_list)))

            run_thread_pool(function=parser.convert_to_wav, my_iter=mp3_list)

            _logger.info("Added {} total training examples.".format(parser.add_count))
            _logger.info("Removed {} total training examples.".format(parser.remove_count))

        else:
            _logger.info("Skipping MP3 feature engineering. Will use existing mfcc data for training")


if __name__ == "__main__":
    run = Run(CommonVoiceModels.Country)
    run.load_data(method="train", percentage=0.01)
    run.train_model(model=AudioLSTM, RNN_TYPE="LSTM")
