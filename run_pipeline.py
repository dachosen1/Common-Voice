import logging

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder

from model import __version__
from model.LSTM import AudioLSTM
from model.config.config import Model, Train_Pipeline
from model.model_manager import train
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

    train_model = train(model, train_data_loader, val_data_loader, early_stopping=False)
    _logger.info("Save model in directory")
    torch.save(
        train_model.state_dict(),
        "model/trained_model" + "/model_gender_{}.pth".format(__version__),
    )


if __name__ == "__main__":
    run_training(
        model=AudioLSTM,
        train_dir=Train_Pipeline.TRAIN_DIR,
        val_dir=Train_Pipeline.VAL_DIR,
    )
