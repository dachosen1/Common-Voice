import logging

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder

from LSTM import AudioLSTM
from config.config import Model
from model import __version__
from model_manager import train
from utils.utlis import csv_loader

_logger = logging.getLogger(__name__)


def run_training() -> None:
    train_dataset = DatasetFolder(
        root=r"C:\Users\ander\Documents\train_data",
        loader=csv_loader,
        extensions=".csv",
    )

    val_dataset = DatasetFolder(
        root=r"C:\Users\ander\Documents\dev_data", loader=csv_loader, extensions=".csv"
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=Model.BATCH_SIZE, shuffle=True
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=Model.BATCH_SIZE, shuffle=True
    )

    model = AudioLSTM(
        num_layer=Model.NUM_LAYERS,
        input_size=Model.INPUT_SIZE,
        hidden_size=Model.HIDDEN_DIM,
        output_size=Model.OUTPUT_SIZE,
        dropout=Model.DROPOUT,
    )

    train_model = train(model, train_dataloader, val_dataloader)
    _logger.info("Save model in directory")
    torch.save(
        train_model.state_dict(),
        "model/trained_model" + "/model_{}.pt".format(__version__),
    )


if __name__ == "__main__":
    run_training()
