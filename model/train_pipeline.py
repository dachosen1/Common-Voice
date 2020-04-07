import logging

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder

from LSTM import AudioLSTM
from config.config import Model
from model_manager import train
from utils.utlis import csv_loader

_logger = logging.getLogger(__name__)


def run_training() -> None:
    dataset = DatasetFolder(
        root = "Development/data/clean", loader = csv_loader, extensions = ".csv"
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = Model.BATCH_SIZE, shuffle = True)

    model = AudioLSTM(
        num_layer = Model.NUM_LAYERS,
        input_size = Model.INPUT_SIZE,
        hidden_size = Model.HIDDEN_DIM,
        output_size = Model.OUTPUT_SIZE,
        dropout = Model.DROPOUT,
    )

    train_model = train(model, dataloader, dataloader)
    _logger.info("Save model in directory")
    torch.save(train_model.state_dict(), "model/trained_model" + "/model.pt")


if __name__ == "__main__":
    run_training()
