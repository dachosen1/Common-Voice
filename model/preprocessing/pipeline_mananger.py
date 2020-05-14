import logging
import os
import typing as t

import joblib
import torch

from model import LSTM
from model import __version__ as _version
from model.config import config

_logger = logging.getLogger(__name__)


def remove_old_pipelines(*, files_to_keep: t.List[str]) -> None:
    """
    Remove old model pipelines. This is to ensure there is a simple one-to-one mapping between the package version and the model version to be imported and used by other applications.
    However, we do also include the immediate previous pipeline version for differential testing purposes.
    """
    do_not_delete = files_to_keep + ["__init__.py"]
    for model_file in config.TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in do_not_delete:
            model_file.unlink()


def save_pipeline(*, pipeline_to_persist) -> None:
    """
    Saves the versioned model, and overwrites any previous saved models. This ensures that when the package is
    published, there is only one trained model that can be called, and we know exactly how it was built.
    :param pipeline_to_persist:
    :return:
    """

    save_file_name = f"{config.MODEL_NAME}{_version}.pth"
    save_path = config.TRAINED_MODEL_DIR / save_file_name

    remove_old_pipelines(files_to_keep=[save_file_name])
    joblib.dump(pipeline_to_persist, save_path)
    _logger.info(f"saved pipeline: {save_file_name}")


def load_pipeline(MODEL_NAME: str) -> object:
    """
    Load a saved PyTorch model
    :param MODEL_NAME:  Name of the model to parse
    :return:
    """
    file_path = os.path.join(config.TRAINED_MODEL_DIR, MODEL_NAME + _version + ".pth")
    trained_model = LSTM.AudioLSTM.load_state_dict(torch.load(file_path))
    return trained_model
