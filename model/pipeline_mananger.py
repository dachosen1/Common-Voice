import logging
import os
import typing as t

import joblib

from model import LSTM
from model import __version__ as _version
from model.config import config

_logger = logging.getLogger(__name__)


def remove_old_pipelines(*, files_to_keep: t.List[str]) -> None:
    """
    Remove old model pipelines. This is to ensure there is a simple one-to-one mapping between the package version
    and the model version to be imported and used by other applications. However, we do also include the immediate
    previous pipeline version for differential testing purposes.
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


def load_model(model_name: str) -> object:
    """
    Load a saved PyTorch model
    :param model_name:  Name of the model to parse
    :return: torch model and model path
    """

    model = LSTM.AudioLSTM(
        num_layer=config.MODEL_PARAM['NUM_LAYERS'],
        input_size=config.MODEL_PARAM['INPUT_SIZE'],
        hidden_size=config.MODEL_PARAM['HIDDEN_DIM'],
        output_size=config.MODEL_PARAM['OUTPUT_SIZE'],
        dropout=config.MODEL_PARAM['DROPOUT'],
        batch_size=1
    )
    model.eval()
    file_path = os.path.join(config.TRAINED_MODEL_DIR, model_name + _version + ".pt")
    return model, file_path
