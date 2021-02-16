import logging
import os

import numpy as np
import torch

from audio_model import config
from audio_model.pipeline_mananger import load_model
from audio_model.utils import generate_pred

_logger = logging.getLogger("audio_model")


def model_pipeline_test(model_name):
    """
    Test model performance on a test set
    """
    dir_path = config.DataDirectory.DEV_DIR
    model, path = load_model(model_name)
    model.load_state_dict(torch.load(path))
    model.eval()
    model.init_hidden()

    test_dir = os.path.join(dir_path, model_name.LABEL, config.TrainingTestingSplitDirectory.TEST_DIR)

    assert os.path.exists(test_dir)
    test_accuracy = {}
    for label in os.listdir(test_dir):
        test_set = os.path.join(test_dir, label)
        accuracy = 0
        for value in os.listdir(test_set):
            data = np.load(os.path.join(test_set, value))
            pred_label, _ = generate_pred(mel=data, model=model, label=model_name.OUTPUT, model_name=model_name)
            if pred_label.lower() == label.lower():
                accuracy += 1
        test_accuracy[label] = accuracy / len(os.listdir(test_set))

    _logger.info(
        test_accuracy
    )
    return test_accuracy
