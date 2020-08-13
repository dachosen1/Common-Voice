import os

import pandas as pd
import torch

from audio_model.config import config
from audio_model.pipeline_mananger import load_model
from utlis import generate_pred


def test_model(dir_path):
    model, path = load_model(config.GENDER_MODEL_NAME)
    model.load_state_dict(torch.load(path))
    model.eval()

    model.init_hidden()
    directory = dir_path
    director_list = os.listdir(directory)

    assert os.path.exists(dir_path)

    for path in director_list:
        data = pd.read_csv(os.path.join(directory, path), header=None).to_numpy()
        generate_pred(data, model, config.GENDER_LABEL)
