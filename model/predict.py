import os

import pandas as pd
import torch

from model.config import config
from model.pipeline_mananger import load_model


def predict(dir_path):
    model, path = load_model(config.GENDER_MODEL_NAME)
    model.load_state_dict(torch.load(path))
    model.eval()

    model.init_hidden()
    directory = dir_path
    director_list = os.listdir(directory)

    assert os.path.exists(dir_path)

    for path in director_list:
        data = pd.read_csv(os.path.join(directory, path), header=None).to_numpy()
        data = torch.from_numpy(data).view(1, 13, 44).float()

        if torch.cuda.is_available():
            model.cuda()
            data = data.cuda()

        out = model(data)
        prob = torch.topk(out, k=1).values
        pred = torch.topk(out, k=1).indices
        label = config.GENDER_LABEL[int(pred.cpu().data.numpy())]
        print(f'Prediction: {label}, Probability: {round(float(prob.flatten()[0]), 5)}')


if __name__ == '__main__':
    predict()
