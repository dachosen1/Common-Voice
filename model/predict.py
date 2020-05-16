import os

import pandas as pd
import torch

from model.config import config
from model.pipeline_mananger import load_model


def predict(dir_path):
    model, path = load_model(config.GENDER_MODEL_NAME)
    model.load_state_dict(torch.load(path))
    model.eval()

    h = model.init_hidden(1)
    directory = dir_path
    director_list = os.listdir(directory)

    for path in director_list:
        data = pd.read_csv(os.path.join(directory, path), header=None).to_numpy()
        data = torch.from_numpy(data)

        if torch.cuda.is_available():
            data = data.flatten().cuda()
            model.cuda()

        out, _ = model(data.float().view(1, config.Model.INPUT_SIZE, -1), h)
        # out = out.contiguous().view(-1)
        pred = torch.round(out.squeeze())
        prob = out.float().cpu()
        label = config.GENDER_LABEL[int(pred.cpu().data.numpy())]
        print(label, prob.float().cpu().detach().cpu().numpy()[0])


if __name__ == '__main__':
    predict()
