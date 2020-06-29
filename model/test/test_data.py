import os
import random

from numpy import genfromtxt

from model.config.config import GENDER_LABEL, LocalTrainPipeline, MODEL_PARAM


def check_directory(directory, sample_size=random.randint(1, 1000), input_size=MODEL_PARAM['INPUT_SIZE']):
    for keys, value in GENDER_LABEL.items():
        new_path = os.path.join(directory, value)
        list_director_new_path = random.sample(os.listdir(new_path), sample_size)

        for data in list_director_new_path:
            data = genfromtxt(os.path.join(new_path, data), delimiter=',')
            assert data.shape[1] == input_size


def test_val_data_dimensions():
    check_directory(directory=LocalTrainPipeline.TEST_DIR)


def test_train_data_dimensions():
    check_directory(directory=LocalTrainPipeline.TRAIN_DIR)


def test_test_data_dimensions():
    check_directory(directory=LocalTrainPipeline.TEST_DIR)
