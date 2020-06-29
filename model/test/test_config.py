import os

from model.config.config import Storage, Pipeline, PACKAGE_ROOT, TRAINED_MODEL_DIR


def test_model_config_storage_path():
    assert os.path.exists(Storage.CLIPS_DIR)
    assert os.path.exists(Storage.DEV_DIR)
    assert os.path.exists(Storage.ROOT_DIR)
    assert os.path.exists(Storage.TRAIN_DIR)


def test_gender_train_path():
    assert os.path.exists(Pipeline.TRAIN_DIR)
    assert os.path.exists(Pipeline.TEST_DIR)
    assert os.path.exists(Pipeline.VAL_DIR)


def test_package_directory():
    assert os.path.exists(PACKAGE_ROOT)
    assert os.path.exists(TRAINED_MODEL_DIR)