import os

from model.config.config import LocalStorage, LocalTrainPipeline, PACKAGE_ROOT, TRAINED_MODEL_DIR


def test_model_config_storage_path():
    assert os.path.exists(LocalStorage.CLIPS_DIR)
    assert os.path.exists(LocalStorage.DEV_DIR)
    assert os.path.exists(LocalStorage.ROOT_DIR)
    assert os.path.exists(LocalStorage.TRAIN_DIR)


def test_gender_train_path():
    assert os.path.exists(LocalTrainPipeline.TRAIN_DIR)
    assert os.path.exists(LocalTrainPipeline.TEST_DIR)
    assert os.path.exists(LocalTrainPipeline.VAL_DIR)


def test_package_directory():
    assert os.path.exists(PACKAGE_ROOT)
    assert os.path.exists(TRAINED_MODEL_DIR)