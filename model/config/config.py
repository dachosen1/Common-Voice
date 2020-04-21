class Bucket:
    RAW_DATA = "common-voice-all"
    META_DATA = "common-voice-data"
    VALIDATION_SET = "common-voice-dev"
    TEST_SET = "common-voice-test"
    TRAIN_SET = "common-voice-train"


class Model:
    OUTPUT_SIZE = 2
    HIDDEN_DIM = 512
    NUM_LAYERS = 10
    DROPOUT = 0.30
    INPUT_SIZE = 648
    BATCH_SIZE = 256


class Train:
    LEARNING_RATE = 0.00001
    GRADIENT_CLIP = 50
    EPOCH = 10


class Storage:
    WAV_PATH = "/home/jupyter/wav"
    RAW_DATA_PATH = "/home/jupyter/"
    PARENT_FOLDER_PATH = "/home/jupyter/common-voice-train"
    DATA_CLIPS_PATH = r"C:\Users\ander\Documents\common-voice-data\clips"


class Train_Pipeline:
    TRAIN_DIR = "/home/jupyter/wav/gender/train_data"
    VAL_DIR = "/home/jupyter/wav/gender/val_data"
    TEST_DIR = "/home/jupyter/wav/gender/test_data"
