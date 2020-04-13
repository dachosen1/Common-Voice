class Bucket:
    RAW_DATA = "common-voice-all"
    META_DATA = "common-voice-data"
    VALIDATION_SET = "common-voice-dev"
    TEST_SET = "common-voice-test"
    TRAIN_SET = "common-voice-train"


class Model:
    OUTPUT_SIZE = 2
    HIDDEN_DIM = 256
    NUM_LAYERS = 3
    DROPOUT = 0.25
    INPUT_SIZE = 648
    BATCH_SIZE = 1024


class Train:
    LEARNING_RATE = 0.0001
    GRADIENT_CLIP = 10
    EPOCH = 200


class Storage:
    WAV_PATH = r"C:\Users\ander\Documents\common-voice-all\wav"
    RAW_DATA_PATH = r"C:\Users\ander\Documents\common-voice-all"
    PARENT_FOLDER_PATH = r"C:\Users\ander\Documents\common-voice-dev"


class Train_Pipeline:
    TRAIN_DIR = r"/home/an2908/common-voice-dev/gender/train_data"
    VAL_DIR = r"/home/an2908/common-voice-dev/gender/val_data"
