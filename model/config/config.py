import os

import model

PACKAGE_ROOT = os.path.dirname(model.__file__)
TRAINED_MODEL_DIR = os.path.join(PACKAGE_ROOT, "trained_model")


class Common_voice_models:
    class Frame:
        FRAME = dict(SAMPLE_RATE=44100, NUMCEP=13, NFILT=26, NFFT=1103, MASK_THRESHOLD=0.01)

    class Gender(Frame):
        TRAIN_PARAM = {'LEARNING_RATE': 0.001, 'GRADIENT_CLIP': 15, 'EPOCH': 1}
        OUTPUT = {0: "Female", 1: "Male"}
        NAME = "model_gender-"
        PARAM = {'HIDDEN_DIM': 128, 'NUM_LAYERS': 5, 'DROPOUT': 0.30, 'INPUT_SIZE': 13, 'BATCH_SIZE': 125,
                 'OUTPUT_SIZE': 2}
        LABEL = 'gender'

    class Age(Frame):
        TRAIN_PARAM = {'LEARNING_RATE': 0.001, 'GRADIENT_CLIP': 15, 'EPOCH': 1}
        OUTPUT = {
            0: "Fifties",
            1: "Fourties",
            2: "Teens",
            3: "Thirties",
            4: "Twenties",
        }

        NAME = "model_age-"
        PARAM = {'HIDDEN_DIM': 128, 'NUM_LAYERS': 5, 'DROPOUT': 0.30, 'INPUT_SIZE': 13, 'BATCH_SIZE': 125,
                 'OUTPUT_SIZE': 5}
        LABEL = 'age'

    class Country(Frame):
        TRAIN_PARAM = {'LEARNING_RATE': 0.001, 'GRADIENT_CLIP': 15, 'EPOCH': 1}
        OUTPUT = {0: "Australia", 1: "Canada", 2: "England", 3: "Indian", 4: "New Zealand", 5: "Scotland", 6: "USA"}
        NAME = "model_gender-"
        PARAM = {'HIDDEN_DIM': 128, 'NUM_LAYERS': 5, 'DROPOUT': 0.30, 'INPUT_SIZE': 13, 'BATCH_SIZE': 125,
                 'OUTPUT_SIZE': 2}
        LABEL = 'accent'


DO_NOT_INCLUDE = [
    "african",
    "hongkong",
    "ireland",
    "malaysia",
    "philippines",
    "singapore",
    "southatlandtic",
    "wales",
    "nineties",
    "seventies",
    "sixties",
]


class Bucket:
    RAW_DATA = "common-voice-voice-all"
    META_DATA = "common-voice-voice-data"
    VALIDATION_SET = "common-voice-voice-dev"
    TEST_SET = "common-voice-voice-test"
    TRAIN_SET = "common-voice-voice-train"


class DataDirectory:
    ROOT_DIR = r"C:\Users\ander\Documents\common-voice-data"
    DEV_DIR = r"C:\Users\ander\Documents\common-voice-dev"
    CLIPS_DIR = r"C:\Users\ander\Documents\common-voice-data\clips"
    TRAIN_DIR = r"C:\Users\ander\Documents\common-voice-train"


class TrainingTestingSplitDirectory:
    TRAIN_DIR = r"train_data"
    VAL_DIR = r"val_data"
    TEST_DIR = r"test_data"
