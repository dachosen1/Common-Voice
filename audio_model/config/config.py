import os

import audio_model

PACKAGE_ROOT = os.path.dirname(audio_model.__file__)
TRAINED_MODEL_DIR = os.path.join(PACKAGE_ROOT, "trained_model")


class CommonVoiceModels:

    class Frame:
        FRAME = dict(SAMPLE_RATE=22050, NUMCEP=13, NFILT=26, NFFT=2048, TOP_DB=10, FMAX=8000, N_MELS=128)

    class Gender(Frame):
        OUTPUT = {0: "Female",
                  1: "Male"}

        NAME = "model_gender-"
        PARAM = {'HIDDEN_DIM': 256, 'NUM_LAYERS': 4, 'DROPOUT': 0.3, 'INPUT_SIZE': 128, 'BATCH_SIZE': 525,
                 'OUTPUT_SIZE': 2, 'LEARNING_RATE': 0.001, 'GRADIENT_CLIP': 15, 'EPOCH': 20}
        LABEL = 'gender'

    class Age(Frame):
        OUTPUT = {
            0: "Fifties",
            1: "Fourties",
            2: "Teens",
            3: "Thirties",
            4: "Twenties",
        }

        NAME = "model_age-"
        PARAM = {'HIDDEN_DIM': 128, 'NUM_LAYERS': 5, 'DROPOUT': 0, 'INPUT_SIZE': 128, 'BATCH_SIZE': 128,
                 'OUTPUT_SIZE': 5, 'LEARNING_RATE': 0.01, 'GRADIENT_CLIP': 0, 'EPOCH': 1}
        LABEL = 'age'

    class Country(Frame):
        OUTPUT = {0: 'Australia',
                  1: 'Canada',
                  2: 'England',
                  3: 'Indian',
                  4: 'American'}

        NAME = "model_country-"
        PARAM = {'HIDDEN_DIM': 8, 'NUM_LAYERS': 16, 'DROPOUT': 0.0, 'INPUT_SIZE': 128, 'BATCH_SIZE': 1024,
                 'OUTPUT_SIZE': 5, 'LEARNING_RATE': 0.01, 'GRADIENT_CLIP': 0, 'EPOCH': 5}
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
    'bermuda',
    'newzealand',
    'scotland',
    'eighties'
]


class Bucket:
    RAW_DATA = "common-voice-voice-all"
    META_DATA = "common-voice-voice-data"
    VALIDATION_SET = "common-voice-voice-dev"
    TEST_SET = "common-voice-voice-test"
    TRAIN_SET = "common-voice-voice-train"


class DataDirectory:
    DATA_DIR = r"C:\Users\ander\Documents\common-voice-data"
    DEV_DIR = r"C:\Users\ander\Documents\common-voice-dev"
    CLIPS_DIR = r"C:\Users\ander\Documents\cv-corpus-5.1-2020-06-22\en\clips"
    TRAIN_DIR = r"C:\Users\ander\Documents\common-voice-train"


# class DataDirectory:
#     DATA_DIR = "/home/jupyter/common-voice-data"
#     DEV_DIR = "/home/jupyter/common-voice-dev"
#     CLIPS_DIR = "/home/jupyter/common-voice-data/clips"

class TrainingTestingSplitDirectory:
    TRAIN_DIR = r"train_data"
    VAL_DIR = r"val_data"
    TEST_DIR = r"test_data"


class GoogleCloud:
    MODEL_BUCKET_NAME = 'audio_model-version'
    TRAINED_MODEL_DIR = 'Trained_Models'
