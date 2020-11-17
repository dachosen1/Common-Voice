import os

from audio_model import audio_model

PACKAGE_ROOT = os.path.dirname(audio_model.__file__)
TRAINED_MODEL_DIR = os.path.join(PACKAGE_ROOT, "trained_model")

FRAME = dict(SAMPLE_RATE=16000, NUMCEP=13, NFILT=26, NFFT=2048, TOP_DB=40, FMAX=8000, N_MELS=128, MIN_LEVEL_DB=-100,
             HOP_LENGTH=275, MASK_THRESHOLD=0.01, WIN_LENGTH=1100, REF_LEVEL_DB=20)


class Gender:
    OUTPUT = {0: "Female",
              1: "Male"}

    NAME = "model_gender-"
    PARAM = {'HIDDEN_DIM': 120, 'NUM_LAYERS': 2, 'DROPOUT': 0.3, 'INPUT_SIZE': 128, 'BATCH_SIZE': 512,
             'OUTPUT_SIZE': 2, 'LEARNING_RATE': 0.001, 'GRADIENT_CLIP': 15, 'EPOCH': 40}
    LABEL = 'gender'


class Age:
    OUTPUT = {
        0: "Fifties",
        1: "Fourties",
        2: "Teens",
        3: "Thirties",
        4: "Twenties",
    }

    NAME = "model_age-"
    PARAM = {'HIDDEN_DIM': 256, 'NUM_LAYERS': 4, 'DROPOUT': 0, 'INPUT_SIZE': 44, 'BATCH_SIZE': 525,
             'OUTPUT_SIZE': 5, 'LEARNING_RATE': 0.001, 'GRADIENT_CLIP': 15, 'EPOCH': 5}
    LABEL = 'age'


class Country:
    OUTPUT = {0: 'Australia',
              1: 'Canada',
              2: 'England',
              3: 'Indian',
              4: 'American'}

    NAME = "model_country-"
    PARAM = {'HIDDEN_DIM': 8, 'NUM_LAYERS': 16, 'DROPOUT': 0.3, 'INPUT_SIZE': 216, 'BATCH_SIZE': 256,
             'OUTPUT_SIZE': 5, 'LEARNING_RATE': 0.001, 'GRADIENT_CLIP': 20, 'EPOCH': 25}
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


class DataDirectory:
    DATA_DIR = r"C:\Users\ander\Documents\common-voice-data"
    DEV_DIR = r"C:\Users\ander\Documents\common-voice-dev"
    CLIPS_DIR = r"C:\Users\ander\Documents\common-voice-data\clips"


# class DataDirectory:
#     DATA_DIR = "/home/jupyter/comm-voice-training/common-voice-data"
#     DEV_DIR = "/home/jupyter/comm-voice-training/common-voice-dev"
#     CLIPS_DIR = "/home/jupyter/comm-voice-training/common-voice-data/clips"

class TrainingTestingSplitDirectory:
    TRAIN_DIR = r"train_data"
    VAL_DIR = r"val_data"
    TEST_DIR = r"test_data"


class GoogleCloud:
    MODEL_BUCKET_NAME = 'audio_model-version'
    TRAINED_MODEL_DIR = 'Trained_Models'
