import pathlib
import model

PACKAGE_ROOT = pathlib.Path(model.__file__).resolve().parent
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_model"

GENDER_MODEL_NAME = "model_gender-"
AGE_MODEL_NAME = "model_age-"
COUNTRY_MODEL_NAME = "model_country-"

FRAME = dict(SAMPLE_RATE=44100,
             NUMCEP=13,
             NFILT=26,
             NFFT=1103,
             MASK_THRESHOLD=0.01)

GENDER_LABEL = {0: 'female',
                1: 'male'}


class Bucket:
    RAW_DATA = "common-voice-voice-all"
    META_DATA = "common-voice-voice-data"
    VALIDATION_SET = "common-voice-voice-dev"
    TEST_SET = "common-voice-voice-test"
    TRAIN_SET = "common-voice-voice-train"


MODEL_PARAM = dict(OUTPUT_SIZE=2,
                   HIDDEN_DIM=128,
                   NUM_LAYERS=2,
                   DROPOUT=0.30,
                   INPUT_SIZE=13,
                   BATCH_SIZE=256)

TRAIN_PARAM = dict(
    LEARNING_RATE=0.001,
    GRADIENT_CLIP=15,
    EPOCH=4)

ALL_PARAM = dict(Train=TRAIN_PARAM,
                 Model=MODEL_PARAM,
                 Frame=FRAME
                 )


class Storage:
    ROOT_DIR = r"C:\Users\ander\Documents\common-voice-data"
    DEV_DIR = r"C:\Users\ander\Documents\common-voice-dev"
    CLIPS_DIR = r"C:\Users\ander\Documents\common-voice-data\clips"
    TRAIN_DIR = r"C:\Users\ander\Documents\common-voice-train"


class Pipeline:
    TRAIN_DIR = r"C:\Users\ander\Documents\common-voice-dev\gender\train_data"
    VAL_DIR = r"C:\Users\ander\Documents\common-voice-dev\gender\val_data"
    TEST_DIR = r"C:\Users\ander\Documents\common-voice-dev\gender\test_data"
