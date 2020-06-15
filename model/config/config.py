import pathlib
import model

PACKAGE_ROOT = pathlib.Path(model.__file__).resolve().parent
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_model"

GENDER_MODEL_NAME = "model_gender-"
AGE_MODEL_NAME = "model_age-"
COUNTRY_MODEL_NAME = "model_country-"

FRAME = dict(SAMPLE_RATE=16000,
             FMAX=8000,
             N_MELS=128,
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
                   INPUT_SIZE=FRAME['N_MELS'],
                   BATCH_SIZE=256)

TRAIN_PARAM = dict(
    LEARNING_RATE=0.001,
    GRADIENT_CLIP=15,
    EPOCH=30)

ALL_PARAM = dict(Train=TRAIN_PARAM,
                 Model=MODEL_PARAM,
                 Frame=FRAME
                 )


class GcpStorage:
    CLIPS_DIR = "/home/jupyter/clips"
    ROOT_DIR = "/home/jupyter/"
    DEV_DIR = "/home/jupyter/common-voice-voice-train"


class GcpTrainPipeline:
    TRAIN_DIR = "/home/jupyter/wav/gender/train_data"
    VAL_DIR = "/home/jupyter/wav/gender/val_data"
    TEST_DIR = "/home/jupyter/wav/gender/test_data"


class LocalStorage:
    WAV_DIR = r"C:\Users\ander\Documents\common-voice-data\wav"
    ROOT_DIR = r"C:\Users\ander\Documents\common-voice-data"
    DEV_DIR = r"C:\Users\ander\Documents\common-voice-dev"
    CLIPS_DIR = r"C:\Users\ander\Documents\common-voice-data\clips"
    TRAIN_DIR = r"C:\Users\ander\Documents\common-voice-dev"


class LocalTrainPipeline:
    TRAIN_DIR = r"C:\Users\ander\Documents\common-voice-dev\gender\train_data"
    VAL_DIR = r"C:\Users\ander\Documents\common-voice-dev\gender\val_data"
    TEST_DIR = r"C:\Users\ander\Documents\common-voice-dev\gender\test_data"
