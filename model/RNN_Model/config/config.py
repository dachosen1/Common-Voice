class Bucket:
    RAW_DATA = "common-voice-all"
    META_DATA = "common-voice-data"
    VALIDATION_SET = "common-voice-dev"
    TEST_SET = "common-voice-test"
    TRAIN_SET = "common-voice-train"


class Auth:
    AUTH_KEY_PATH = r"C:\Users\ander\Google Drive\Projects\Common Voice\Development\common-voice-6d593b0eee09.json"


class Model:
    OUTPUT_SIZE = 2
    HIDDEN_DIM = 256
    NUM_LAYERS = 10
    DROPOUT = 0.5
    BATCH = 1000
    LEARNING_RATE = 0.001
    GRADIENT_CLIP = 10
    EPOCH = 10
