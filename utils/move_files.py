import os
import shutil

import pandas as pd

TRAIN_MALE_PATH = "C:\\Users\\ander\\Documents\\train\\gender\\male"
TRAIN_FEMALE_PATH = "C:\\Users\\ander\\Documents\\train\\gender\\female"

DEV_MALE_PATH = "C:\\Users\\ander\\Documents\\dev\\gender\\male"
DEV_FEMALE_PATH = "C:\\Users\\ander\\Documents\\dev\\gender\\female"

TEST_MALE_PATH = "C:\\Users\\ander\\Documents\\test\\gender\\male"
TEST_FEMALE_PATH = "C:\\Users\\ander\\Documents\\test\\gender\\female"

CLIP_PATH = "C:\\Users\\ander\\Documents\\clips"
ALL_CLIPS = "C:\\Users\\ander\\Documents\\clips"

TRAIN_FILE_PATH = "data/invalidated.tsv"
DEV_FILE_PATH = "data/dev.tsv"
TEST_FILE_PATH = "data/test.tsv"


def get_male_female_file(PATH):
    """
    Module to search for return the files name for male and female
    """
    DATA = pd.read_csv(PATH, delimiter = "\t")
    GENDER_SPLIT = DATA[DATA["gender"].notna()]
    male_mp3_list = GENDER_SPLIT[GENDER_SPLIT["gender"] == "male"]["path"]
    female_mp3_list = GENDER_SPLIT[GENDER_SPLIT["gender"] == "female"]["path"]

    return male_mp3_list, female_mp3_list


def move_files(AUDIO_FILE_NAME, FILE_SAVE_PATH):
    """
    Module to search for audio file name in master audio file and move it to it's proper folder

    :param AUDIO_FILE_NAME
    :param: AUDIO_FILE_NAME

    """
    ALL_FILES = os.listdir(ALL_CLIPS)
    for audio_file in AUDIO_FILE_NAME:
        if audio_file in ALL_FILES:
            shutil.copyfile(
                os.path.join(CLIP_PATH, audio_file),
                os.path.join(FILE_SAVE_PATH, audio_file),
            )


def move_batch(FILE_PATH, MALE_PATH, FEMALE_PATH):
    male_mp3, female_mp3 = get_male_female_file(FILE_PATH)
    move_files(male_mp3, MALE_PATH)
    move_files(female_mp3, FEMALE_PATH)


if __name__ == "__main__":
    move_batch(TRAIN_FILE_PATH, TRAIN_MALE_PATH, TRAIN_FEMALE_PATH)
    move_batch(DEV_FILE_PATH, DEV_MALE_PATH, DEV_FEMALE_PATH)
    move_batch(TEST_FILE_PATH, TEST_MALE_PATH, TEST_FEMALE_PATH)
