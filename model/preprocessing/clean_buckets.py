import collections

import gcsfs
import pandas as pd
from google.cloud import storage
from tqdm import tqdm

from model.config import config


def upload_blob(bucket_name: str, source_file_name: str, destination_blob_name: str) -> object:
    """

    :param bucket_name:
    :param source_file_name:
    :param destination_blob_name:
    :return:

    """
    storage_client = storage.Client.from_service_account_json(config.Auth.AUTH_KEY_PATH)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print(
        "File {} uploaded to {}.".format(
            source_file_name, destination_blob_name
        )
    )


def clean_bucket(bucket: str, name: str, project: str) -> None:
    'Find all the file names that do not have one of the 3 labels of age, gender and accent'

    fs = gcsfs.GCSFileSystem(project = project)

    with fs.open('{}/{}.tsv'.format(bucket, name)) as f:
        data = pd.read_csv(f, delimiter = '\t')

    data = data[['path', 'age', 'gender', 'accent']]

    print('There are {} audio files in the development set'.format(data.shape[0]))

    columns = ['gender', 'age', 'accent']
    clean_files = []

    for column in columns:
        mp3 = data[-data[column].isna()]['path']
        clean_files.extend(mp3)

    clean_files = set(clean_files)

    path = data['path']
    mp3_to_remove = collections.deque()

    for mp3 in tqdm(path):
        if mp3 not in clean_files:
            mp3_to_remove.append(mp3)

    clean_files = pd.DataFrame(list(mp3_to_remove))
    clean_files.to_csv('remove-{}.csv'.format(name))

    upload_blob(bucket_name = config.Bucket.META_DATA, source_file_name = 'remove-{}.csv'.format(name),
                destination_blob_name = 'subject_to_removal/{}'.format(name))

    print(
        '{} mp3s do not have labels, leaving {} in the {} labeled mp3'.format(len(clean_files),
                                                                              data.shape[0] - len(clean_files),
                                                                              name))
    print('Removed {}% of the data'.format(round((len(mp3_to_remove) / data.shape[0]) * 100, 2)))


if __name__ == '__main__':
    file_list = ['validated', 'train', 'test', 'dev', 'other', 'invalidated']

    for file in file_list:
        clean_bucket(bucket = config.Bucket.META_DATA, name = file, project = 'common-voice-270516')
