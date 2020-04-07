import collections
import os

import gcsfs
import google
import pandas as pd
from google.cloud import storage
from tqdm import tqdm

from model.config import config


def upload_from_file(file, bucket, bucket_folder) -> None:
    """
    Upload document to a Google GCP bucket.
    :param file: name name of the document in the directory to upload
    :param bucket: Google CCP bucket name
    :param bucket_folder: create or upload to an existing folder with a GCP bucket
    """

    file_name = file.split(".")[0]
    blob = bucket.blob(f"{bucket_folder}/{file_name}")
    blob.upload_from_filename(file)
    print(f"Uploaded {file} to the {bucket_folder} folder of the {bucket}")

    os.remove(file)


def bucket_migration(local_directory, authentication_key, storage_bucket) -> None:
    """
    Migrate content from a local directory to a Google Cloud storage bucket.

    :param local_directory: Path to the the directory on local machine
    :param authentication_key: Path Google Cloud Authentication key JSON. To obtain an authentication please see below.

    Set GOOGLE_APPLICATION_CREDENTIALS or explicitly create credentials and re-run the application. For more
    information, please see https://cloud.google.com/docs/authentication/getting-started.

    After the key has been obtained, download it to your machine and provide the path to the location.

    :param storage_bucket: Google CGP storage bucket name.

    Example:

    bucket_migration(
        local_directory = r"C:\Users\ander\Documents\en\clips",
        authentication_key = config.Auth.AUTH_KEY_PATH,
        storage_bucket = config.Bucket.RAW_DATA,
    )
    """

    AUTH_KEY = authentication_key
    storage_client = storage.Client.from_service_account_json(AUTH_KEY)
    bucket = storage_client.bucket(storage_bucket)
    all_files = os.listdir(os.chdir(local_directory))

    [
        upload_from_file(file = file, bucket = bucket, bucket_folder = "clips")
        for file in tqdm(all_files)
    ]


def list_blobs(bucket_name):
    """Lists all the blobs in the bucket."""
    storage_client = storage.Client()
    blobs = storage_client.list_blobs(bucket_name)
    list_files = [blob.name for blob in tqdm(blobs)]

    return list_files


def delete_mp3_from_bucket(file_list: list, bucket: google.cloud.storage.bucket.Bucket) -> None:
    """
    Iterates through a list of file names, and removes them from a google cloud bucket if they exist.
    :param file_list:  list of file names
    :param bucket: Google cloud storage objects bucket
    """

    for mp3 in tqdm(file_list):
        file_name = mp3.split('.')[0]
        delete_blob(bucket, file_name)


def delete_blob(bucket_blob: google.cloud.storage.bucket.Bucket, blob_name: str) -> None:
    """Deletes a blob from the bucket."""

    blob = bucket_blob.blob(f'clips/{blob_name}')
    blob.delete()


def delete_extra_file(name: chr, bucket: str) -> None:
    """
    Function was created to load all the files in the delete and remove those files by name in the raw folders. Those
    files are removed since they are not labeled with either Gender, Age, or  Country of Origin.

    The files in the deleted bucket are saved in one of 6 folders.'validated', 'train', 'test', 'dev', 'other', 'invalidated'

    They represent the original folders in common voice dataset

    :param name: File name to upload
    :param bucket: Bucket name

    Example:


    file_list = ['validated', 'train', 'test', 'dev', 'other', 'invalidated']

    for file in file_list:
        delete_extra_file(name = file, bucket = config.Bucket.RAW_DATA)

    """
    storage_client = storage.Client()
    project = storage_client.project

    fs = gcsfs.GCSFileSystem(project = project)

    with fs.open(f"{config.Bucket.META_DATA}/delete/{name}") as f:
        data = pd.read_csv(f)

    mp3_to_remove = data.iloc[:, 1].to_list()
    bucket = storage_client.bucket(bucket)
    delete_mp3_from_bucket(file_list = mp3_to_remove, bucket = bucket)


def upload_blob(bucket_name: str, source_file_name: str, destination_blob_name: str) -> object:
    """

    Upload file to a google cloud cloud bucket

    :param bucket_name:
    :param source_file_name:
    :param destination_blob_name:

    example:
    upload_blob(bucket_name = config.Bucket.META_DATA, source_file_name = 'test.csv',
    destination_blob_name = 'blob_name/test)

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
    """

    Find all the file names that do not have one of the 3 labels of age, gender and accent

    :param bucket: name of Google Cloud bucket
    :param name:  name of file or google cloud  blob
    :param project: unique google cloud project name



    file_list = ['validated', 'train', 'test', 'dev', 'other', 'invalidated']

    for file in file_list:
        clean_bucket(bucket = config.Bucket.META_DATA, name = file, project = 'common-voice-270516')
    """

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
