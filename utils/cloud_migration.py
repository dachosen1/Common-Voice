import os

from google.cloud import storage
from tqdm import tqdm

from model.RNN_Model.config import config


def upload_from_file(file, bucket, bucket_folder) -> None:
    """
    Upload document to a Google GCP bucket.
    :param file: file name of the document in the directory to upload
    :param bucket: Google CCP bucket name
    :param bucket_folder: create or upload to an existing folder with a GCP bucket
    """

    file_name = file.split(".")[0]
    blob = bucket.blob(f"{bucket_folder}/{file_name}")
    blob.upload_from_filename(file)
    print(f"Successfully uploaded {file} to the {bucket_folder} folder of the {bucket}")


def local_migration(local_directory, authentication_key, storage_bucket):
    """
    Migrate content from a local directory to a Google Cloud storage bucket.

    :param local_directory: Path to the the directory on local machine
    :param authentication_key: Path Google Cloud Authentication key JSON. To obtain an authentication please see below.

    Set GOOGLE_APPLICATION_CREDENTIALS or explicitly create credentials and re-run the application. For more information,
    please see https://cloud.google.com/docs/authentication/getting-started.

    After the key has been obtained, download it to your machine and provide the path to the location.

    :param storage_bucket: Google CGP storage bucket name.
    """

    AUTH_KEY = authentication_key
    storage_client = storage.Client.from_service_account_json(AUTH_KEY)
    bucket = storage_client.bucket(storage_bucket)
    all_files = os.listdir(os.chdir(local_directory))

    [
        upload_from_file(file = file, bucket = bucket, bucket_folder = "clips")
        for file in tqdm(all_files)
    ]


if __name__ == "__main__":
    local_migration(
        local_directory = r"C:\Users\ander\Documents\en\clips",
        authentication_key = config.Auth.AUTH_KEY_PATH,
        storage_bucket = config.Bucket.RAW_DATA,
    )
