from google.cloud import storage

bucket_name = "common-voice-all"

storage_client = storage.Client()
bucket = storage_client.bucket(bucket_name)
