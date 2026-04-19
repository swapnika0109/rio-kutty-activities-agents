from google.cloud import storage
from ...utils.config import get_settings
from ...utils.logger import setup_logger

settings = get_settings()
logger = setup_logger(__name__)

class StorageBucketService: # Rename from FirestoreService
    def __init__(self):
        self._storage_client = None
        self._bucket_name = settings.GOOGLE_CLOUD_BUCKET

    @property
    def client(self):
        if self._storage_client is None:
            project = settings.GOOGLE_CLOUD_PROJECT
            if settings.GOOGLE_APPLICATION_CREDENTIALS:
                self._storage_client = storage.Client.from_service_account_json(
                    settings.GOOGLE_APPLICATION_CREDENTIALS,
                    project=project
                )
            else:
                self._storage_client = storage.Client(project=project)
        return self._storage_client

    async def upload_file(
        self,
        filename: str,
        file_content: bytes | str,
        content_type: str | None = None,
    ) -> str | None:
        """
        Uploads file to GCS using the service account credentials.
        Returns the public GCS URL on success, None on failure.
        URL format: https://storage.googleapis.com/{bucket}/{filename}
        """
        try:
            bucket = self.client.bucket(self._bucket_name)
            blob = bucket.blob(filename)
            blob.upload_from_string(file_content, content_type=content_type)
            url = f"https://storage.googleapis.com/{self._bucket_name}/{filename}"
            logger.info(f"Uploaded {filename} to {self._bucket_name} → {url}")
            return url
        except Exception as e:
            logger.error(f"Storage upload failed for {filename}: {str(e)}")
            return None

    async def delete_file(self, filename: str):
        try:
            logger.info(f"Deleting {filename} from {self._bucket_name}")
            bucket = self.client.bucket(self._bucket_name)
            blob = bucket.blob(filename)
            blob.delete()
            logger.info(f"Deleted {filename} from {self._bucket_name}")
            return True
        except Exception as e:
            logger.error(f"Storage delete failed: {str(e)}")
            return False