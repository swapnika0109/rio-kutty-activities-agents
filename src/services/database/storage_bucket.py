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

    async def upload_file(self, filename: str, file_content: bytes | str):
        """Dedicated method for the lines you checked"""
        try:
            bucket = self.client.bucket(self._bucket_name)
            blob = bucket.blob(filename)
            # Note: storage calls are synchronous; in a high-traffic async app, 
            # you might want to run this in a threadpool
            blob.upload_from_string(file_content)
            logger.info(f"Uploaded {filename} to {self._bucket_name}")
            return True
        except Exception as e:
            logger.error(f"Storage upload failed: {str(e)}")
            return False

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