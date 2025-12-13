from google.cloud import firestore
from ...utils.config import get_settings
from ...utils.logger import setup_logger

settings = get_settings()
logger = setup_logger(__name__)

class FirestoreService:
    def __init__(self):
        if settings.GOOGLE_APPLICATION_CREDENTIALS:
             self.db = firestore.Client.from_service_account_json(
                json_credentials_path=settings.GOOGLE_APPLICATION_CREDENTIALS,
                project=settings.GOOGLE_CLOUD_PROJECT, 
                database=settings.FIRESTORE_DATABASE
            )
        else:
            self.db = firestore.Client(
                project=settings.GOOGLE_CLOUD_PROJECT, 
                database=settings.FIRESTORE_DATABASE
            )

    async def save_activity(self, story_id: str, activity_type: str, activity_data: dict):
        """
        Saves an activity and updates the story status in a single batch.
        """
        try:
            batch = self.db.batch()
            
            # 1. Save the actual activity data
            # Assuming a collection structure like: stories/{story_id}/activities/{activity_type}
            # Or a separate top-level collection. Let's use top-level as per plan.
            activity_ref = self.db.collection('activities_v1').document() # Auto-ID
            batch.set(activity_ref, {
                **activity_data,
                'story_id': story_id,
                'type': activity_type,
                'created_at': firestore.SERVER_TIMESTAMP
            })

            # 2. Update the parent story to say this activity is ready
            story_ref = self.db.collection('riostories_v3').document(story_id)
            batch.update(story_ref, {
                f'activities.{activity_type}': 'ready',
                'updated_at': firestore.SERVER_TIMESTAMP
            })

            # Commit both atomically
            batch.commit()
            logger.info(f"Saved {activity_type} for story {story_id}")
            
        except Exception as e:
            logger.error(f"Firestore save failed: {str(e)}")
            raise e