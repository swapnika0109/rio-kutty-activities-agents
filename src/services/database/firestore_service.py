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
            # 1. Find the story document by querying the 'story_id' field
            # (Since document ID != story_id)
            story_query = self.db.collection('riostories_v3').where("story_id", "==", story_id).limit(1)
            docs = list(story_query.stream())
            
            if not docs:
                logger.error(f"Story with ID {story_id} not found in riostories_v3")
                raise Exception(f"Story {story_id} not found")
            
            story_ref = docs[0].reference

            # 2. Prepare Batch
            batch = self.db.batch()
            
            # 3. Save the actual activity data
            activity_ref = self.db.collection('activities_v1').document() # Auto-ID
            batch.set(activity_ref, {
                **activity_data,
                'story_id': story_id,
                'type': activity_type,
                'created_at': firestore.SERVER_TIMESTAMP
            })

            # 4. Update the parent story status
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

    async def get_story(self, story_id: str):
        """
        Gets a story from the database by querying the story_id field.
        """
        try:
            query = self.db.collection('riostories_v3').where("story_id", "==", story_id).limit(1)
            docs = list(query.stream())
            
            if not docs:
                return None
                
            return docs[0].to_dict()

        except Exception as e:
            logger.error(f"Firestore get failed: {str(e)}")
            return None