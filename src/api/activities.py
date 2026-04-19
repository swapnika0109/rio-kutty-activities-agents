import base64
import json

from fastapi import APIRouter, BackgroundTasks, Response, status
from pydantic import BaseModel

from ..workflows.activity_workflow import app_workflow
from ..services.database.firestore_service import FirestoreService
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

router = APIRouter(tags=["activities"])


class ActivityRequest(BaseModel):
    story_id: str
    age: str
    language: str = "en"


class PubSubMessage(BaseModel):
    message: dict = None
    subscription: str = None
    data: str = None


async def _run_activities_workflow(request: ActivityRequest):
    """WF5 only — existing backward-compatible path."""
    try:
        story = await FirestoreService().get_story(request.story_id)
        if not story:
            logger.error(f"Story {request.story_id} not found")
            return
        config = {
            "configurable": {
                "thread_id": request.story_id,
                "story_id": request.story_id,
                "story_text": story.get("story_text", ""),
                "age": request.age,
                "language": story.get("language", "en"),
            }
        }
        initial_state = {
            "activities": {},
            "completed": [],
            "errors": {},
            "retry_count": {},
            "status": "pending",
        }
        await app_workflow.ainvoke(initial_state, config=config)
        logger.info(f"WF5 completed for story {request.story_id}")
    except Exception as e:
        logger.exception(f"WF5 failed for story {request.story_id}: {e}")


@router.post("/generate-activities")
async def generate_activities(request: ActivityRequest, background_tasks: BackgroundTasks):
    """Existing endpoint — triggers WF5 activities only (backward compatible)."""
    logger.info(f"Received request for story {request.story_id}")
    background_tasks.add_task(_run_activities_workflow, request)
    return {"status": "accepted", "message": "Activity generation started", "story_id": request.story_id}


@router.post("/pubsub-handler")
async def pubsub_handler(pubsub_msg: PubSubMessage, background_tasks: BackgroundTasks):
    """Handles Cloud Pub/Sub push messages."""
    logger.info(f"Received pubsub message: {pubsub_msg}")

    data = None
    if pubsub_msg.data:
        data = pubsub_msg.data
    elif pubsub_msg.message and "data" in pubsub_msg.message:
        data = pubsub_msg.message["data"]

    if not data:
        logger.error("No data found in pubsub message")
        return Response(status_code=status.HTTP_400_BAD_REQUEST)

    try:
        decoded_data = base64.b64decode(data).decode("utf-8")
        logger.info(f"Received Message: {decoded_data}")
        data_json = json.loads(decoded_data)

        activity_request = ActivityRequest(
            story_id=data_json["story_id"],
            age=data_json["age"],
            language=data_json.get("language", "en"),
        )
        background_tasks.add_task(_run_activities_workflow, activity_request)
        return Response(status_code=status.HTTP_202_ACCEPTED)
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        return Response(status_code=status.HTTP_400_BAD_REQUEST)
    except Exception as e:
        logger.error(f"Error processing pubsub message: {e}")
        return Response(status_code=status.HTTP_400_BAD_REQUEST)
