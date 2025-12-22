from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
import uvicorn
import base64
import json
import os
from .workflows.activity_workflow import app_workflow
from .services.database.firestore_service import FirestoreService
from .utils.logger import setup_logger

logger = setup_logger(__name__)

logger.info("Starting application...")

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    logger.info("Application startup complete.")

class ActivityRequest(BaseModel):
    story_id: str
    age: int
    language: str = "en"

class PubSubMessage(BaseModel):
    message: dict
    subscription: str

async def run_workflow(request: ActivityRequest):
    """Background task to run the LangGraph workflow"""
    try:
        # Get the story from the database
        story = await FirestoreService().get_story(request.story_id)
        if not story:
            logger.error(f"Story with ID {request.story_id} not found")
            raise Exception(f"Story {request.story_id} not found")

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
            "retry_count": {}
        }
        
        # Invoke the workflow
        # Note: In production, use a persistent checkpointer (e.g. Postgres) 
        # instead of MemorySaver to handle restarts.
        await app_workflow.ainvoke(initial_state, config=config)
        logger.info(f"Workflow completed for story {request.story_id}")
        
    except Exception as e:
        logger.error(f"Workflow failed for story {request.story_id}: {e}")
        

@app.post("/generate-activities")
async def generate_activities(request: ActivityRequest, background_tasks: BackgroundTasks):
    """
    Endpoint called by the Go backend.
    Returns immediately (202 Accepted) and processes in background.
    """
    logger.info(f"Received request for story {request.story_id}")
    background_tasks.add_task(run_workflow, request)
    return {"status": "accepted", "message": "Activity generation started", "story_id": request.story_id}

@app.post("/pubsub-handler")
async def pubsub_handler(pubsub_msg: PubSubMessage):
    """
    Endpoint called by the Go backend.
    Returns immediately (202 Accepted) and processes in background.
    """
    logger.info(f"Received request for pubsub activity generation {request}")
    if "data" not in pubsub_msg:
        return Response(status_code=400, message="Invalid pubsub message")
    
    try:
        decoded_data = base64.b64decode(pubsub_msg["data"]).decode("utf-8")
        data_json = json.loads(decoded_data)
        logger.info(f"Received Message: {decoded_data}")
        ActivityRequest(story_id=data_json["story_id"], age=data_json["age"], language=data_json["language"])
        background_tasks.add_task(run_workflow, request)
        return Response(status_code=status.HTTP_202_ACCEPTED, message="Activity generation started")
    except Exception as e:
        logger.error(f"Error processing pubsub message: {e}")
        return Response(status_code=status.HTTP_400_BAD_REQUEST, message="Invalid pubsub message")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)