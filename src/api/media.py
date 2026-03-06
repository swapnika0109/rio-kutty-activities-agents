from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from ..workflows.master_workflow import master_workflow
from ..services.database.firestore_service import FirestoreService
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

router = APIRouter(tags=["media"])


class GenerateMediaRequest(BaseModel):
    story_id: str
    age: str
    language: str = "en"


class ResumeWorkflowRequest(BaseModel):
    thread_id: str
    decision: str  # "retry" | "skip" | "override"


async def _run_master_workflow(request: GenerateMediaRequest):
    try:
        story = await FirestoreService().get_story(request.story_id)
        if not story:
            logger.error(f"Story {request.story_id} not found for media generation")
            return
        config = {
            "configurable": {
                "thread_id": f"{request.story_id}_master",
                "story_id": request.story_id,
                "age": request.age,
                "language": story.get("language", request.language),
            }
        }
        initial_state = {
            "story_id": request.story_id,
            "story": story,
            "topics": None,
            "workflow_statuses": {},
            "workflow_retries": {},
            "human_loop_requests": {},
            "human_decisions": {},
            "errors": {},
        }
        await master_workflow.ainvoke(initial_state, config=config)
        logger.info(f"Master workflow completed for story {request.story_id}")
    except Exception as e:
        logger.exception(f"Master workflow failed for story {request.story_id}: {e}")


@router.post("/generate-media/{story_id}", status_code=202)
async def generate_media(story_id: str, request: GenerateMediaRequest, background_tasks: BackgroundTasks):
    """Triggers parallel WF3 (image) + WF4 (audio) + WF5 (activities)."""
    request.story_id = story_id
    logger.info(f"Media generation triggered for story_id={story_id}")
    background_tasks.add_task(_run_master_workflow, request)
    return {"status": "accepted", "message": "Media generation started", "story_id": story_id}


@router.post("/resume-workflow", status_code=202)
async def resume_workflow(request: ResumeWorkflowRequest):
    """Resumes a human-in-loop workflow after admin review. decision: retry | skip | override"""
    from langgraph.types import Command
    logger.info(f"Resuming workflow thread_id={request.thread_id} with decision={request.decision}")
    try:
        config = {"configurable": {"thread_id": request.thread_id}}
        await master_workflow.ainvoke(None, config=config, command=Command(resume=request.decision))
        return {"status": "resumed", "thread_id": request.thread_id, "decision": request.decision}
    except Exception as e:
        logger.error(f"Resume failed for thread_id={request.thread_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/workflow-status/{story_id}")
async def workflow_status(story_id: str):
    """Returns status of all workflows for a story."""
    db = FirestoreService()
    return await db.get_workflow_status(story_id)
