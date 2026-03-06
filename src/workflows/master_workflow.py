"""
Master Workflow — orchestrates the full story pipeline.

This module does NOT manage WF1 or WF2 directly — those are triggered by separate
API calls (POST /generate-topics → human picks → POST /select-topic).
The master workflow is responsible only for the PARALLEL phase:

    POST /generate-media/{story_id}
        ↓
    [dispatch_parallel]  ← asyncio.gather over 3 compiled subgraphs
      ├→ WF3 image_workflow
      ├→ WF4 audio_workflow
      └→ WF5 activity_workflow (existing)
        ↓
    [collect_and_check]
      ├→ all completed → [finalize] → END
      └→ any needs_human → [publish_hitl_notification] → interrupt() ← pause
                           → admin calls POST /resume-workflow
                           → [handle_decision] → END

Why asyncio.gather instead of LangGraph Send()?
- WF3, WF4, WF5 are fully self-contained compiled subgraphs that manage their
  own state, retries, and checkpointing. They don't share state with each other.
- asyncio.gather gives true Python-level parallelism with a clean fan-in pattern.
- Each subgraph checkpoints independently under its own thread_id.
- Master just calls them, collects results, and handles failures.

Human-in-loop (HITL) pattern:
- LangGraph interrupt() suspends the graph at the collect_results_node.
- Full state persists in FirestoreCheckpointer under the master's thread_id.
- Admin calls POST /resume-workflow with {thread_id, decision}.
- LangGraph resumes via graph.ainvoke(None, config, command=Command(resume=decision)).
"""

import asyncio
import json
import os
from typing import Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from langchain_core.runnables import RunnableConfig
from google.cloud import pubsub_v1

from ..models.state import MasterWorkflowState
from ..workflows.image_workflow import image_workflow
from ..workflows.audio_workflow import audio_workflow
from ..workflows.activity_workflow import app_workflow as activity_workflow
from ..services.database.firestore_service import FirestoreService
from ..services.database.checkpoint_service import FirestoreCheckpointer
from ..utils.logger import setup_logger
from ..utils.config import get_settings

logger = setup_logger(__name__)
settings = get_settings()

firestore = FirestoreService()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sub_thread_id(story_id: str, wf: str) -> str:
    """Each subgraph gets its own thread_id so their checkpoints don't collide."""
    return f"{story_id}_{wf}"


def _build_sub_config(
    story_id: str, wf: str, story: dict, age: str, language: str, theme: str
) -> dict:
    """Build config.configurable for a subgraph invocation."""
    return {
        "configurable": {
            "thread_id": _sub_thread_id(story_id, wf),
            "story_id": story_id,
            "story_text": story.get("story_text", ""),
            "story_title": story.get("title", ""),
            "age": age,
            "language": language,
            "theme": theme,
        }
    }


def _publish_hitl_notification(story_id: str, failed_workflows: list[dict]) -> None:
    """
    Publishes a Pub/Sub message to notify admin of workflows needing human review.
    Non-blocking — if publish fails, we log and continue (interrupt() still fires).
    """
    topic = settings.HUMAN_LOOP_NOTIFICATION_TOPIC
    if not topic:
        logger.warning("[Master] HUMAN_LOOP_NOTIFICATION_TOPIC not configured — skipping Pub/Sub")
        return

    try:
        publisher = pubsub_v1.PublisherClient()
        data = json.dumps({
            "story_id": story_id,
            "failed_workflows": failed_workflows,
            "action_required": "Review and resume via POST /resume-workflow",
        }).encode("utf-8")
        future = publisher.publish(topic, data)
        future.result(timeout=5)
        logger.info(f"[Master] HITL notification published to {topic}")
    except Exception as e:
        logger.error(f"[Master] Failed to publish HITL notification: {e}")


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

async def dispatch_parallel_node(state: MasterWorkflowState, config: RunnableConfig) -> dict:
    """
    Dispatches WF3, WF4, WF5 in parallel using asyncio.gather.

    Why gather instead of LangGraph's parallel edges?
    Each sub-workflow is a fully compiled graph with its own state and checkpoint
    thread. gather lets us run them truly concurrently and collect their final
    states atomically, without needing to model their internal state in master.
    """
    cfg = config.get("configurable", {})
    story_id = state.get("story_id") or cfg.get("story_id")
    age = cfg.get("age", "3-4")
    language = cfg.get("language", "English")
    theme = cfg.get("theme", "theme1")
    story = state.get("story") or {}

    logger.info(f"[Master] Dispatching parallel workflows for story_id={story_id} theme={theme}")

    wf3_config = _build_sub_config(story_id, "wf3", story, age, language, theme)
    wf4_config = _build_sub_config(story_id, "wf4", story, age, language, theme)
    wf5_config = {
        "configurable": {
            "thread_id": _sub_thread_id(story_id, "wf5"),
            "story_id": story_id,
            "story_text": story.get("story_text", ""),
            "age": age,
            "language": language,
        }
    }

    wf3_initial = {
        "story_text": story.get("story_text", ""),
        "story_title": story.get("title", ""),
        "retry_count": 0,
        "status": "pending",
        "completed": [],
        "errors": {},
    }
    wf4_initial = {
        "story_text": story.get("story_text", ""),
        "language": language,
        "voice": settings.TTS_VOICE_NAME,
        "retry_count": 0,
        "status": "pending",
        "completed": [],
        "errors": {},
    }
    wf5_initial = {
        "activities": {},
        "images": {},
        "completed": [],
        "errors": {},
        "retry_count": {},
        "status": "pending",
    }

    # Run all 3 in parallel; return_exceptions=True so one failure doesn't cancel others
    results = await asyncio.gather(
        image_workflow.ainvoke(wf3_initial, config=wf3_config),
        audio_workflow.ainvoke(wf4_initial, config=wf4_config),
        activity_workflow.ainvoke(wf5_initial, config=wf5_config),
        return_exceptions=True,
    )

    wf3_result, wf4_result, wf5_result = results

    def _extract_status(result, wf_id: str) -> str:
        if isinstance(result, Exception):
            logger.error(f"[Master] {wf_id} raised exception: {result}")
            return "needs_human"
        return result.get("status", "needs_human")

    statuses = {
        "wf3": _extract_status(wf3_result, "wf3"),
        "wf4": _extract_status(wf4_result, "wf4"),
        "wf5": _extract_status(wf5_result, "wf5"),
    }
    errors = {}
    for wf_id, result in [("wf3", wf3_result), ("wf4", wf4_result), ("wf5", wf5_result)]:
        if isinstance(result, Exception):
            errors[wf_id] = str(result)
        elif result.get("errors"):
            errors[wf_id] = str(result["errors"])

    logger.info(f"[Master] Parallel results: {statuses}")
    return {
        "workflow_statuses": statuses,
        "errors": errors,
    }


async def collect_results_node(state: MasterWorkflowState, config: RunnableConfig) -> dict:
    """
    Checks parallel workflow statuses.
    If any workflow needs human review, publishes Pub/Sub and calls interrupt().
    interrupt() suspends the graph here and persists state to Firestore.
    The graph resumes when admin calls POST /resume-workflow.
    """
    statuses = state.get("workflow_statuses", {})
    failed = [
        {"workflow_id": wf_id, "error": state.get("errors", {}).get(wf_id, "unknown")}
        for wf_id, status in statuses.items()
        if status == "needs_human"
    ]

    if not failed:
        logger.info("[Master] All parallel workflows completed successfully")
        return {}

    story_id = state.get("story_id")
    logger.warning(f"[Master] {len(failed)} workflow(s) need human review: {[f['workflow_id'] for f in failed]}")

    _publish_hitl_notification(story_id, failed)

    # interrupt() suspends here. The value passed is the interrupt "payload"
    # visible to the admin when they query the workflow state.
    # Graph resumes when admin calls: graph.ainvoke(None, config, command=Command(resume=decision))
    decision = interrupt({
        "message": "One or more workflows failed and need human review",
        "story_id": story_id,
        "failed_workflows": failed,
        "instructions": "Call POST /resume-workflow with decision: 'retry', 'skip', or 'override'",
    })

    # decision is the value passed by admin via Command(resume=decision)
    human_decisions = {f["workflow_id"]: decision for f in failed}
    return {
        "human_decisions": human_decisions,
        "workflow_statuses": {
            **statuses,
            **{f["workflow_id"]: "human_loop" for f in failed},
        },
    }


async def handle_decision_node(state: MasterWorkflowState, config: RunnableConfig) -> dict:
    """
    Handles admin's decision after HITL:
    - "retry": re-invoke the failed workflow(s) (not implemented here — admin re-triggers via API)
    - "skip":  mark workflow as skipped and proceed to finalize
    - "override": treat as completed (admin has manually handled the issue)
    """
    decisions = state.get("human_decisions", {})
    statuses = dict(state.get("workflow_statuses", {}))

    for wf_id, decision in decisions.items():
        if decision in ("skip", "override"):
            statuses[wf_id] = "skipped"
            logger.info(f"[Master] Admin decision for {wf_id}: {decision}")
        else:
            logger.info(f"[Master] Admin decision for {wf_id}: retry — admin should re-trigger")

    return {"workflow_statuses": statuses}


async def finalize_node(state: MasterWorkflowState, config: RunnableConfig) -> dict:
    """Mark overall pipeline as complete."""
    story_id = state.get("story_id")
    statuses = state.get("workflow_statuses", {})
    logger.info(f"[Master] Pipeline finalized for story_id={story_id}: {statuses}")
    return {}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def route_after_collect(
    state: MasterWorkflowState,
) -> Literal["finalize", "handle_decision"]:
    """Route to finalize if all done, or handle_decision after HITL resume."""
    statuses = state.get("workflow_statuses", {})
    if any(s == "human_loop" for s in statuses.values()):
        return "handle_decision"
    return "finalize"


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

master = StateGraph(MasterWorkflowState)

master.add_node("dispatch_parallel", dispatch_parallel_node)
master.add_node("collect_results", collect_results_node)
master.add_node("handle_decision", handle_decision_node)
master.add_node("finalize", finalize_node)

master.set_entry_point("dispatch_parallel")
master.add_edge("dispatch_parallel", "collect_results")
master.add_conditional_edges(
    "collect_results",
    route_after_collect,
    {"finalize": "finalize", "handle_decision": "handle_decision"},
)
master.add_edge("handle_decision", "finalize")
master.add_edge("finalize", END)

if os.environ.get("USE_MEMORY_CHECKPOINTER", "false").lower() == "true":
    checkpointer = MemorySaver()
else:
    checkpointer = FirestoreCheckpointer()

master_workflow = master.compile(checkpointer=checkpointer)
