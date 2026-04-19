"""
Langfuse tracing utility for LangGraph workflows.

Langfuse is free and open-source (MIT). Free cloud tier at cloud.langfuse.com.
Self-hostable via Docker if needed.

Setup:
1. Sign up at https://cloud.langfuse.com  (free)
2. Create a project and copy your Public Key + Secret Key
3. Add to .env:
       LANGFUSE_ENABLED=true
       LANGFUSE_PUBLIC_KEY=pk-lf-...
       LANGFUSE_SECRET_KEY=sk-lf-...
       LANGFUSE_HOST=https://cloud.langfuse.com   # or self-hosted URL

Usage in workflow invocations:
    from src.utils.tracing import get_trace_callbacks

    config = {
        "configurable": {...},
        "callbacks": get_trace_callbacks(name="WF2-story", metadata={"story_id": sid}),
    }
    await workflow.ainvoke(state, config=config)

What you see in the Langfuse dashboard:
- Each workflow run as a trace with a name and timeline
- Every LLM call (Gemini, FLUX) as a nested span with input/output/latency/cost
- Every LangGraph node as a named step
- Errors highlighted inline
- Full metadata (story_id, theme, age, language) on every trace
"""

from typing import Optional
from .config import get_settings
from .logger import setup_logger

logger = setup_logger(__name__)
settings = get_settings()

# Cached Langfuse client instance (None when disabled or keys missing)
_langfuse_client = None


def _get_client():
    """Lazy-init the Langfuse client. Returns None if disabled or misconfigured."""
    global _langfuse_client
    if _langfuse_client is not None:
        return _langfuse_client

    if not settings.LANGFUSE_ENABLED:
        return None
    if not settings.LANGFUSE_PUBLIC_KEY or not settings.LANGFUSE_SECRET_KEY:
        logger.warning("[Tracing] LANGFUSE_ENABLED=true but keys are missing — tracing disabled")
        return None

    try:
        from langfuse import Langfuse
        _langfuse_client = Langfuse(
            public_key=settings.LANGFUSE_PUBLIC_KEY,
            secret_key=settings.LANGFUSE_SECRET_KEY,
            host=settings.LANGFUSE_HOST,
        )
        logger.info(f"[Tracing] Langfuse connected: {settings.LANGFUSE_HOST}")
        return _langfuse_client
    except ImportError:
        logger.warning("[Tracing] langfuse package not installed — pip install langfuse")
        return None
    except Exception as e:
        logger.error(f"[Tracing] Langfuse init failed: {e}")
        return None


def get_trace_callbacks(
    name: str,
    metadata: Optional[dict] = None,
    tags: Optional[list[str]] = None,
    session_id: Optional[str] = None,
) -> list:
    """
    Returns a list of LangChain/LangGraph callbacks for Langfuse tracing.
    Returns an empty list when Langfuse is disabled — callers don't need to branch.

    Args:
        name:       Human-readable trace name shown in the dashboard
                    (e.g. "WF1-topics", "WF2-story", "WF3-image", "master")
        metadata:   Extra key-value data attached to the trace
                    (e.g. {"story_id": "abc", "theme": "theme1", "age": "3-4"})
        tags:       Labels for filtering in the dashboard (e.g. ["production", "batch"])
        session_id: Groups related traces into a session (e.g. use story_id or batch UUID)

    Example:
        callbacks = get_trace_callbacks(
            name="WF2-story",
            metadata={"story_id": story_id, "theme": theme, "age": age},
            session_id=story_id,
        )
        config = {"configurable": {...}, "callbacks": callbacks}
    """
    client = _get_client()
    if client is None:
        return []

    try:
        from langfuse.callback import CallbackHandler
        handler = CallbackHandler(
            public_key=settings.LANGFUSE_PUBLIC_KEY,
            secret_key=settings.LANGFUSE_SECRET_KEY,
            host=settings.LANGFUSE_HOST,
            trace_name=name,
            metadata=metadata or {},
            tags=tags or [],
            session_id=session_id,
        )
        return [handler]
    except Exception as e:
        logger.error(f"[Tracing] Failed to create callback handler: {e}")
        return []


def flush():
    """
    Flushes any pending Langfuse events to the server.
    Call this at app shutdown to avoid losing the last few traces.
    """
    client = _get_client()
    if client:
        try:
            client.flush()
        except Exception as e:
            logger.error(f"[Tracing] Flush failed: {e}")
