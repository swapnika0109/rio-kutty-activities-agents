import os
import uvicorn
from fastapi import FastAPI

from src.api import stories, media, activities, health
from src.utils.logger import setup_logger
from src.utils.tracing import flush as flush_traces

logger = setup_logger(__name__)

app = FastAPI(title="Rio Kutty Story Management")


@app.on_event("startup")
async def startup_event():
    logger.info("Application startup complete.")


@app.on_event("shutdown")
async def shutdown_event():
    flush_traces()   # ensure last Langfuse events reach the server before shutdown
    logger.info("Application shutdown complete.")


app.include_router(stories.router)
app.include_router(media.router)
app.include_router(activities.router)
app.include_router(health.router)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
