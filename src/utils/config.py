import os
from functools import lru_cache
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
     # App Settings
    APP_ENV: str = "development"
    LOG_LEVEL: str = "INFO"
    
    # Google Cloud
    GOOGLE_CLOUD_PROJECT : str = "riokutty"
    GOOGLE_CLOUD_BUCKET : str = "kutty_bucket"
    GOOGLE_API_KEY: str
    FIRESTORE_DATABASE: str = "(default)"
    GOOGLE_APPLICATION_CREDENTIALS: str | None = None # Path to service account JSON
    
    # Cost & CO2 Optimization: AI Models
    # Default to Flash for speed and lower cost/CO2
    GEMINI_MODEL: str = "gemini-2.0-flash-lite" 
    GEMINI_FALLBACK_MODEL: str = "gemini-2.0-flash"
    MULTIMODAL_MODEL: str = "gemini-2.5-flash-image"
    IMAGE_MODEL: str = "imagen-3"

    # Cost Optimization: Limits
    MAX_RETRIES: int = 3
    RETRY_DELAY_SECONDS: int = 2
    CIRCUIT_BREAKER_FAILURE_THRESHOLD: int = 5
    CIRCUIT_BREAKER_RECOVERY_TIMEOUT_SECONDS: int = 60
    RATE_LIMIT_TOKENS_PER_SECOND: float = 3.0
    RATE_LIMIT_BURST_CAPACITY: int = 6

    # Prompt versioning (per activity agent)
    MCQ_PROMPT_VERSION: str = "latest"
    ART_PROMPT_VERSION: str = "latest"
    MORAL_PROMPT_VERSION: str = "latest"
    SCIENCE_PROMPT_VERSION: str = "latest"

     # Performance & Scaling
    MAX_CONCURRENCY: int = 10
    HF_TOKEN: str
    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()