import os
from functools import lru_cache
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
     # App Settings
    APP_ENV: str = "development"
    LOG_LEVEL: str = "INFO"
    
    # Google Cloud
    GOOGLE_CLOUD_PROJECT: str
    GOOGLE_API_KEY: str
    FIRESTORE_DATABASE: str = "(default)"
    GOOGLE_APPLICATION_CREDENTIALS: str = None # Path to service account JSON
    
    # Cost & CO2 Optimization: AI Models
    # Default to Flash for speed and lower cost/CO2
    GEMINI_MODEL: str = "gemini-1.5-flash" 
    IMAGE_MODEL: str = "imagen-3"

    # Cost Optimization: Limits
    MAX_RETRIES: int = 3
    RETRY_DELAY_SECONDS: int = 2

     # Performance & Scaling
    MAX_CONCURRENCY: int = 10
    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()