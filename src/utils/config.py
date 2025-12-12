     1|
     2|import os
     3|from functools import lru_cache
     4|from pydantic_settings import BaseSettings
     5|class Settings(BaseSettings):
     6|     # App Settings
     7|    APP_ENV: str = "development"
     8|    LOG_LEVEL: str = "INFO"
     9|    
    10|    # Google Cloud
    11|    GOOGLE_CLOUD_PROJECT: str
    12|    GOOGLE_API_KEY: str
    13|    FIRESTORE_DATABASE: str = "(default)"
    14|    GOOGLE_APPLICATION_CREDENTIALS: str = None # Path to service account JSON
    15|    
    16|    # Cost & CO2 Optimization: AI Models
    17|    # Default to Flash for speed and lower cost/CO2
    18|    GEMINI_MODEL: str = "gemini-1.5-flash" 
    19|    IMAGE_MODEL: str = "imagen-3"
    20|
    21|    # Cost Optimization: Limits
    22|    MAX_RETRIES: int = 3
    23|    RETRY_DELAY_SECONDS: int = 2
    24|
    25|     # Performance & Scaling
    26|    MAX_CONCURRENCY: int = 10
    27|    class Config:
    28|        env_file = ".env"
    29|
    30|@lru_cache()
    31|def get_settings():
    32|    return Settings()
    33|