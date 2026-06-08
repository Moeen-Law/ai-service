"""
Application Settings

Centralized configuration management using pydantic-settings.
All settings are loaded from environment variables or .env file.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    Follows 12-factor app principles for configuration management.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # Service Configuration
    SERVICE_NAME: str = "ai-service"
    VERSION: str = "0.1.0"
    ENVIRONMENT: Literal["development", "staging", "production"] = "development"

    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False

    # Logging
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    LOG_FORMAT: Literal["json", "text"] = "json"
    LOG_FILE_PATH: str = str(Path("logs") / "ai-service.log")
    LOG_OVERWRITE_ON_START: bool = True

    # API Configuration
    API_V1_PREFIX: str = "/v1"
    ENABLE_DOCS: bool = True

    # Request Limits
    MAX_REQUEST_SIZE_MB: int = 10
    REQUEST_TIMEOUT_SECONDS: int = 30

    # --- Files Service (uploaded file retrieval) ---
    FILES_SERVICE_BASE_URL: str = ""
    FILES_SERVICE_DOWNLOAD_PATH_TEMPLATE: str = "/files/api/v1/files/{file_id}"
    FILES_SERVICE_UPLOAD_URL_PATH: str = "/files/api/v1/files/upload-url"
    FILES_SERVICE_AUTH_TOKEN: str = ""
    FILES_SERVICE_TIMEOUT_SECONDS: int = 20
    FILES_SERVICE_VERIFY_TLS: bool = True
    FILES_SERVICE_UPLOAD_BUCKET: str = "AI_DOCUMENTS"

    # --- Uploaded file processing limits ---
    FILES_MAX_COUNT: int = 10
    FILES_MAX_SIZE_BYTES: int = 8 * 1024 * 1024
    FILES_MAX_EXTRACTED_CHARS_PER_FILE: int = 15000
    FILES_MAX_EXTRACTED_TOTAL_CHARS: int = 50000

    # --- AI / LLM Configuration ---
    GEMINI_API_KEYS: str = ""
    GEMINI_API_KEY: str = ""
    LLM_MODEL: str
    INTENT_MODEL: str = "llama-3.1-8b-instant"
    GROQ_API_KEY: str = ""
    LLM_TEMPERATURE: float = 0.0
    LLM_REQUEST_TIMEOUT_SECONDS: int = 30
    LLM_MAX_RETRIES: int = 2
    LLM_RETRY_BASE_DELAY_SECONDS: float = 0.4

    # --- Qdrant Vector Database ---
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_PORT: int = 443
    QDRANT_API_KEY: str = ""
    QDRANT_COLLECTION_NAME: str = "egyptian_law_scraped"

    # --- Embedding Configuration ---
    EMBEDDING_MODEL: str = "intfloat/multilingual-e5-base"
    EMBEDDING_DIMENSION: int = 768
    EMBEDDING_DEVICE: str = "cpu"

    # --- Hybrid RAG Configuration ---
    VECTOR_WEIGHT: float = 0.8
    BM25_WEIGHT: float = 0.2
    RETRIEVAL_K: int = 4
    MAX_FRONTEND_SOURCES: int = 7
    # --- Redis Configuration ---
    REDIS_URL: str = ""

    TAVILY_API_KEY: str = ""
    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.ENVIRONMENT == "development"

    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.ENVIRONMENT == "production"


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Uses LRU cache to ensure settings are only loaded once.
    """
    return Settings()