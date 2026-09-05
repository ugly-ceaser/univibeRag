from pydantic_settings import BaseSettings, SettingsConfigDict
import os

class Settings(BaseSettings):
    ENVIRONMENT: str = "development"
    CORS_ORIGINS: str = "*"
    DATABASE_URL: str = "postgresql+asyncpg://postgres:12345@localhost:5433/uniVibes"
    REDIS_URL: str = "redis://localhost:6379"
    JWT_SECRET: str = "martiedcknrdcrdkcndklncfdvc"
    OPENAI_API_KEY: str = ""
    GEMINI_API_KEY: str = ""
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    LLM_MODEL: str = "gpt-4o-mini"
    STORAGE_DIR: str = "./media_storage"
    
    # RAG settings
    TOP_K_CHUNKS: int = 6
    TOP_K_MEMORY: int = 3
    GAP_SIMILARITY_THRESHOLD: float = 0.85
    
    # Rate Limiting settings
    RATE_LIMIT_UPLOAD_PER_MIN: int = 5
    RATE_LIMIT_ASK_PER_MIN: int = 15

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
