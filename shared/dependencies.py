from typing import AsyncGenerator, Callable, Awaitable
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import Depends, HTTPException, status

from core.database import get_db
from core.security import get_current_user, AuthenticatedUser
from core.storage import LocalStorage
from core.rate_limiter import InMemoryRateLimiter
from core.cache import InMemoryCache

from shared.interfaces import (
    FileStorage, RateLimiter, CacheClient, Embedder, ChatModel,
    ChunkingStrategy, OCREngine, AssetRepository, VectorStore,
    MessageRepository, GapRepository, StudentProfileRepository
)
from ingestion.chunkers import HeaderAwareChunker
from ingestion.parsers import CompositeOCREngine
from ingestion.repository import SQLAlchemyAssetRepository, SQLAlchemyVectorStore
from tutor.repository import SQLAlchemyMessageRepository, SQLAlchemyGapRepository, SQLAlchemyStudentProfileRepository
import os
from config import settings
from tutor.llm import MockEmbedder, OpenAIEmbedder, GeminiEmbedder, GeminiChatModel, OpenAIChatModel, MockChatModel
from tutor.service import TutorService

# Global singletons for stateful/in-memory infrastructure
_storage_instance = LocalStorage()
_rate_limiter_instance = InMemoryRateLimiter()
_cache_instance = InMemoryCache()
_chunker_instance = HeaderAwareChunker()
_ocr_instance = CompositeOCREngine()

def get_storage() -> FileStorage:
    return _storage_instance

def get_rate_limiter() -> RateLimiter:
    return _rate_limiter_instance

RATE_LIMIT_WINDOW_SECONDS = 60


class RateLimitExceeded(HTTPException):
    """429 that also carries `retry_after` so the error handler can put it in the JSON body.
    The app reads `retryAfter` from the body first, because on web the Retry-After header
    can be hidden by CORS."""

    def __init__(self, detail: str, retry_after: int):
        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=detail,
            headers={"Retry-After": str(retry_after)},
        )
        self.retry_after = retry_after

# Per-student limits, read lazily so .env / settings changes apply without code edits.
# "ask" is one shared bucket for every route that triggers LLM calls
# (/courses/{id}/ask and the mobile /ai/chat/course alias), so a client can't
# double its budget by alternating routes.
_RATE_LIMITS: dict[str, tuple[Callable[[], int], str]] = {
    "ask": (lambda: settings.RATE_LIMIT_ASK_PER_MIN, "questions"),
    "upload": (lambda: settings.RATE_LIMIT_UPLOAD_PER_MIN, "uploads"),
}

def rate_limit(kind: str) -> Callable[..., Awaitable[None]]:
    """Route dependency enforcing the per-student, per-minute limit for `kind`. Raises 429."""
    get_limit, noun = _RATE_LIMITS[kind]

    async def _enforce(
        user: AuthenticatedUser = Depends(get_current_user),
        limiter: RateLimiter = Depends(get_rate_limiter),
    ) -> None:
        limit = get_limit()
        is_limited, _ = await limiter.is_rate_limited(
            f"rate:{kind}:{user.id}", limit=limit, window_seconds=RATE_LIMIT_WINDOW_SECONDS
        )
        if is_limited:
            raise RateLimitExceeded(
                detail=f"Rate limit exceeded: max {limit} {noun} per minute. Please try again shortly.",
                retry_after=RATE_LIMIT_WINDOW_SECONDS,
            )

    return _enforce

def get_cache() -> CacheClient:
    return _cache_instance

def get_file_storage() -> FileStorage:
    return _storage_instance

def get_embedder() -> Embedder:
    openai_key = settings.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
    gemini_key = settings.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY")
    if openai_key:
        return OpenAIEmbedder(api_key=openai_key)
    elif gemini_key:
        return GeminiEmbedder(api_key=gemini_key)
    return MockEmbedder()

def get_chunker() -> ChunkingStrategy:
    return _chunker_instance

def get_ocr_engine() -> OCREngine:
    return _ocr_instance

def get_chat_model() -> ChatModel:
    openai_key = settings.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
    gemini_key = settings.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY")
    if openai_key:
        return OpenAIChatModel(api_key=openai_key)
    elif gemini_key:
        return GeminiChatModel(api_key=gemini_key)
    return MockChatModel()


def get_asset_repository(session: AsyncSession = Depends(get_db)) -> AssetRepository:
    return SQLAlchemyAssetRepository(session)

def get_vector_store(session: AsyncSession = Depends(get_db)) -> VectorStore:
    return SQLAlchemyVectorStore(session)

def get_message_repository(session: AsyncSession = Depends(get_db)) -> MessageRepository:
    return SQLAlchemyMessageRepository(session)

def get_gap_repository(session: AsyncSession = Depends(get_db)) -> GapRepository:
    return SQLAlchemyGapRepository(session)

def get_student_profile_repository(session: AsyncSession = Depends(get_db)) -> StudentProfileRepository:
    return SQLAlchemyStudentProfileRepository(session)

def get_tutor_service(
    vector_store: VectorStore = Depends(get_vector_store),
    message_repo: MessageRepository = Depends(get_message_repository),
    gap_repo: GapRepository = Depends(get_gap_repository),
    embedder: Embedder = Depends(get_embedder),
    chat_model: ChatModel = Depends(get_chat_model),
    profile_repo: StudentProfileRepository = Depends(get_student_profile_repository)
) -> TutorService:
    return TutorService(
        vector_store=vector_store,
        message_repo=message_repo,
        gap_repo=gap_repo,
        embedder=embedder,
        chat_model=chat_model,
        profile_repo=profile_repo
    )

