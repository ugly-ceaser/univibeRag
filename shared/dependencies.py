from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import Depends

from core.database import get_db
from core.storage import LocalStorage
from core.rate_limiter import InMemoryRateLimiter
from core.cache import InMemoryCache

from shared.interfaces import (
    FileStorage, RateLimiter, CacheClient, Embedder, ChatModel,
    ChunkingStrategy, OCREngine, AssetRepository, VectorStore,
    MessageRepository, GapRepository, StudentProfileRepository
)
from ingestion.chunkers import HeaderAwareChunker
from ingestion.parsers import PyTesseractOCREngine
from ingestion.repository import SQLAlchemyAssetRepository, SQLAlchemyVectorStore
from tutor.repository import SQLAlchemyMessageRepository, SQLAlchemyGapRepository, SQLAlchemyStudentProfileRepository
from tutor.llm import MockEmbedder, GeminiChatModel, MockChatModel
from tutor.service import TutorService

# Global singletons for stateful/in-memory infrastructure
_storage_instance = LocalStorage()
_rate_limiter_instance = InMemoryRateLimiter()
_cache_instance = InMemoryCache()
_embedder_instance = MockEmbedder()
_chunker_instance = HeaderAwareChunker()
_ocr_instance = PyTesseractOCREngine()
_chat_model_instance = GeminiChatModel()

def get_storage() -> FileStorage:
    return _storage_instance

def get_rate_limiter() -> RateLimiter:
    return _rate_limiter_instance

def get_cache() -> CacheClient:
    return _cache_instance

def get_file_storage() -> FileStorage:
    return _storage_instance

def get_embedder() -> Embedder:
    return _embedder_instance

def get_chunker() -> ChunkingStrategy:
    return _chunker_instance

def get_ocr_engine() -> OCREngine:
    return _ocr_instance

def get_chat_model() -> ChatModel:
    return _chat_model_instance

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

