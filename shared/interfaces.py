from typing import Protocol, Any, runtime_checkable
from dataclasses import dataclass, field
from datetime import datetime


class ProviderUnavailableError(RuntimeError):
    """Raised when an upstream AI provider (embeddings/LLM) cannot serve a request,
    e.g. quota exhausted, auth failure, network error. Mapped to HTTP 503 by the API."""


@dataclass
class RawDocument:
    filename: str
    text_content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    sections: list[dict[str, Any]] = field(default_factory=list)

@dataclass
class ChunkConfig:
    max_tokens: int = 500
    overlap_tokens: int = 50
    header_preserve: bool = True

@dataclass
class ChunkDraft:
    text: str
    chunk_index: int
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class ChunkWithVector:
    chunk_index: int
    text: str
    embedding: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class TutorResponse:
    grounded: bool
    answer: str
    gaps: list[str] = field(default_factory=list)
    cited_chunk_ids: list[str] = field(default_factory=list)

@dataclass
class TutorAnswer:
    grounded_answer: str
    supplementary_answer: str | None = None
    cited_chunk_ids: list[str] = field(default_factory=list)
    gaps_addressed: list[str] = field(default_factory=list)
    confidence: float = 0.0



@runtime_checkable
class Embedder(Protocol):
    def embed(self, text: str) -> list[float]: ...
    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...


@runtime_checkable
class ChatModel(Protocol):
    def complete(self, messages: list[dict[str, Any]], response_schema: type | None = None, max_tokens: int | None = None) -> Any: ...


@runtime_checkable
class OCREngine(Protocol):
    def extract_text(self, file_bytes: bytes, mime_type: str) -> str: ...


@runtime_checkable
class FileStorage(Protocol):
    def save_file(self, filename: str, content: bytes) -> str: ...
    def get_file(self, storage_path: str) -> bytes: ...
    def delete_file(self, storage_path: str) -> bool: ...


@runtime_checkable
class ChunkingStrategy(Protocol):
    def chunk(self, raw_doc: RawDocument, config: ChunkConfig | None = None) -> list[ChunkDraft]: ...


@runtime_checkable
class DocumentParser(Protocol):
    def parse(self, file_bytes: bytes, mime_type: str, ocr_engine: OCREngine | None = None) -> RawDocument: ...


@runtime_checkable
class AssetRepository(Protocol):
    async def find_by_hash(self, content_hash: str) -> Any | None: ...
    async def create_asset(self, content_hash: str, storage_path: str) -> Any: ...
    async def update_status(self, asset_id: str, status: str, error_message: str | None = None) -> None: ...
    async def link_document(self, course_id: str, student_id: str, asset_id: str, filename: str, mime_type: str) -> Any: ...
    async def get_documents_by_course(self, course_id: str, student_id: str) -> list[Any]: ...


@runtime_checkable
class VectorStore(Protocol):
    async def upsert_chunks(self, asset_id: str, chunks_with_vectors: list[ChunkWithVector]) -> None: ...
    async def search_chunks(self, query_vector: list[float], course_id: str, student_id: str | None = None, top_k: int = 6) -> list[dict[str, Any]]: ...
    async def search_messages(self, query_vector: list[float], course_id: str, student_id: str, top_k: int = 3) -> list[dict[str, Any]]: ...


@runtime_checkable
class MessageRepository(Protocol):
    async def save_message(self, course_id: str, student_id: str, role: str, content: str, embedding: list[float] | None = None, grounded: bool = True, sources: list[str] | None = None) -> Any: ...
    async def get_history(self, course_id: str, student_id: str, limit: int = 50) -> list[Any]: ...
    async def clear_history(self, course_id: str, student_id: str) -> int: ...


@runtime_checkable
class GapRepository(Protocol):
    async def get_for_course(self, course_id: str) -> list[Any]: ...
    async def cache_gap(self, course_id: str, topic: str, topic_embedding: list[float], supplementary_answer: str) -> Any: ...


@runtime_checkable
class StudentProfileRepository(Protocol):
    async def get_profile(self, student_id: str) -> dict[str, Any] | None: ...
    async def set_preferred_name(self, student_id: str, name: str) -> dict[str, Any]: ...


@runtime_checkable
class RateLimiter(Protocol):
    async def is_rate_limited(self, key: str, limit: int, window_seconds: int) -> tuple[bool, int]: ...


@runtime_checkable
class CacheClient(Protocol):
    async def get(self, key: str) -> str | None: ...
    async def set(self, key: str, value: str, ttl_seconds: int | None = None) -> None: ...
    async def delete(self, key: str) -> None: ...

