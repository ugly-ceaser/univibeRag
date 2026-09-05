import pytest
from unittest.mock import AsyncMock, MagicMock
from shared.interfaces import (
    AssetRepository, VectorStore, FileStorage, Embedder, OCREngine,
    RawDocument, ChunkDraft, ChunkWithVector
)
from ingestion.core import hash_bytes, chunk_document, attach_vectors
from ingestion.chunkers import HeaderAwareChunker
from ingestion.pipeline import ingest_document, process_asset_background

def test_hash_bytes_deterministic():
    content1 = b"Sample document bytes"
    content2 = b"Sample document bytes"
    content3 = b"Different document bytes"
    
    hash1 = hash_bytes(content1)
    hash2 = hash_bytes(content2)
    hash3 = hash_bytes(content3)

    assert hash1 == hash2
    assert hash1 != hash3
    assert len(hash1) == 64

def test_attach_vectors():
    mock_embedder = MagicMock(spec=Embedder)
    mock_embedder.embed_batch.return_value = [[0.1, 0.2], [0.3, 0.4]]
    
    chunks = [
        ChunkDraft(text="Chunk 1", chunk_index=0),
        ChunkDraft(text="Chunk 2", chunk_index=1)
    ]
    
    cwv = attach_vectors(chunks, mock_embedder)
    assert len(cwv) == 2
    assert cwv[0].embedding == [0.1, 0.2]
    assert cwv[1].embedding == [0.3, 0.4]

@pytest.mark.asyncio
async def test_ingest_document_duplicate_handling():
    mock_asset_repo = AsyncMock(spec=AssetRepository)
    mock_vector_store = AsyncMock(spec=VectorStore)
    mock_file_storage = MagicMock(spec=FileStorage)
    mock_embedder = MagicMock(spec=Embedder)
    mock_chunker = HeaderAwareChunker()
    
    # Simulate existing asset found
    mock_existing_asset = MagicMock()
    mock_existing_asset.id = "asset-uuid-123"
    mock_asset_repo.find_by_hash.return_value = mock_existing_asset
    
    file_bytes = b"Duplicate file content"
    asset_id = await ingest_document(
        file_bytes=file_bytes,
        filename="lecture1.txt",
        mime_type="text/plain",
        course_id="course-101",
        student_id="student-456",
        asset_repo=mock_asset_repo,
        vector_store=mock_vector_store,
        file_storage=mock_file_storage,
        embedder=mock_embedder,
        chunker=mock_chunker,
        run_background=True
    )
    
    assert asset_id == "asset-uuid-123"
    mock_asset_repo.link_document.assert_called_once_with(
        "course-101", "student-456", "asset-uuid-123", "lecture1.txt", "text/plain"
    )
    # File storage save should NOT be called for duplicates
    mock_file_storage.save_file.assert_not_called()

@pytest.mark.asyncio
async def test_process_asset_background_flow():
    mock_asset_repo = AsyncMock(spec=AssetRepository)
    mock_vector_store = AsyncMock(spec=VectorStore)
    mock_embedder = MagicMock(spec=Embedder)
    mock_embedder.embed_batch.return_value = [[0.1] * 1536]
    mock_chunker = HeaderAwareChunker()

    file_bytes = b"# Chapter 1\nThis is a sample document for testing the background pipeline."
    
    await process_asset_background(
        asset_id="asset-999",
        file_bytes=file_bytes,
        filename="notes.md",
        mime_type="text/markdown",
        asset_repo=mock_asset_repo,
        vector_store=mock_vector_store,
        chunker=mock_chunker,
        embedder=mock_embedder
    )

    # Check state transitions
    status_calls = [call[0] for call in mock_asset_repo.update_status.call_args_list]
    assert ("asset-999", "parsing") in status_calls
    assert ("asset-999", "embedding") in status_calls
    assert ("asset-999", "ready") in status_calls
    mock_vector_store.upsert_chunks.assert_called_once()
