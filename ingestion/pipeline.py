import logging
from typing import Any
from shared.interfaces import AssetRepository, VectorStore, FileStorage, Embedder, ChunkingStrategy, OCREngine
from ingestion.core import hash_bytes, chunk_document, attach_vectors
from ingestion.parsers import get_parser_for_filename

logger = logging.getLogger(__name__)

async def process_asset_background(
    asset_id: str,
    file_bytes: bytes,
    filename: str,
    mime_type: str,
    asset_repo: AssetRepository,
    vector_store: VectorStore,
    chunker: ChunkingStrategy,
    embedder: Embedder,
    ocr_engine: OCREngine | None = None
) -> None:
    """Background execution step for document parsing, chunking, and embedding."""
    try:
        # Step 1: Parsing
        await asset_repo.update_status(asset_id, "parsing")
        parser = get_parser_for_filename(filename, mime_type)
        raw_doc = parser.parse(file_bytes, mime_type, ocr_engine=ocr_engine)

        if not raw_doc.text_content.strip():
            await asset_repo.update_status(asset_id, "failed", error_message="Extracted text content was empty.")
            return

        # Step 2: Embedding & Chunking
        await asset_repo.update_status(asset_id, "embedding")
        chunks = chunk_document(raw_doc, chunker)
        chunks_with_vectors = attach_vectors(chunks, embedder)

        # Step 3: Vector Storage
        await vector_store.upsert_chunks(asset_id, chunks_with_vectors)

        # Step 4: Ready
        await asset_repo.update_status(asset_id, "ready")
        logger.info(f"Successfully processed asset {asset_id} with {len(chunks_with_vectors)} chunks.")

    except Exception as e:
        logger.exception(f"Failed processing asset {asset_id}: {str(e)}")
        await asset_repo.update_status(asset_id, "failed", error_message=str(e))


async def ingest_document(
    file_bytes: bytes,
    filename: str,
    mime_type: str,
    course_id: str,
    student_id: str,
    asset_repo: AssetRepository,
    vector_store: VectorStore,
    file_storage: FileStorage,
    embedder: Embedder,
    chunker: ChunkingStrategy,
    ocr_engine: OCREngine | None = None,
    run_background: bool = True
) -> str:
    """
    Ingests a document for a course and student.
    Handles duplicate hashing, file storage, document linking, and background or sync processing.
    Returns asset_id.
    """
    content_hash = hash_bytes(file_bytes)

    # Check for existing duplicate asset
    existing_asset = await asset_repo.find_by_hash(content_hash)
    if existing_asset is not None:
        # Link document to this course/student without re-processing
        await asset_repo.link_document(course_id, student_id, existing_asset.id, filename, mime_type)
        return existing_asset.id

    # Create new asset and save file
    storage_path = file_storage.save_file(filename, file_bytes)
    asset = await asset_repo.create_asset(content_hash, storage_path)
    await asset_repo.link_document(course_id, student_id, asset.id, filename, mime_type)

    if run_background:
        # Caller handles adding process_asset_background to BackgroundTasks or async task
        pass
    else:
        await process_asset_background(
            asset.id, file_bytes, filename, mime_type, asset_repo, vector_store, chunker, embedder, ocr_engine
        )

    return asset.id
