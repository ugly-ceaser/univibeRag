from typing import Annotated
from fastapi import APIRouter, Depends, UploadFile, File, BackgroundTasks, HTTPException, status
from pydantic import BaseModel, Field

from config import settings
from core.security import get_current_user, AuthenticatedUser
from shared.dependencies import (
    get_asset_repository,
    get_vector_store,
    get_file_storage,
    get_embedder,
    get_chunker,
    get_ocr_engine,
    rate_limit
)
from shared.interfaces import AssetRepository, VectorStore, FileStorage, Embedder, ChunkingStrategy, OCREngine
from ingestion.pipeline import process_asset_background
from ingestion.core import hash_bytes

router = APIRouter(prefix="/courses/{course_id}/materials", tags=["materials"])

class MaterialResponse(BaseModel):
    asset_id: str = Field(..., description="Unique ID of the material asset")
    status: str = Field(..., description="Ingestion status (pending, parsing, embedding, ready, failed)")
    filename: str = Field(..., description="Original filename")
    message: str = Field(..., description="Status summary message")

@router.post(
    "/upload",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=MaterialResponse,
    summary="Upload Course Material",
    description="Uploads a PDF, DOCX, TXT, or Image file for background parsing, header-aware chunking, vector embedding, and indexing.",
    dependencies=[Depends(rate_limit("upload"))]
)
@router.post(
    "",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=MaterialResponse,
    summary="Upload Course Material Alias",
    include_in_schema=False,
    dependencies=[Depends(rate_limit("upload"))]
)
async def upload_course_material(
    course_id: str,
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    user: AuthenticatedUser = Depends(get_current_user),
    asset_repo: AssetRepository = Depends(get_asset_repository),
    vector_store: VectorStore = Depends(get_vector_store),
    file_storage: FileStorage = Depends(get_file_storage),
    embedder: Embedder = Depends(get_embedder),
    chunker: ChunkingStrategy = Depends(get_chunker),
    ocr_engine: OCREngine = Depends(get_ocr_engine)
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is missing")

    max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024
    too_large = HTTPException(
        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
        detail=f"File exceeds the {settings.MAX_UPLOAD_MB} MB upload limit."
    )
    # Reject on declared size first, then read at most max_bytes + 1 so an
    # undeclared/lying size can't make us buffer an arbitrarily large file.
    if file.size is not None and file.size > max_bytes:
        raise too_large
    file_bytes = await file.read(max_bytes + 1)
    if len(file_bytes) > max_bytes:
        raise too_large
    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    content_hash = hash_bytes(file_bytes)
    existing_asset = await asset_repo.find_by_hash(content_hash)

    if existing_asset is not None:
        await asset_repo.link_document(course_id, user.id, existing_asset.id, file.filename, file.content_type or "")
        return MaterialResponse(
            asset_id=existing_asset.id,
            status=existing_asset.status,
            filename=file.filename,
            message="File already exists and has been linked to this course."
        )

    # Save file and create new asset
    storage_path = file_storage.save_file(file.filename, file_bytes)
    asset = await asset_repo.create_asset(content_hash, storage_path)
    await asset_repo.link_document(course_id, user.id, asset.id, file.filename, file.content_type or "")

    # Enqueue background task
    background_tasks.add_task(
        process_asset_background,
        asset.id,
        file_bytes,
        file.filename,
        file.content_type or "",
        asset_repo,
        vector_store,
        chunker,
        embedder,
        ocr_engine
    )

    return MaterialResponse(
        asset_id=asset.id,
        status="pending",
        filename=file.filename,
        message="Document uploaded successfully and enqueued for ingestion."
    )


@router.get(
    "",
    summary="List Course Materials",
    description="Retrieves all material documents uploaded and indexed for a given course."
)
async def list_course_materials(
    course_id: str,
    user: AuthenticatedUser = Depends(get_current_user),
    asset_repo: AssetRepository = Depends(get_asset_repository)
):
    docs = await asset_repo.get_documents_by_course(course_id, user.id)
    return {"course_id": course_id, "materials": docs, "data": docs}
