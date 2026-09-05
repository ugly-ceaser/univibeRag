from typing import Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete
from sqlalchemy.orm import selectinload
from shared.interfaces import AssetRepository, VectorStore, ChunkWithVector
from db.models import MaterialAsset, Document, Chunk, Message

class SQLAlchemyAssetRepository(AssetRepository):
    def __init__(self, session: AsyncSession):
        self.session = session

    async def find_by_hash(self, content_hash: str) -> MaterialAsset | None:
        stmt = select(MaterialAsset).where(MaterialAsset.content_hash == content_hash)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def create_asset(self, content_hash: str, storage_path: str) -> MaterialAsset:
        asset = MaterialAsset(
            content_hash=content_hash,
            storage_path=storage_path,
            status="pending"
        )
        self.session.add(asset)
        await self.session.flush()
        return asset

    async def update_status(self, asset_id: str, status: str, error_message: str | None = None) -> None:
        stmt = (
            update(MaterialAsset)
            .where(MaterialAsset.id == asset_id)
            .values(status=status, error_message=error_message)
        )
        await self.session.execute(stmt)
        await self.session.commit()

    async def link_document(self, course_id: str, student_id: str, asset_id: str, filename: str, mime_type: str) -> Document:
        doc = Document(
            course_id=course_id,
            student_id=student_id,
            material_asset_id=asset_id,
            filename=filename,
            mime_type=mime_type
        )
        self.session.add(doc)
        await self.session.flush()
        await self.session.commit()
        return doc

    async def get_documents_by_course(self, course_id: str, student_id: str) -> list[dict[str, Any]]:
        stmt = (
            select(Document, MaterialAsset)
            .join(MaterialAsset, Document.material_asset_id == MaterialAsset.id)
            .where(Document.course_id == course_id)
        )
        result = await self.session.execute(stmt)
        items = []
        for doc, asset in result.all():
            items.append({
                "id": doc.id,
                "asset_id": asset.id,
                "course_id": doc.course_id,
                "student_id": doc.student_id,
                "filename": doc.filename,
                "mime_type": doc.mime_type,
                "status": asset.status,
                "error_message": asset.error_message,
                "created_at": doc.created_at.isoformat() if doc.created_at else None
            })
        return items


class SQLAlchemyVectorStore(VectorStore):
    def __init__(self, session: AsyncSession):
        self.session = session

    async def upsert_chunks(self, asset_id: str, chunks_with_vectors: list[ChunkWithVector]) -> None:
        # Delete old chunks for this asset if re-indexing
        del_stmt = delete(Chunk).where(Chunk.material_asset_id == asset_id)
        await self.session.execute(del_stmt)

        chunk_records = []
        for cwv in chunks_with_vectors:
            chunk_records.append(
                Chunk(
                    material_asset_id=asset_id,
                    chunk_index=cwv.chunk_index,
                    content=cwv.text,
                    embedding=cwv.embedding,
                    extra_metadata=cwv.metadata
                )
            )
        self.session.add_all(chunk_records)
        await self.session.commit()

    async def search_chunks(self, query_vector: list[float], course_id: str, top_k: int = 6) -> list[dict[str, Any]]:
        # Vector similarity search using pgvector l2_distance or cosine_distance
        # Filter chunks by documents linked to course_id
        stmt = (
            select(Chunk, Document.id.label("document_id"), Document.filename)
            .join(MaterialAsset, Chunk.material_asset_id == MaterialAsset.id)
            .join(Document, Document.material_asset_id == MaterialAsset.id)
            .where(Document.course_id == course_id)
            .where(MaterialAsset.status == "ready")
            .order_by(Chunk.embedding.l2_distance(query_vector))
            .limit(top_k)
        )
        result = await self.session.execute(stmt)
        hits = []
        for row in result.all():
            chunk = row[0]
            hits.append({
                "chunk_id": chunk.id,
                "content": chunk.content,
                "chunk_index": chunk.chunk_index,
                "document_id": row.document_id,
                "filename": row.filename,
                "metadata": chunk.extra_metadata or {}
            })
        return hits

    async def search_messages(self, query_vector: list[float], course_id: str, student_id: str, top_k: int = 3) -> list[dict[str, Any]]:
        stmt = (
            select(Message)
            .where(Message.course_id == course_id)
            .where(Message.student_id == student_id)
            .where(Message.embedding.is_not(None))
            .order_by(Message.embedding.l2_distance(query_vector))
            .limit(top_k)
        )
        result = await self.session.execute(stmt)
        messages = result.scalars().all()
        return [
            {
                "id": m.id,
                "role": m.role,
                "content": m.content,
                "created_at": m.created_at.isoformat() if m.created_at else None
            }
            for m in messages
        ]
