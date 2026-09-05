import pytest
from db.models import MaterialAsset, Document, Chunk, Message, CourseGap
from core.database import Base

def test_db_models_structure():
    """Verify that all ORM models are registered on Base metadata with correct tablenames."""
    tables = list(Base.metadata.tables.keys())
    assert "material_assets" in tables
    assert "documents" in tables
    assert "chunks" in tables
    assert "messages" in tables
    assert "course_gaps" in tables

def test_model_instantiation():
    """Verify ORM model creation with default values."""
    asset = MaterialAsset(content_hash="abc123hash", storage_path="/path/to/file.pdf", status="pending")
    assert asset.content_hash == "abc123hash"
    assert asset.status == "pending"

    doc = Document(course_id="CSC101", student_id="user123", material_asset_id=asset.id, filename="lecture1.pdf")
    assert doc.course_id == "CSC101"
    assert doc.filename == "lecture1.pdf"

    chunk = Chunk(material_asset_id=asset.id, content="Sample content chunk", chunk_index=0)
    assert chunk.content == "Sample content chunk"

    msg = Message(course_id="CSC101", student_id="user123", role="user", content="What is SN1?", grounded=True)
    assert msg.role == "user"
    assert msg.grounded is True

    gap = CourseGap(course_id="CSC101", topic="Haber process", topic_embedding=[0.1]*1536, supplementary_answer="An industrial process...")
    assert gap.topic == "Haber process"
    assert len(gap.topic_embedding) == 1536
