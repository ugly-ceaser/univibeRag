import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Text, Boolean, Integer, DateTime, ForeignKey, Index, JSON
from pgvector.sqlalchemy import Vector
from core.database import Base

def generate_uuid() -> str:
    return str(uuid.uuid4())

def utc_now() -> datetime:
    return datetime.now(timezone.utc)

class MaterialAsset(Base):
    __tablename__ = 'material_assets'

    id = Column(String(36), primary_key=True, default=generate_uuid)
    content_hash = Column(String(64), unique=True, nullable=False, index=True)
    storage_path = Column(String(512), nullable=False)
    status = Column(String(32), nullable=False, default='pending', index=True)  # pending, processing, ready, failed
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)


class Document(Base):
    __tablename__ = 'documents'

    id = Column(String(36), primary_key=True, default=generate_uuid)
    course_id = Column(String(128), nullable=False, index=True)
    student_id = Column(String(128), nullable=False, index=True)
    material_asset_id = Column(String(36), ForeignKey('material_assets.id', ondelete='CASCADE'), nullable=False, index=True)
    filename = Column(String(256), nullable=False)
    mime_type = Column(String(128), nullable=False, default='application/pdf')
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)


class Chunk(Base):
    __tablename__ = 'chunks'

    id = Column(String(36), primary_key=True, default=generate_uuid)
    material_asset_id = Column(String(36), ForeignKey('material_assets.id', ondelete='CASCADE'), nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False, default=0)
    content = Column(Text, nullable=False)
    embedding = Column(Vector(1536), nullable=True)
    extra_metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)


class Message(Base):
    __tablename__ = 'messages'

    id = Column(String(36), primary_key=True, default=generate_uuid)
    course_id = Column(String(128), nullable=False, index=True)
    student_id = Column(String(128), nullable=False, index=True)
    role = Column(String(32), nullable=False)  # user, assistant, system
    content = Column(Text, nullable=False)
    embedding = Column(Vector(1536), nullable=True)
    grounded = Column(Boolean, nullable=False, default=True)
    sources = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)


class CourseGap(Base):
    __tablename__ = 'course_gaps'

    id = Column(String(36), primary_key=True, default=generate_uuid)
    course_id = Column(String(128), nullable=False, index=True)
    topic = Column(Text, nullable=False)
    topic_embedding = Column(Vector(1536), nullable=False)
    supplementary_answer = Column(Text, nullable=False)
    hit_count = Column(Integer, nullable=False, default=1)
    first_seen_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    last_seen_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)


class StudentProfile(Base):
    __tablename__ = 'student_profiles'

    student_id = Column(String(128), primary_key=True)
    preferred_name = Column(String(128), nullable=True)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False)


# Index definitions for performance
Index('idx_documents_course_student', Document.course_id, Document.student_id)
Index('idx_messages_course_student', Message.course_id, Message.student_id)
Index('idx_gaps_course', CourseGap.course_id)

