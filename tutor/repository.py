from typing import Any
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete, update
from shared.interfaces import MessageRepository, GapRepository, StudentProfileRepository
from db.models import Message, CourseGap, StudentProfile


class SQLAlchemyMessageRepository(MessageRepository):
    def __init__(self, session: AsyncSession):
        self.session = session

    async def save_message(
        self,
        course_id: str,
        student_id: str,
        role: str,
        content: str,
        embedding: list[float] | None = None,
        grounded: bool = True,
        sources: list[str] | None = None
    ) -> Message:
        msg = Message(
            course_id=course_id,
            student_id=student_id,
            role=role,
            content=content,
            embedding=embedding,
            grounded=grounded,
            sources=sources
        )
        self.session.add(msg)
        await self.session.flush()
        await self.session.commit()
        return msg

    async def get_history(self, course_id: str, student_id: str, limit: int = 50) -> list[dict[str, Any]]:
        stmt = (
            select(Message)
            .where(Message.course_id == course_id)
            .where(Message.student_id == student_id)
            .order_by(Message.created_at.asc())
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        messages = result.scalars().all()
        return [
            {
                "id": m.id,
                "role": m.role,
                "content": m.content,
                "grounded": m.grounded,
                "sources": m.sources or [],
                "created_at": m.created_at.isoformat() if m.created_at else None
            }
            for m in messages
        ]

    async def clear_history(self, course_id: str, student_id: str) -> int:
        stmt = (
            delete(Message)
            .where(Message.course_id == course_id)
            .where(Message.student_id == student_id)
        )
        result = await self.session.execute(stmt)
        await self.session.commit()
        return result.rowcount


class SQLAlchemyGapRepository(GapRepository):
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_for_course(self, course_id: str) -> list[dict[str, Any]]:
        stmt = (
            select(CourseGap)
            .where(CourseGap.course_id == course_id)
            .order_by(CourseGap.hit_count.desc())
        )
        result = await self.session.execute(stmt)
        gaps = result.scalars().all()
        return [
            {
                "id": g.id,
                "course_id": g.course_id,
                "topic": g.topic,
                "supplementary_answer": g.supplementary_answer,
                "hit_count": g.hit_count,
                "last_seen_at": g.last_seen_at.isoformat() if g.last_seen_at else None
            }
            for g in gaps
        ]

    async def cache_gap(
        self,
        course_id: str,
        topic: str,
        topic_embedding: list[float],
        supplementary_answer: str
    ) -> CourseGap:
        # Check if topic gap already exists for this course
        stmt = (
            select(CourseGap)
            .where(CourseGap.course_id == course_id)
            .where(CourseGap.topic == topic)
        )
        result = await self.session.execute(stmt)
        existing = result.scalar_one_or_none()

        now = datetime.now(timezone.utc)
        if existing:
            existing.hit_count += 1
            existing.last_seen_at = now
            if supplementary_answer and not existing.supplementary_answer:
                existing.supplementary_answer = supplementary_answer
            await self.session.commit()
            return existing
        else:
            gap = CourseGap(
                course_id=course_id,
                topic=topic,
                topic_embedding=topic_embedding,
                supplementary_answer=supplementary_answer,
                hit_count=1,
                first_seen_at=now,
                last_seen_at=now
            )
            self.session.add(gap)
            await self.session.flush()
            await self.session.commit()
            return gap


class SQLAlchemyStudentProfileRepository(StudentProfileRepository):
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_profile(self, student_id: str) -> dict[str, Any] | None:
        stmt = select(StudentProfile).where(StudentProfile.student_id == student_id)
        result = await self.session.execute(stmt)
        profile = result.scalar_one_or_none()
        if not profile:
            return None
        return {
            "student_id": profile.student_id,
            "preferred_name": profile.preferred_name
        }

    async def set_preferred_name(self, student_id: str, name: str) -> dict[str, Any]:
        stmt = select(StudentProfile).where(StudentProfile.student_id == student_id)
        result = await self.session.execute(stmt)
        profile = result.scalar_one_or_none()
        if profile:
            profile.preferred_name = name
        else:
            profile = StudentProfile(student_id=student_id, preferred_name=name)
            self.session.add(profile)
        await self.session.flush()
        await self.session.commit()
        return {
            "student_id": profile.student_id,
            "preferred_name": profile.preferred_name
        }

