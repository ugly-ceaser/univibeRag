from typing import Annotated
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from core.security import get_current_user, AuthenticatedUser
from shared.dependencies import (
    get_tutor_service,
    get_message_repository,
    get_gap_repository,
    get_rate_limiter
)
from shared.interfaces import MessageRepository, GapRepository, RateLimiter, TutorAnswer
from tutor.service import TutorService

router = APIRouter(prefix="/courses/{course_id}", tags=["tutor"])

class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=1000, description="Question asked by the student")

@router.post(
    "/ask",
    response_model=TutorAnswer,
    summary="Ask AI Tutor",
    description="Submits a student question to the RAG pipeline. Performs vector search on course materials, enforces strict grounding, extracts citations, and identifies missing topic gaps."
)
async def ask_course_tutor(
    course_id: str,
    body: QuestionRequest,
    user: AuthenticatedUser = Depends(get_current_user),
    tutor_service: TutorService = Depends(get_tutor_service),
    rate_limiter: RateLimiter = Depends(get_rate_limiter)
):
    # Enforce rate limiting: max 20 questions per minute per student
    rate_key = f"rate:ask:{user.id}"
    is_limited, retry_after = await rate_limiter.is_rate_limited(rate_key, limit=20, window_seconds=60)
    if is_limited:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Please try again in {retry_after} seconds."
        )

    answer = await tutor_service.ask_tutor(
        course_id=course_id,
        student_id=user.id,
        question=body.question
    )
    return answer


@router.get(
    "/history",
    summary="Get Chat History",
    description="Retrieves previous user questions and assistant answers for the given course and authenticated student."
)
async def get_course_chat_history(
    course_id: str,
    limit: int = 50,
    user: AuthenticatedUser = Depends(get_current_user),
    message_repo: MessageRepository = Depends(get_message_repository)
):
    history = await message_repo.get_history(course_id=course_id, student_id=user.id, limit=limit)
    return {
        "course_id": course_id,
        "student_id": user.id,
        "count": len(history),
        "messages": history
    }


@router.delete(
    "/history",
    summary="Clear Chat History",
    description="Clears all saved conversation messages for the current course and student session."
)
async def clear_course_chat_history(
    course_id: str,
    user: AuthenticatedUser = Depends(get_current_user),
    message_repo: MessageRepository = Depends(get_message_repository)
):
    deleted_count = await message_repo.clear_history(course_id=course_id, student_id=user.id)
    return {
        "course_id": course_id,
        "student_id": user.id,
        "deleted_count": deleted_count,
        "message": "Chat history cleared successfully."
    }


@router.get(
    "/gaps",
    summary="Get Missing Course Material Gaps",
    description="Lists all identified course material topic gaps flagged when students asked questions not answered by uploaded materials."
)
async def get_course_material_gaps(
    course_id: str,
    user: AuthenticatedUser = Depends(get_current_user),
    gap_repo: GapRepository = Depends(get_gap_repository)
):
    gaps = await gap_repo.get_for_course(course_id)
    return {
        "course_id": course_id,
        "gaps": gaps
    }


@router.get(
    "/chats/session",
    summary="Get Mobile Chat Session History",
    description="Mobile app alias route for retrieving chat session details and message history."
)
async def get_chat_session(
    course_id: str,
    user: AuthenticatedUser = Depends(get_current_user),
    message_repo: MessageRepository = Depends(get_message_repository)
):
    history = await message_repo.get_history(course_id=course_id, student_id=user.id, limit=50)
    return {
        "session_id": f"{course_id}:{user.id}",
        "course_id": course_id,
        "student_id": user.id,
        "active": True,
        "messages": history
    }
