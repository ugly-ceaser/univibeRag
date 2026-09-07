import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import settings
from core.database import init_db
from core.security import get_current_user, AuthenticatedUser
from shared.dependencies import get_message_repository
from shared.interfaces import MessageRepository
from ingestion.api import router as ingestion_router
from tutor.api import router as tutor_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag_backend")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize DB tables & vector extension
    logger.info("Initializing RAG backend database & pgvector extension...")
    try:
        await init_db()
        logger.info("Database initialized successfully.")
    except Exception as e:
        logger.warning(f"Database initialization warning (DB may be offline or already setup): {str(e)}")
    yield
    # Shutdown
    logger.info("Shutting down RAG backend.")

app = FastAPI(
    title="UniVibe RAG Backend",
    description="Decoupled microservice for course material ingestion, vector search, and AI tutoring.",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS Middleware
origins = settings.CORS_ORIGINS.split(",") if settings.CORS_ORIGINS else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health Check Endpoint
@app.get("/health", tags=["system"])
async def health_check():
    return {
        "status": "healthy",
        "service": "rag_backend",
        "version": "1.0.0",
        "environment": settings.ENVIRONMENT
    }

# Register Feature Routers
app.include_router(ingestion_router)
app.include_router(tutor_router)

from datetime import datetime, timezone
from typing import Any
from pydantic import BaseModel, Field
from shared.dependencies import get_tutor_service
from tutor.service import TutorService

class MobileAIChatRequest(BaseModel):
    message: str
    courseId: str
    context: dict[str, Any] | None = None
    conversationHistory: list[dict[str, Any]] | None = None
    userMode: str | None = "balanced"

# Mobile App Alias Routes
@app.get("/ai/courses/{course_id}/chats/session", tags=["mobile_alias"])
async def mobile_chat_session(
    course_id: str,
    user: AuthenticatedUser = Depends(get_current_user),
    message_repo: MessageRepository = Depends(get_message_repository)
):
    history = await message_repo.get_history(course_id=course_id, student_id=user.id, limit=50)
    messages_formatted = []
    for i, m in enumerate(history):
        msg_id = m.get("id") or f"msg-{i}"
        role = m.get("role") or "user"
        content = m.get("content") or ""
        created_at = m.get("created_at") or datetime.now(timezone.utc).isoformat()
        messages_formatted.append({
            "id": str(msg_id),
            "role": role,
            "content": content,
            "createdAt": str(created_at)
        })

    session_obj = {
        "id": f"{course_id}:{user.id}",
        "studentId": user.id,
        "courseId": course_id,
        "title": f"{course_id} AI Tutor",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "messages": messages_formatted
    }

    return {
        "session_id": f"{course_id}:{user.id}",
        "course_id": course_id,
        "student_id": user.id,
        "active": True,
        "messages": history,
        "data": {
            "session": session_obj,
            "course": {
                "id": course_id,
                "name": course_id,
                "code": course_id
            }
        },
        "message": "Session history retrieved successfully"
    }

@app.post("/ai/chat/course", tags=["mobile_alias"])
async def mobile_course_chat(
    body: MobileAIChatRequest,
    user: AuthenticatedUser = Depends(get_current_user),
    tutor_service: TutorService = Depends(get_tutor_service)
):
    answer = await tutor_service.ask_tutor(
        course_id=body.courseId,
        student_id=user.id,
        question=body.message
    )
    
    content_parts = []
    if answer.grounded_answer:
        content_parts.append(answer.grounded_answer)
    if answer.supplementary_answer:
        content_parts.append(answer.supplementary_answer)
    
    full_response = "\n\n".join(content_parts) or "No response could be generated."
    sources = answer.cited_chunk_ids or []
    is_grounded = bool(answer.grounded_answer and answer.grounded_answer.strip())
    if getattr(answer, "confidence", 0.0) > 0:
        confidence = answer.confidence
    elif is_grounded and sources:
        confidence = min(0.98, round(0.85 + (len(sources) * 0.04), 2))
    elif is_grounded:
        confidence = 0.80
    else:
        confidence = 0.50

    return {
        "data": {
            "response": full_response,
            "confidence": confidence,
            "sources": sources,
            "suggestions": answer.gaps_addressed or [],
            "cached": False,
            "model": settings.LLM_MODEL,
            "is_supplementary": bool(answer.supplementary_answer and not answer.grounded_answer)
        },
        "message": "Response generated successfully",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sessionId": f"{body.courseId}:{user.id}"
    }

@app.delete("/ai/sessions/{session_id}", tags=["mobile_alias"])
async def mobile_delete_session(
    session_id: str,
    user: AuthenticatedUser = Depends(get_current_user),
    message_repo: MessageRepository = Depends(get_message_repository)
):
    course_id = session_id.split(":")[0] if ":" in session_id else session_id
    deleted_count = await message_repo.clear_history(course_id=course_id, student_id=user.id)
    return {
        "message": "Chat history cleared successfully.",
        "deleted_count": deleted_count
    }

# Global Exception Handlers
@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "status_code": exc.status_code,
            "detail": exc.detail
        }
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled server error: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": True,
            "status_code": 500,
            "detail": "An internal server error occurred."
        }
    )
