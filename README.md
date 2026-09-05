# AI Course Tutor — RAG System Backend Service

FastAPI-based microservice implementing header-aware document ingestion, vector similarity search (`pgvector`), and 2-pass AI tutoring for student course materials.

---

## 🚀 Features

- **Private Student Course Corpora**: Per-student material isolation for uploading PDFs, DOCX, TXT, and image files.
- **Smart Document Parsing & Chunking**: Header-aware and slide-boundary aware text splitting with sha256 content deduplication.
- **2-Pass Grounded AI Tutor**:
  - **Pass 1 (Grounded Answer)**: Strictly answers student questions using relevant course material chunks retrieved via `pgvector` similarity search.
  - **Pass 2 (Supplementary Information)**: Identifies missing topic gaps and provides clearly labeled supplementary context.
- **Chat History & Session Management**: Seamlessly persists and retrieves conversation turns per course.
- **API Gateway Proxy Ready**: Fully integrated with the main Express API Gateway (`univibe_backend`) and mobile frontend (`app`).

---

## 🛠️ Technology Stack

- **Framework**: Python 3.10+ & FastAPI
- **Server**: Uvicorn (ASGI)
- **Database**: PostgreSQL 15+ with `pgvector` extension (SQLAlchemy AsyncIO + asyncpg)
- **Cache & Rate Limiter**: Redis
- **AI Models**: OpenAI (`gpt-4o-mini`, `text-embedding-3-small`) / Google Gemini SDK
- **Testing**: Pytest & Pytest-Asyncio

---

## ⚙️ Prerequisites

1. **Python**: Python 3.10 or higher.
2. **PostgreSQL with pgvector**: Running PostgreSQL on port `5433` (e.g., using `pgvector/pgvector:pg15` Docker container).
3. **Redis**: Running Redis instance on port `6379`.

---

## 📥 Environment Configuration (`.env`)

Create or update `.env` in the root of `rag_backend`:

```env
ENVIRONMENT="development"
CORS_ORIGINS="*"

# Database & Cache
DATABASE_URL="postgresql+asyncpg://postgres:12345@localhost:5433/uniVibes_rag"
REDIS_URL="redis://localhost:6379"

# Security (Matches main backend JWT_SECRET)
JWT_SECRET="martiedcknrdcrdkcndklncfdvc"

# AI Provider API Keys
OPENAI_API_KEY="sk-proj-..."
GEMINI_API_KEY=""

# Model Settings
EMBEDDING_MODEL="text-embedding-3-small"
LLM_MODEL="gpt-4o-mini"
STORAGE_DIR="./media_storage"
```

---

## 🚀 Quickstart: How to Start the App

### 1. Set Up Virtual Environment & Install Dependencies

```bash
# Navigate to rag_backend directory
cd rag_backend

# Create virtual environment
python -m venv .venv

# Activate virtual environment (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Start PostgreSQL Container with pgvector

Ensure your PostgreSQL container is running (the `docker-compose.yml` file is located in `univibe_backend`):

```bash
# From univibe_backend directory
cd ../univibe_backend
docker compose up -d db

# Or directly using the file path:
docker compose -f ../univibe_backend/docker-compose.yml up -d db
```

### 3. Start the RAG Backend Service

Run the Uvicorn development server on port `8000`:

```bash
# Using active virtual environment
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

*(Note: On server startup, `main.py` automatically initializes the database tables and enables `CREATE EXTENSION IF NOT EXISTS vector;`).*

### 4. Verify Health Endpoint

Open your browser or terminal:

```bash
curl http://localhost:8000/health
```

Expected Output:
```json
{
  "status": "healthy",
  "service": "rag_backend",
  "version": "1.0.0",
  "environment": "development"
}
```

---

## 🧪 Running Tests

To run the complete test suite (unit tests, RAG pipeline tests, and API integration tests):

```bash
# Run pytest in virtual environment
pytest
```

---

## 📡 API Endpoints Summary

### 📚 Course Material Ingestion
- `GET /courses/{course_id}/materials` — List uploaded materials for a course.
- `POST /courses/{course_id}/materials/upload` — Upload PDF/DOCX/Image for parsing, chunking, and embedding.

### 🤖 AI Tutor
- `POST /courses/{course_id}/ask` — Submit a question to the 2-pass grounded RAG pipeline.
- `GET /courses/{course_id}/history` — Retrieve chat history.
- `GET /courses/{course_id}/gaps` — List identified material coverage gaps.

### 📱 Mobile App Gateway Aliases
- `GET /ai/courses/{course_id}/chats/session` — Retrieve active chat session history.
- `POST /ai/chat/course` — Submit question from mobile frontend.
- `DELETE /ai/sessions/{session_id}` — Clear chat history for a session.
