import uuid
import jwt
import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from config import settings
from main import app
from shared.interfaces import AssetRepository, VectorStore, MessageRepository, GapRepository
from shared.dependencies import (
    get_asset_repository,
    get_vector_store,
    get_message_repository,
    get_gap_repository
)

class InMemoryAssetRepository(AssetRepository):
    def __init__(self):
        self.assets = {}
        self.documents = []

    async def find_by_hash(self, content_hash: str):
        for a in self.assets.values():
            if a["content_hash"] == content_hash:
                return MagicMock(id=a["id"], status=a["status"])
        return None

    async def create_asset(self, content_hash: str, storage_path: str):
        asset_id = str(uuid.uuid4())
        self.assets[asset_id] = {
            "id": asset_id,
            "content_hash": content_hash,
            "storage_path": storage_path,
            "status": "pending",
            "error_message": None
        }
        return MagicMock(id=asset_id, status="pending")

    async def update_status(self, asset_id: str, status: str, error_message: str | None = None):
        if asset_id in self.assets:
            self.assets[asset_id]["status"] = status
            self.assets[asset_id]["error_message"] = error_message

    async def link_document(self, course_id: str, student_id: str, asset_id: str, filename: str, mime_type: str):
        doc = {
            "id": str(uuid.uuid4()),
            "course_id": course_id,
            "student_id": student_id,
            "material_asset_id": asset_id,
            "filename": filename,
            "mime_type": mime_type
        }
        self.documents.append(doc)
        return doc

    async def get_documents_by_course(self, course_id: str, student_id: str):
        return [d for d in self.documents if d["course_id"] == course_id]


class InMemoryVectorStore(VectorStore):
    def __init__(self):
        self.chunks = []

    async def upsert_chunks(self, asset_id: str, chunks_with_vectors: list):
        for cwv in chunks_with_vectors:
            self.chunks.append({
                "asset_id": asset_id,
                "chunk_index": cwv.chunk_index,
                "content": cwv.text,
                "embedding": cwv.embedding,
                "metadata": cwv.metadata,
                "chunk_id": f"chunk-{len(self.chunks)}"
            })

    async def search_chunks(self, query_vector: list[float], course_id: str, top_k: int = 6):
        return [
            {"chunk_id": c["chunk_id"], "content": c["content"], "filename": "sample.md"}
            for c in self.chunks[:top_k]
        ]

    async def search_messages(self, query_vector: list[float], course_id: str, student_id: str, top_k: int = 3):
        return []


class InMemoryMessageRepository(MessageRepository):
    def __init__(self):
        self.messages = []

    async def save_message(self, course_id, student_id, role, content, embedding=None, grounded=True, sources=None):
        msg = {
            "id": str(uuid.uuid4()),
            "course_id": course_id,
            "student_id": student_id,
            "role": role,
            "content": content,
            "grounded": grounded,
            "sources": sources or []
        }
        self.messages.append(msg)
        return msg

    async def get_history(self, course_id, student_id, limit=50):
        return [m for m in self.messages if m["course_id"] == course_id and m["student_id"] == student_id][:limit]

    async def clear_history(self, course_id, student_id):
        before = len(self.messages)
        self.messages = [m for m in self.messages if not (m["course_id"] == course_id and m["student_id"] == student_id)]
        return before - len(self.messages)


class InMemoryGapRepository(GapRepository):
    def __init__(self):
        self.gaps = []

    async def get_for_course(self, course_id: str):
        return [g for g in self.gaps if g["course_id"] == course_id]

    async def cache_gap(self, course_id: str, topic: str, topic_embedding: list, supplementary_answer: str):
        gap = {
            "id": str(uuid.uuid4()),
            "course_id": course_id,
            "topic": topic,
            "supplementary_answer": supplementary_answer,
            "hit_count": 1
        }
        self.gaps.append(gap)
        return gap


@pytest.fixture(autouse=True)
def override_dependencies():
    asset_repo = InMemoryAssetRepository()
    vector_store = InMemoryVectorStore()
    message_repo = InMemoryMessageRepository()
    gap_repo = InMemoryGapRepository()

    app.dependency_overrides[get_asset_repository] = lambda: asset_repo
    app.dependency_overrides[get_vector_store] = lambda: vector_store
    app.dependency_overrides[get_message_repository] = lambda: message_repo
    app.dependency_overrides[get_gap_repository] = lambda: gap_repo

    yield

    app.dependency_overrides.clear()


@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def valid_auth_headers():
    payload = {
        "userId": "e2e-student-123",
        "role": "student",
        "department": "Computer Science",
        "level": 300
    }
    token = jwt.encode(payload, settings.JWT_SECRET, algorithm="HS256")
    return {"Authorization": f"Bearer {token}"}

def test_health_check(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["service"] == "rag_backend"

def test_unauthenticated_requests_blocked(client):
    res_upload = client.post("/courses/cs101/materials/upload")
    assert res_upload.status_code in [401, 403]

    res_ask = client.post("/courses/cs101/ask", json={"question": "What is AI?"})
    assert res_ask.status_code in [401, 403]

    res_history = client.get("/courses/cs101/history")
    assert res_history.status_code in [401, 403]

def test_full_e2e_rag_flow(client, valid_auth_headers):
    course_id = "cs101-intro"
    
    # Step 1: Upload a course material file
    file_content = b"# CS101: Introduction to Algorithms\nAn algorithm is a step-by-step procedure for solving a problem."
    files = {"file": ("lecture1.md", file_content, "text/markdown")}
    
    res_upload = client.post(
        f"/courses/{course_id}/materials/upload",
        files=files,
        headers=valid_auth_headers
    )
    assert res_upload.status_code == 202
    upload_data = res_upload.json()
    assert upload_data["status"] in ["pending", "ready"]
    assert "asset_id" in upload_data

    # Step 2: List course materials
    res_list = client.get(f"/courses/{course_id}/materials", headers=valid_auth_headers)
    assert res_list.status_code == 200
    assert res_list.json()["course_id"] == course_id

    # Step 3: Ask tutor a question
    res_ask = client.post(
        f"/courses/{course_id}/ask",
        json={"question": "What is an algorithm?"},
        headers=valid_auth_headers
    )
    assert res_ask.status_code == 200
    tutor_ans = res_ask.json()
    assert "grounded_answer" in tutor_ans
    assert "cited_chunk_ids" in tutor_ans

    # Step 4: Retrieve chat history
    res_hist = client.get(f"/courses/{course_id}/history", headers=valid_auth_headers)
    assert res_hist.status_code == 200
    assert res_hist.json()["count"] >= 2

    # Step 5: Mobile chat session endpoint alias
    res_mobile = client.get(f"/ai/courses/{course_id}/chats/session", headers=valid_auth_headers)
    assert res_mobile.status_code == 200
    mobile_data = res_mobile.json()
    assert mobile_data["active"] is True
    assert mobile_data["session_id"] == f"{course_id}:e2e-student-123"

    # Step 6: Clear chat history
    res_clear = client.delete(f"/courses/{course_id}/history", headers=valid_auth_headers)
    assert res_clear.status_code == 200
    assert res_clear.json()["deleted_count"] >= 2
