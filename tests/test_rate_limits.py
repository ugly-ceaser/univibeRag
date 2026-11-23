"""Spend-path guards: per-student rate limits, request size caps, upload size caps."""
from unittest.mock import AsyncMock, MagicMock

import jwt
import pytest
from fastapi.testclient import TestClient

from config import settings
from main import app
from shared.dependencies import (
    get_tutor_service,
    get_asset_repository,
    get_vector_store,
    get_file_storage,
)
from tutor.llm import MockChatModel, MockEmbedder
from tests.test_provider_resilience import build_service

ASK = "/courses/cs101/ask"
MOBILE_CHAT = "/ai/chat/course"
UPLOAD = "/courses/cs101/materials/upload"


def ask_body(path: str, text: str = "What is an algorithm?") -> dict:
    return {"question": text} if path == ASK else {"message": text, "courseId": "cs101"}


def headers_for(user_id: str) -> dict:
    token = jwt.encode({"userId": user_id}, settings.JWT_SECRET, algorithm="HS256")
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def client():
    asset_repo = AsyncMock()
    asset_repo.find_by_hash.return_value = None
    asset_repo.create_asset.return_value = MagicMock(id="asset-1", status="pending")
    storage = MagicMock()  # never touch the real media_storage dir
    storage.save_file.return_value = "mem://file"

    app.dependency_overrides[get_tutor_service] = lambda: build_service(MockEmbedder(), MockChatModel())
    app.dependency_overrides[get_asset_repository] = lambda: asset_repo
    app.dependency_overrides[get_vector_store] = lambda: AsyncMock()
    app.dependency_overrides[get_file_storage] = lambda: storage
    yield TestClient(app)
    app.dependency_overrides.clear()


# ---------- ask / chat rate limits ----------

@pytest.mark.parametrize("path", [ASK, MOBILE_CHAT])
def test_ask_limit_uses_settings_default(client, path):
    h = headers_for("student-default")
    limit = settings.RATE_LIMIT_ASK_PER_MIN
    assert limit == 15

    for i in range(limit):
        res = client.post(path, json=ask_body(path), headers=h)
        assert res.status_code == 200, f"request {i + 1} unexpectedly rejected: {res.text}"

    res = client.post(path, json=ask_body(path), headers=h)
    assert res.status_code == 429
    assert res.headers.get("retry-after") == "60"
    body = res.json()
    assert body["error"] is True
    assert "max 15 questions per minute" in body["detail"]
    # The app reads retryAfter from the body first (Retry-After header may be hidden by CORS on web).
    assert body["retryAfter"] == 60


def test_ask_and_mobile_chat_share_one_bucket(client, monkeypatch):
    monkeypatch.setattr(settings, "RATE_LIMIT_ASK_PER_MIN", 4)
    h = headers_for("student-shared")

    for path in [ASK, MOBILE_CHAT, ASK, MOBILE_CHAT]:
        assert client.post(path, json=ask_body(path), headers=h).status_code == 200

    # Budget is spent on both routes, not 4 per route.
    assert client.post(ASK, json=ask_body(ASK), headers=h).status_code == 429
    assert client.post(MOBILE_CHAT, json=ask_body(MOBILE_CHAT), headers=h).status_code == 429


def test_limit_is_per_student(client, monkeypatch):
    monkeypatch.setattr(settings, "RATE_LIMIT_ASK_PER_MIN", 1)
    alice, bob = headers_for("alice"), headers_for("bob")

    assert client.post(ASK, json=ask_body(ASK), headers=alice).status_code == 200
    assert client.post(ASK, json=ask_body(ASK), headers=alice).status_code == 429
    assert client.post(ASK, json=ask_body(ASK), headers=bob).status_code == 200


# ---------- question size caps ----------

@pytest.mark.parametrize("path", [ASK, MOBILE_CHAT])
def test_oversized_question_rejected(client, path):
    res = client.post(path, json=ask_body(path, "x" * 1001), headers=headers_for("student-size"))
    assert res.status_code == 422


@pytest.mark.parametrize("path", [ASK, MOBILE_CHAT])
def test_max_length_question_accepted(client, path):
    res = client.post(path, json=ask_body(path, "x" * 1000), headers=headers_for("student-size-ok"))
    assert res.status_code == 200


def test_empty_mobile_message_rejected(client):
    res = client.post(MOBILE_CHAT, json=ask_body(MOBILE_CHAT, ""), headers=headers_for("student-empty"))
    assert res.status_code == 422


# ---------- upload limits ----------

def upload(client, h, content: bytes, name: str = "notes.md"):
    return client.post(UPLOAD, files={"file": (name, content, "text/markdown")}, headers=h)


def test_upload_rate_limit(client):
    h = headers_for("student-uploader")
    limit = settings.RATE_LIMIT_UPLOAD_PER_MIN
    assert limit == 5

    for i in range(limit):
        assert upload(client, h, f"# Lecture {i}\nbody".encode()).status_code == 202

    res = upload(client, h, b"# Lecture 6\nbody")
    assert res.status_code == 429
    assert res.headers.get("retry-after") == "60"
    assert "max 5 uploads per minute" in res.json()["detail"]
    assert res.json()["retryAfter"] == 60


def test_upload_alias_route_shares_upload_limit(client, monkeypatch):
    monkeypatch.setattr(settings, "RATE_LIMIT_UPLOAD_PER_MIN", 1)
    h = headers_for("student-alias")
    assert upload(client, h, b"# a\nbody").status_code == 202
    res = client.post("/courses/cs101/materials", files={"file": ("b.md", b"# b\nbody", "text/markdown")}, headers=h)
    assert res.status_code == 429


def test_upload_size_cap(client, monkeypatch):
    monkeypatch.setattr(settings, "MAX_UPLOAD_MB", 1)
    h = headers_for("student-bigfile")
    one_mb = 1024 * 1024

    assert upload(client, h, b"a" * one_mb).status_code == 202

    res = upload(client, h, b"a" * (one_mb + 1))
    assert res.status_code == 413
    assert "1 MB upload limit" in res.json()["detail"]


def test_unauthenticated_401_keeps_www_authenticate_header(client):
    res = client.post(ASK, json=ask_body(ASK), headers={"Authorization": "Bearer not-a-jwt"})
    assert res.status_code == 401
    assert res.headers.get("www-authenticate") == "Bearer"
