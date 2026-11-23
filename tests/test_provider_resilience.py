"""Provider outage handling (503 instead of 500) and event-loop non-blocking guarantees."""
import asyncio
import time
from typing import Any
from unittest.mock import AsyncMock

import httpx
import jwt
import pytest

from config import settings
from main import app
from shared.dependencies import get_tutor_service
from shared.interfaces import Embedder, ChatModel, ProviderUnavailableError
from tutor.llm import MockChatModel, MockEmbedder, OpenAIEmbedder
from tutor.service import TutorService

PROVIDER_DELAY_S = 0.5
CONCURRENT_REQUESTS = 5


class FailingEmbedder(Embedder):
    def embed(self, text: str) -> list[float]:
        raise ProviderUnavailableError("OpenAI embedding provider error: credit_balance_exhausted")

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        raise ProviderUnavailableError("OpenAI embedding provider error: credit_balance_exhausted")


class SlowEmbedder(Embedder):
    """Simulates a blocking network call made by a synchronous provider SDK."""
    def embed(self, text: str) -> list[float]:
        time.sleep(PROVIDER_DELAY_S)
        return MockEmbedder().embed(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        time.sleep(PROVIDER_DELAY_S)
        return MockEmbedder().embed_batch(texts)


class SlowChatModel(ChatModel):
    def complete(self, messages: list[dict[str, Any]], response_schema: type | None = None, max_tokens: int | None = None) -> Any:
        time.sleep(PROVIDER_DELAY_S)
        return MockChatModel().complete(messages, response_schema=response_schema, max_tokens=max_tokens)


def build_service(embedder: Embedder, chat_model: ChatModel) -> TutorService:
    vector_store = AsyncMock()
    vector_store.search_chunks.return_value = [
        {"chunk_id": "sample-chunk-1", "content": "An algorithm is a step-by-step procedure.", "filename": "l1.md", "similarity": 0.9}
    ]
    message_repo = AsyncMock()
    message_repo.get_history.return_value = []
    profile_repo = AsyncMock()
    profile_repo.get_profile.return_value = None
    return TutorService(
        vector_store=vector_store,
        message_repo=message_repo,
        gap_repo=AsyncMock(),
        embedder=embedder,
        chat_model=chat_model,
        profile_repo=profile_repo,
    )


@pytest.fixture
def auth_headers():
    token = jwt.encode({"userId": "resilience-student"}, settings.JWT_SECRET, algorithm="HS256")
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def override_tutor_service():
    def _set(service: TutorService):
        app.dependency_overrides[get_tutor_service] = lambda: service
    yield _set
    app.dependency_overrides.clear()


def make_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


# ---------- #1: provider outage -> clean 503 ----------

def test_openai_embedder_wraps_errors_as_provider_unavailable(monkeypatch):
    import openai

    class ExplodingClient:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("Error code: 429 - insufficient_quota")

    monkeypatch.setattr(openai, "OpenAI", ExplodingClient)
    with pytest.raises(ProviderUnavailableError, match="insufficient_quota"):
        OpenAIEmbedder(api_key="sk-test-key").embed("hello")


@pytest.mark.parametrize("path,body", [
    ("/courses/cs101/ask", {"question": "What is an algorithm?"}),
    ("/ai/chat/course", {"message": "What is an algorithm?", "courseId": "cs101"}),
])
async def test_provider_outage_returns_503_not_500(path, body, auth_headers, override_tutor_service, caplog):
    override_tutor_service(build_service(FailingEmbedder(), MockChatModel()))

    async with make_client() as client:
        res = await client.post(path, json=body, headers=auth_headers)

    assert res.status_code == 503
    assert res.headers.get("retry-after") == "60"
    data = res.json()
    assert data["error"] is True
    assert "temporarily unavailable" in data["detail"]
    # Provider internals (quota/billing details) must not leak to clients...
    assert "credit_balance_exhausted" not in res.text
    # ...but must be logged for operators.
    assert any("credit_balance_exhausted" in r.getMessage() and r.levelname == "ERROR" for r in caplog.records)


# ---------- #2: blocking provider calls don't stall the event loop ----------

async def test_concurrent_asks_run_in_parallel():
    service = build_service(SlowEmbedder(), SlowChatModel())

    start = time.perf_counter()
    answers = await asyncio.gather(*[
        service.ask_tutor(course_id="cs101", student_id=f"s{i}", question="What is an algorithm?")
        for i in range(CONCURRENT_REQUESTS)
    ])
    elapsed = time.perf_counter() - start

    assert all(a.grounded_answer for a in answers)
    # Each ask = embed + pass-1 completion = 2 * delay. Serialized, 5 asks would take ~5s.
    per_ask = 2 * PROVIDER_DELAY_S
    assert elapsed < per_ask * 2, f"asks appear serialized: {elapsed:.2f}s for {CONCURRENT_REQUESTS} requests"


async def test_health_stays_responsive_while_ask_in_flight(auth_headers, override_tutor_service):
    override_tutor_service(build_service(SlowEmbedder(), SlowChatModel()))

    async with make_client() as client:
        # Measure from the moment the ask is scheduled: if a provider call blocks the
        # event loop, /health can't even be dispatched until that call returns.
        start = time.perf_counter()
        ask_task = asyncio.create_task(
            client.post("/courses/cs101/ask", json={"question": "What is an algorithm?"}, headers=auth_headers)
        )
        await asyncio.sleep(0.05)  # let the ask reach the provider call
        health = await client.get("/health")
        health_latency = time.perf_counter() - start
        ask_done_before_health = ask_task.done()

        ask_res = await ask_task

    assert health.status_code == 200
    assert not ask_done_before_health, "ask finished before /health was served; test is not exercising concurrency"
    assert health_latency < PROVIDER_DELAY_S, f"/health blocked for {health_latency:.2f}s by an in-flight ask"
    assert ask_res.status_code == 200
