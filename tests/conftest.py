import pytest
from config import settings


@pytest.fixture(autouse=True)
def no_real_llm_providers(monkeypatch):
    """Keep the test suite hermetic: never call real OpenAI/Gemini APIs.

    Without this, a real key in .env makes tests spend money and fail whenever
    the provider account is out of quota. Tests that need a key patch it
    explicitly (e.g. patch.object(settings, "OPENAI_API_KEY", ...)), which
    still works because those patches are applied after this fixture.
    """
    monkeypatch.setattr(settings, "OPENAI_API_KEY", "")
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)


@pytest.fixture(autouse=True)
def fresh_rate_limiter(monkeypatch):
    """Give every test an empty rate limiter so request counts never leak between tests."""
    import shared.dependencies
    from core.rate_limiter import InMemoryRateLimiter

    limiter = InMemoryRateLimiter()
    monkeypatch.setattr(shared.dependencies, "_rate_limiter_instance", limiter)
    return limiter
