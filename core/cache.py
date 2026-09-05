import time
from typing import NamedTuple
from shared.interfaces import CacheClient

class CacheEntry(NamedTuple):
    value: str
    expires_at: float | None

class InMemoryCache(CacheClient):
    """In-memory key-value cache client with TTL support."""
    def __init__(self):
        self._store: dict[str, CacheEntry] = {}

    async def get(self, key: str) -> str | None:
        entry = self._store.get(key)
        if entry is None:
            return None
            
        if entry.expires_at is not None and time.time() > entry.expires_at:
            del self._store[key]
            return None
            
        return entry.value

    async def set(self, key: str, value: str, ttl_seconds: int | None = None) -> None:
        expires_at = (time.time() + ttl_seconds) if ttl_seconds is not None else None
        self._store[key] = CacheEntry(value=value, expires_at=expires_at)

    async def delete(self, key: str) -> None:
        self._store.pop(key, None)
