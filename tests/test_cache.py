import pytest
import asyncio
from core.cache import InMemoryCache

@pytest.mark.asyncio
async def test_cache_set_and_get():
    """Verify setting and retrieving values from cache."""
    cache = InMemoryCache()
    await cache.set("gap:SN1", "SN1 is a unimolecular substitution step.")
    
    val = await cache.get("gap:SN1")
    assert val == "SN1 is a unimolecular substitution step."

@pytest.mark.asyncio
async def test_cache_miss_returns_none():
    """Verify cache miss returns None."""
    cache = InMemoryCache()
    val = await cache.get("nonexistent_key")
    assert val is None

@pytest.mark.asyncio
async def test_cache_ttl_expiration():
    """Verify cached items expire after specified TTL."""
    cache = InMemoryCache()
    await cache.set("temp_key", "temp_value", ttl_seconds=1)
    
    val = await cache.get("temp_key")
    assert val == "temp_value"
    
    # Wait for TTL expiration
    await asyncio.sleep(1.1)
    
    val_after = await cache.get("temp_key")
    assert val_after is None

@pytest.mark.asyncio
async def test_cache_delete():
    """Verify explicit key deletion."""
    cache = InMemoryCache()
    await cache.set("key_to_delete", "data")
    await cache.delete("key_to_delete")
    
    val = await cache.get("key_to_delete")
    assert val is None
