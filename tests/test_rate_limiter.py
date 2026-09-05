import pytest
import asyncio
from core.rate_limiter import InMemoryRateLimiter

@pytest.mark.asyncio
async def test_rate_limiter_allows_under_threshold():
    """Verify that requests below the threshold limit are allowed with correct remaining quota."""
    limiter = InMemoryRateLimiter()
    key = "user-123:upload"
    
    is_limited, remaining = await limiter.is_rate_limited(key, limit=3, window_seconds=60)
    assert is_limited is False
    assert remaining == 2
    
    is_limited, remaining = await limiter.is_rate_limited(key, limit=3, window_seconds=60)
    assert is_limited is False
    assert remaining == 1

@pytest.mark.asyncio
async def test_rate_limiter_blocks_exceeding_threshold():
    """Verify that exceeding threshold limit returns is_limited=True and remaining=0."""
    limiter = InMemoryRateLimiter()
    key = "user-456:ask"
    limit = 2
    
    # 2 requests allowed
    await limiter.is_rate_limited(key, limit=limit, window_seconds=60)
    await limiter.is_rate_limited(key, limit=limit, window_seconds=60)
    
    # 3rd request blocked
    is_limited, remaining = await limiter.is_rate_limited(key, limit=limit, window_seconds=60)
    assert is_limited is True
    assert remaining == 0

@pytest.mark.asyncio
async def test_rate_limiter_resets_after_window_expiry():
    """Verify that requests are allowed again after the sliding window expires."""
    limiter = InMemoryRateLimiter()
    key = "user-789:upload"
    
    # 1 request limit with 1-second window
    await limiter.is_rate_limited(key, limit=1, window_seconds=1)
    
    # Blocked immediately
    is_limited, _ = await limiter.is_rate_limited(key, limit=1, window_seconds=1)
    assert is_limited is True
    
    # Wait for window to expire
    await asyncio.sleep(1.1)
    
    # Allowed again
    is_limited, remaining = await limiter.is_rate_limited(key, limit=1, window_seconds=1)
    assert is_limited is False
    assert remaining == 0
