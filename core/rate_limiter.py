import time
from collections import defaultdict
from shared.interfaces import RateLimiter

class InMemoryRateLimiter(RateLimiter):
    """In-memory sliding window rate limiter."""
    def __init__(self):
        self._requests: dict[str, list[float]] = defaultdict(list)

    async def is_rate_limited(self, key: str, limit: int, window_seconds: int) -> tuple[bool, int]:
        now = time.time()
        cutoff = now - window_seconds
        
        # Clean up old timestamps for key
        timestamps = [t for t in self._requests[key] if t > cutoff]
        
        if len(timestamps) >= limit:
            self._requests[key] = timestamps
            return True, 0

        timestamps.append(now)
        self._requests[key] = timestamps
        remaining = limit - len(timestamps)
        return False, remaining
