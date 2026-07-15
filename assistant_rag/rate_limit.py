"""Small rate limiter abstraction for API safety gates."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Protocol

from .contracts import RateLimitResult


class RateLimiter(Protocol):
    def allow(self, key: str, limit: int, window_seconds: int) -> RateLimitResult:
        ...


@dataclass
class InMemoryRateLimiter:
    _buckets: dict[str, list[float]] = field(default_factory=dict)

    def allow(self, key: str, limit: int, window_seconds: int) -> RateLimitResult:
        now = time.time()
        cutoff = now - window_seconds
        bucket = [stamp for stamp in self._buckets.get(key, []) if stamp > cutoff]
        if len(bucket) >= limit:
            retry_after = max(1, int(window_seconds - (now - bucket[0])))
            self._buckets[key] = bucket
            return RateLimitResult(allowed=False, retry_after_seconds=retry_after)
        bucket.append(now)
        self._buckets[key] = bucket
        return RateLimitResult(allowed=True)


@dataclass
class RedisRateLimiter:
    redis_url: str

    def __post_init__(self) -> None:
        import redis  # type: ignore

        self._client = redis.Redis.from_url(self.redis_url)

    def allow(self, key: str, limit: int, window_seconds: int) -> RateLimitResult:
        now_ms = int(time.time() * 1000)
        window_ms = window_seconds * 1000
        pipe = self._client.pipeline()
        pipe.zremrangebyscore(key, 0, now_ms - window_ms)
        pipe.zcard(key)
        pipe.zadd(key, {str(now_ms): now_ms})
        pipe.expire(key, window_seconds)
        _, count, _, _ = pipe.execute()
        if int(count) >= limit:
            oldest = self._client.zrange(key, 0, 0, withscores=True)
            retry_after = window_seconds
            if oldest:
                retry_after = max(1, int((window_ms - (now_ms - int(oldest[0][1]))) / 1000))
            return RateLimitResult(allowed=False, retry_after_seconds=retry_after)
        return RateLimitResult(allowed=True)
    
    
def build_rate_limiter() -> RateLimiter:
    redis_url = os.getenv("ASSISTANT_REDIS_URL")
    if redis_url:
        try:
            return RedisRateLimiter(redis_url)
        except ImportError:
            pass
    return InMemoryRateLimiter()
