"""
Cache Manager Module

Manages caching for the application using Redis, supporting TTL, 
size limits, and async commands.

Author: Development Team
Date: 2025-09-16
"""

import logging
from typing import Any, Optional

from redis import asyncio as aioredis

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Async Redis-based cache manager.
    """

    def __init__(self, redis_url: str, ttl_seconds: int = 3600, max_size_mb: int = 100):
        self.redis_url = redis_url
        self.ttl_seconds = ttl_seconds
        self.max_size_mb = max_size_mb
        self.redis: Optional[aioredis.Redis] = None

    async def initialize(self):
        """
        Connect to Redis server.
        """
        self.redis = await aioredis.create_redis_pool(self.redis_url)
        logger.info("CacheManager connected to Redis.")

    async def get(self, key: str) -> Optional[bytes]:
        """
        Get value by key.

        Args:
            key: Cache key.

        Returns:
            Cached value or None.
        """
        if self.redis:
            return await self.redis.get(key)
        return None

    async def set(self, key: str, value: bytes, expire: Optional[int] = None) -> None:
        """
        Set key to value with optional expiry.

        Args:
            key: Cache key.
            value: Value to cache.
            expire: Expiry in seconds.
        """
        if self.redis:
            await self.redis.set(key, value, expire=expire or self.ttl_seconds)
            logger.debug(f"Cache set: {key}")

    async def delete(self, key: str) -> None:
        """
        Delete a key from cache.

        Args:
            key: Cache key.
        """
        if self.redis:
            await self.redis.delete(key)
            logger.debug(f"Cache deleted: {key}")

    async def clear_expired(self) -> None:
        """
        Redis handles TTL expiration automatically, but can implement cleanup logic if needed.
        """
        logger.debug("CacheManager clear_expired called (Redis auto expires keys).")

    async def health_check(self) -> dict:
        """
        Health check for Redis connection.

        Returns:
            Health status dict.
        """
        try:
            pong = await self.redis.ping()
            return {"status": "healthy" if pong else "unhealthy"}
        except Exception as e:
            logger.error(f"CacheManager health check failed: {e}")
            return {"status": "degraded", "error": str(e)}

    async def close(self):
        """
        Close Redis connection.
        """
        if self.redis:
            self.redis.close()
            await self.redis.wait_closed()
            logger.info("CacheManager closed Redis connection.")
