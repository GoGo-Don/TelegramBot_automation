"""
Session Store Module

Manages user session storage and retrieval, backed by Redis,
with serialization and expiration handling.

Author: Development Team
Date: 2025-09-16
"""

import json
import logging
from typing import Any, Dict, Optional

from redis import asyncio as aioredis

logger = logging.getLogger(__name__)


class SessionStore:
    """
    Session storage manager backed by Redis.
    """

    def __init__(
        self,
        redis_url: str,
        session_prefix: str = "session:",
        session_ttl_sec: int = 1800,
    ):
        self.redis_url = redis_url
        self.session_prefix = session_prefix
        self.session_ttl_sec = session_ttl_sec
        self.redis: Optional[aioredis.Redis] = None

    async def initialize(self):
        self.redis = await aioredis.create_redis_pool(self.redis_url)
        logger.info("SessionStore connected to Redis.")

    async def save_session(self, user_id: int, session_data: Dict[str, Any]):
        key = f"{self.session_prefix}{user_id}"
        value = json.dumps(session_data)
        await self.redis.set(key, value, expire=self.session_ttl_sec)
        logger.debug(f"Session saved for user {user_id}")

    async def load_session(self, user_id: int) -> Optional[Dict[str, Any]]:
        key = f"{self.session_prefix}{user_id}"
        raw = await self.redis.get(key)
        if not raw:
            logger.debug(f"No session found for user {user_id}")
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse session data for user {user_id}: {e}")
            return None

    async def clear_session(self, user_id: int):
        key = f"{self.session_prefix}{user_id}"
        await self.redis.delete(key)
        logger.info(f"Session cleared for user {user_id}")

    async def close(self):
        if self.redis:
            self.redis.close()
            await self.redis.wait_closed()
            logger.info("SessionStore closed Redis connection.")
