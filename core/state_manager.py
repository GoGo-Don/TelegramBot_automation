"""
StateManager - Centralized State Management for Telegram LLM Decision Engine

Features:
- Async persistent state store with Redis
- Database fallback for long-term persistence (SQLite/PostgreSQL via SQLAlchemy)
- Session management (user, conversation/task states)
- In-memory caching for fast access
- Comprehensive error handling & verbose documentation

Author: Development Team
Date: 2025-09-16
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

# Example models - replace with your real models as needed!
from models.data_models import ProcessingTask
from redis import asyncio as aioredis
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker


class StateManagerError(Exception):
    """Generic error for state manager operations."""

    pass


class SessionNotFoundError(StateManagerError):
    """Raised if session could not be loaded/found."""

    pass


class StateManager:
    """
    StateManager - Async state handling for users, sessions, and tasks.

    Stores session state in Redis, falls back to database for audit/log persistence.
    Caching ensures high performance for frequent access.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        session_timeout_sec: int = 1800,
        database_url: str = "sqlite+aiosqlite:///data/app.db",
    ):
        """
        Initialize StateManager with Redis and DB connection.

        redis_url: Redis server connection string.
        session_timeout_sec: Session timeout in seconds.
        database_url: SQLAlchemy connection string.
        """
        self.redis_url = redis_url
        self.session_timeout_sec = session_timeout_sec
        self.database_url = database_url

        self.redis: Optional[aioredis.Redis] = None
        self.engine = create_async_engine(database_url, future=True, echo=False)
        self.SessionLocal = sessionmaker(
            self.engine, expire_on_commit=False, class_=AsyncSession
        )
        self.logger = logging.getLogger(__name__)

    async def initialize(self) -> None:
        """Connect to Redis and test DB."""
        self.redis = await aioredis.create_redis_pool(self.redis_url)
        async with self.engine.begin() as conn:
            await conn.run_sync(lambda conn: None)  # Connection test

    async def close(self) -> None:
        """Gracefully close Redis and DB."""
        if self.redis:
            self.redis.close()
            await self.redis.wait_closed()
        await self.engine.dispose()
        self.logger.info("StateManager closed all connections.")

    # User session management

    async def save_user_session(self, user_id: int, session_data: dict) -> None:
        """
        Save or update a user session.

        user_id: Telegram User ID
        session_data: Custom dict to store
        """
        key = f"user_session:{user_id}"
        value = json.dumps(session_data)
        await self.redis.set(key, value, expire=self.session_timeout_sec)
        self.logger.debug(f"Session saved user {user_id} [{key}]")

    async def get_user_session(self, user_id: int) -> Optional[dict]:
        """
        Retrieve a user session.

        Returns: Session data dict or None
        Raises: SessionNotFoundError if missing
        """
        key = f"user_session:{user_id}"
        value = await self.redis.get(key)
        if not value:
            self.logger.warning(f"Session not found for user {user_id}")
            raise SessionNotFoundError(f"Session not found for user {user_id}")
        return json.loads(value)

    async def clear_user_session(self, user_id: int) -> None:
        """Delete a user session."""
        key = f"user_session:{user_id}"
        await self.redis.delete(key)
        self.logger.info(f"Session cleared for user {user_id}")

    # Processing task state management

    async def save_task_state(self, task: ProcessingTask) -> None:
        """
        Save task state to Redis (fast) and DB for persistence.

        task: ProcessingTask dataclass/object.
        """
        key = f"task:{task.id}"
        value = json.dumps(task.to_dict())
        await self.redis.set(key, value, expire=60 * 60)
        # Persist to DB (assume SQLAlchemy model ProcessingTask)
        async with self.SessionLocal() as session:
            await session.merge(task)
            await session.commit()
        self.logger.debug(f"Task {task.id} state saved and persisted.")

    async def get_task_state(self, task_id: str) -> Optional[dict]:
        """Retrieve task state from Redis or DB."""
        key = f"task:{task_id}"
        value = await self.redis.get(key)
        if value:
            return json.loads(value)
        # Fallback: DB
        async with self.SessionLocal() as session:
            obj = await session.get(ProcessingTask, task_id)
            if obj:
                return obj.to_dict()
        return None

    async def flush_all(self) -> None:
        """Flush or persist all states as needed."""
        # Extend: Implement as needed for your app (batch flush to DB, etc.)
        self.logger.info("Flushed all state (implement batch logic if needed).")

    # Utility

    async def get_metrics(self) -> Dict[str, Any]:
        """Return simple metrics for monitoring."""
        # Implement more advanced metrics as needed
        kc = await self.redis.keys("user_session:*")
        total_sessions = len(kc) if kc else 0
        return {"total_sessions": total_sessions}

    async def health_check(self) -> Dict[str, Any]:
        """
        Health check for state manager and dependencies.
        """
        result = {"status": "healthy", "redis": None, "database": None}
        try:
            pong = await self.redis.ping()
            result["redis"] = pong
        except Exception as e:
            result["status"] = "degraded"
            result["redis"] = str(e)
        try:
            async with self.engine.begin() as conn:
                await conn.run_sync(lambda conn: None)
            result["database"] = "OK"
        except Exception as e:
            result["status"] = "degraded"
            result["database"] = str(e)
        return result


# Export usage
__all__ = ["StateManager", "StateManagerError", "SessionNotFoundError"]
