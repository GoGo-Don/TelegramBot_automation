"""
Core Data Models

Includes processing tasks, statuses, and priority enumerations.
These models represent application-level business entities.

Author: Development Team
Date: 2025-09-16
"""

from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime
from typing import Dict, Any, Optional


class TaskStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Priority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


class ProcessingTask(BaseModel):
    id: str
    user_id: int
    task_type: str
    data: Dict[str, Any]
    priority: Priority = Priority.NORMAL
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    failed_at: Optional[datetime] = None
    error_message: Optional[str] = None

    def to_dict(self) -> dict:
        return self.dict()
