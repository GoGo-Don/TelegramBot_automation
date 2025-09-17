"""
Telegram Data Models

Defines data structures representing Telegram entities, messages,
media files, user sessions, and conversation states.

Author: GG
Date: 2025-09-16
"""

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class TelegramUser(BaseModel):
    id: int
    username: Optional[str]
    first_name: Optional[str]
    last_name: Optional[str]

    @classmethod
    def from_telegram_user(cls, user) -> "TelegramUser":
        return cls(
            id=user.id,
            username=user.username,
            first_name=user.first_name,
            last_name=user.last_name,
        )


class TelegramMessage(BaseModel):
    message_id: int
    from_user: TelegramUser
    text: Optional[str]
    timestamp: datetime


class MediaType(str, Enum):
    photo = "photo"
    video = "video"
    document = "document"


class MediaFile(BaseModel):
    file_id: str
    file_unique_id: str
    file_type: MediaType
    file_size: Optional[int]
    width: Optional[int]
    height: Optional[int]
    duration: Optional[int]
    caption: Optional[str]
    file_name: Optional[str]
    mime_type: Optional[str]
    local_path: Optional[str] = None
    file_hash: Optional[str] = None
    downloaded_at: Optional[datetime] = None


class ConversationState(str, Enum):
    IDLE = "idle"
    COLLECTING = "collecting"
    WAITING_CONFIRMATION = "waiting_confirmation"
    PROCESSING = "processing"


class UserSession(BaseModel):
    user_id: int
    user: TelegramUser
    created_at: datetime
    last_active_at: Optional[datetime] = None
    collected_media: List[MediaFile] = Field(default_factory=list)
    conversation_history: List[TelegramMessage] = Field(default_factory=list)
    current_state: ConversationState = ConversationState.IDLE

    def add_message(self, message: TelegramMessage):
        self.conversation_history.append(message)
        self.last_active_at = message.timestamp

    def to_dict(self) -> dict:
        return self.dict()
