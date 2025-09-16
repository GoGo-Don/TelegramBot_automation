"""
LLM Request and Response Models

Defines data structures for handling LLM prompts, requests,
responses, and conversation context.

Author: Development Team
Date: 2025-09-16
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class PromptTemplate(BaseModel):
    name: str
    system_prompt: Optional[str]
    user_prompt: str
    variables: List[str]


class LLMRequest(BaseModel):
    system_prompt: Optional[str]
    user_prompt: str
    model: str
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    has_images: bool = False
    image_data: Optional[List[str]] = None  # List of base64-encoded images
    request_id: Optional[str] = None


class LLMResponse(BaseModel):
    content: str
    reasoning: Optional[str]
    confidence: Optional[float]
    usage: Optional[Dict[str, int]] = None
    provider: Optional[str]
    model: Optional[str]


class ConversationContext(BaseModel):
    conversation_id: str
    messages: List[Dict[str, Any]]
    last_updated: Optional[str]
