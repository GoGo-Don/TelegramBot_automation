"""
Decision Engine Models

Defines request/response models and action enumerations for the decision engine.

Author: Development Team
Date: 2025-09-16
"""

from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel


class ActionType(str, Enum):
    CREATE_WOOCOMMERCE_POST = "CREATE_WOOCOMMERCE_POST"
    UPDATE_EXCEL = "UPDATE_EXCEL"
    REPLY_TELEGRAM = "REPLY_TELEGRAM"
    CHAIN_LLM = "CHAIN_LLM"
    REQUEST_MORE_DATA = "REQUEST_MORE_DATA"


class DecisionRequest(BaseModel):
    user_id: int
    analysis_result: Dict[str, Any]
    task: Optional[Any]  # Could be ProcessingTask or similar
    decision_data: Optional[Dict[str, Any]] = None


class DecisionResponse(BaseModel):
    action: ActionType
    parameters: Dict[str, Any]
    result: Optional[Any]
    success: bool = True
    message: Optional[str] = None
