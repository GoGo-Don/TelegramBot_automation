"""
Decision Engine Module

Implements intelligent decision-making logic for the Telegram LLM-based system.
Interprets LLM outputs and decides actions like WooCommerce posting,
Excel updates, follow-up questioning, or chaining further LLM calls.

Author: GG
Date: 2025-09-16
"""

import asyncio
import json
import logging
from typing import Any, Dict, Optional

from telegram import Bot

from core.llm_processor import LLMProcessor
from integrations.excel_handler import ExcelHandler
from integrations.woocommerce_handler import WooCommerceHandler
from models.decision_models import ActionType, DecisionRequest, DecisionResponse
from utils.exceptions import DecisionEngineError

logger = logging.getLogger(__name__)


class DecisionEngine:
    """
    Core decision engine that processes LLM results and triggers appropriate actions.
    """

    def __init__(
        self,
        llm_processor: LLMProcessor,
        woocommerce_handler: Optional[WooCommerceHandler] = None,
        excel_handler: Optional[ExcelHandler] = None,
        telegram_bot: Optional[Bot] = None,
    ):
        self.llm_processor = llm_processor
        self.woocommerce_handler = woocommerce_handler
        self.excel_handler = excel_handler
        self.telegram_bot = telegram_bot

    async def decide(self, decision_request: DecisionRequest) -> DecisionResponse:
        """
        Process the decision request and execute appropriate action.

        Args:
            decision_request: DecisionRequest object with analysis data and context.

        Returns:
            DecisionResponse with action taken and related information.

        Raises:
            DecisionEngineError: If decision or action execution fails.
        """
        logger.info(f"Received decision request for user {decision_request.user_id}")

        try:
            # Call LLM for decision-making if not provided
            if not decision_request.decision_data:
                logger.debug("No prior decision data, requesting LLM.")
                decision_response = await self._call_decision_llm(decision_request)
            else:
                logger.debug("Using provided decision data.")
                decision_response = decision_request.decision_data

            # Parse decision action
            action = decision_response.get("action")
            parameters = decision_response.get("parameters", {})
            fallback = decision_response.get("fallback_action")

            # Validate action
            if action not in ActionType._member_names_:
                logger.warning(f"Unknown action {action}, falling back.")
                action = fallback or ActionType.REQUEST_MORE_DATA.name

            action_type = ActionType[action]

            logger.info(f"Decision Engine selected action: {action_type.name}")

            # Execute action
            result = await self._execute_action(
                action_type, parameters, decision_request.user_id
            )

            return DecisionResponse(
                action=action_type, parameters=parameters, result=result, success=True
            )

        except Exception as ex:
            logger.error(f"DecisionEngine failed: {ex}", exc_info=True)
            raise DecisionEngineError(f"Decision engine failed: {ex}")

    async def _call_decision_llm(
        self, decision_request: DecisionRequest
    ) -> Dict[str, Any]:
        """
        Uses the LLMProcessor to obtain a decision from the LLM.

        Args:
            decision_request: DecisionRequest containing analysis data.

        Returns:
            Parsed LLM decision response as dict.
        """
        llm_response = await self.llm_processor.make_decision(
            decision_request.analysis_result, decision_request.task
        )
        try:
            decision_data = json.loads(llm_response.content)
        except json.JSONDecodeError:
            decision_data = {
                "action": "REQUEST_MORE_DATA",
                "reasoning": "Invalid LLM decision response",
                "parameters": {},
            }
        return decision_data

    async def _execute_action(
        self, action_type: ActionType, parameters: Dict[str, Any], user_id: int
    ) -> Any:
        """
        Executes the chosen action with provided parameters.

        Args:
            action_type: Enum action type.
            parameters: Parameters for the action.
            user_id: User ID to send notifications or perform user-scoped actions.

        Returns:
            Result of action execution, varies by action.
        """
        if action_type == ActionType.CREATE_WOOCOMMERCE_POST:
            if not self.woocommerce_handler:
                raise DecisionEngineError("WooCommerce handler not configured.")
            return await self.woocommerce_handler.create_product_draft(parameters)

        elif action_type == ActionType.UPDATE_EXCEL:
            if not self.excel_handler:
                raise DecisionEngineError("Excel handler not configured.")
            return await self.excel_handler.update_spreadsheet(parameters)

        elif action_type == ActionType.REPLY_TELEGRAM:
            if not self.telegram_bot:
                raise DecisionEngineError("Telegram bot instance is None.")
            message = parameters.get("message", "Additional information needed.")
            chat_id = parameters.get("chat_id", user_id)
            await self.telegram_bot.send_message(chat_id=chat_id, text=message)
            return {"replied": True}

        elif action_type == ActionType.CHAIN_LLM:
            # Not implemented here: trigger another LLM call with parameters
            logger.info("Chaining LLM with new parameters.")
            # You can call llm_processor again here as needed
            return {"chained": True}

        else:
            logger.warning(f"Action {action_type} not implemented.")
            return {"error": "Unsupported action."}
