"""
Telegram Responder Module

Handles generation and sending of responses back to Telegram users,
including follow-up questions, confirmations, and notifications.

Author: GG
Date: 2025-09-16
"""

import asyncio
import logging

from telegram import Bot, ChatAction
from telegram.error import TelegramError

logger = logging.getLogger(__name__)


class TelegramResponder:
    """
    Provides methods to send various types of messages and interactive keyboards
    to Telegram users.
    """

    def __init__(self, bot: Bot):
        self.bot = bot

    async def send_message(
        self, chat_id: int, text: str, reply_markup=None, parse_mode="Markdown"
    ):
        """
        Send a text message to a Telegram chat.

        Args:
            chat_id: Telegram chat ID.
            text: Message text.
            reply_markup: Optional keyboard markup.
            parse_mode: Text parse mode, e.g., 'Markdown', 'HTML'.
        """
        try:
            await self.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
            # Simulate realistic typing delay
            await asyncio.sleep(min(len(text) * 0.05, 3))
            await self.bot.send_message(
                chat_id=chat_id,
                text=text,
                reply_markup=reply_markup,
                parse_mode=parse_mode,
            )
            logger.debug(f"Sent message to chat {chat_id}")
        except TelegramError as e:
            logger.error(f"Failed sending message to chat {chat_id}: {e}")

    async def send_followup_questions(self, chat_id: int, questions: list, reason: str):
        """
        Sends follow-up questions as text with explanation.

        Args:
            chat_id: Telegram chat ID.
            questions: List of question strings.
            reason: Reason for asking follow-up questions.
        """
        if not questions:
            return await self.send_message(
                chat_id, "No follow-up questions at this time."
            )

        msg = f"🤔 I have some follow-up questions to better assist you:\n\n"
        for i, q in enumerate(questions, start=1):
            msg += f"{i}. {q}\n"
        msg += f"\n*Reason:* {reason}"

        await self.send_message(chat_id, msg)

    async def send_confirmation(
        self, chat_id: int, text: str, confirm_callback, cancel_callback
    ):
        """
        Send a confirmation message with inline buttons for confirm/cancel.

        Args:
            chat_id: Telegram chat ID.
            text: Confirmation prompt.
            confirm_callback: Callback data for confirm button.
            cancel_callback: Callback data for cancel button.
        """
        from telegram import InlineKeyboardButton, InlineKeyboardMarkup

        keyboard = InlineKeyboardMarkup(
            [
                [InlineKeyboardButton("✅ Confirm", callback_data=confirm_callback)],
                [InlineKeyboardButton("❌ Cancel", callback_data=cancel_callback)],
            ]
        )
        await self.send_message(chat_id, text, reply_markup=keyboard)
