"""
Telegram Handler Module

This module provides comprehensive Telegram bot functionality with support for
multimedia data reception, conversation state management, and user interaction handling.
It implements advanced features including media group processing, file validation,
and seamless integration with the decision engine.

Author: Development Team
Version: 1.0.0
Date: 2025-09-16
"""

import asyncio
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Callable, Set
from dataclasses import dataclass, field
from collections import defaultdict
import hashlib
import mimetypes

from telegram import (
    Update, Bot, Message, File, PhotoSize, Video, Document, 
    InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup,
    KeyboardButton, ReplyKeyboardRemove, BotCommand
)
from telegram.ext import (
    Application, CommandHandler, MessageHandler, CallbackQueryHandler,
    ConversationHandler, ContextTypes, filters, JobQueue
)
from telegram.error import TelegramError, BadRequest, NetworkError
import aiofiles
from PIL import Image
import magic

from config.settings import get_config
from config.logging_config import get_logger, get_performance_logger, set_logging_context
from models.telegram_models import (
    TelegramUser, TelegramMessage, MediaFile, ConversationState,
    MediaGroup, UserSession, MessageContext
)
from models.data_models import ProcessingTask, TaskStatus, Priority
from core.state_manager import StateManager
from utils.exceptions import (
    TelegramHandlerError, MediaProcessingError, ValidationError,
    StorageError, ConfigurationError
)
from utils.validators import MediaValidator, UserInputValidator
from utils.file_manager import FileManager
from utils.helpers import generate_unique_id, sanitize_filename


# Configure module logger
logger = get_logger(__name__)
perf_logger = get_performance_logger(__name__)


@dataclass
class HandlerConfig:
    """
    Configuration class for Telegram handler settings.
    
    This class encapsulates all configuration parameters specific to
    the Telegram handler functionality including media processing,
    user interaction, and conversation management settings.
    """
    
    # File handling settings
    max_file_size_mb: int = field(default=50)
    allowed_mime_types: Set[str] = field(default_factory=lambda: {
        'image/jpeg', 'image/png', 'image/gif', 'image/webp',
        'video/mp4', 'video/quicktime', 'video/x-msvideo',
        'application/pdf', 'text/plain', 'application/msword',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
    })
    
    # Media processing settings
    image_compression_quality: int = field(default=85)
    thumbnail_size: tuple = field(default=(150, 150))
    video_thumbnail_time: float = field(default=2.0)
    
    # Conversation settings
    session_timeout_minutes: int = field(default=30)
    max_conversation_depth: int = field(default=50)
    auto_cleanup_hours: int = field(default=24)
    
    # Rate limiting
    max_messages_per_minute: int = field(default=30)
    max_files_per_request: int = field(default=10)
    cooldown_seconds: int = field(default=1)
    
    # UI settings
    enable_typing_indicator: bool = field(default=True)
    typing_delay_seconds: float = field(default=0.5)
    enable_progress_updates: bool = field(default=True)
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.max_file_size_mb <= 0 or self.max_file_size_mb > 100:
            raise ValueError("File size must be between 1-100 MB")
        
        if not self.allowed_mime_types:
            raise ValueError("At least one MIME type must be allowed")
        
        logger.debug("Handler configuration validated successfully")


class MediaGroupCollector:
    """
    Collects and manages Telegram media groups (albums).
    
    This class handles the complexities of media group processing where
    Telegram sends each media file as a separate message with a shared
    media_group_id, requiring collection and batch processing.
    """
    
    def __init__(self, timeout_seconds: int = 5):
        """
        Initialize media group collector.
        
        Args:
            timeout_seconds: Time to wait for all media in group
        """
        self.timeout_seconds = timeout_seconds
        self.pending_groups: Dict[str, Dict[str, Any]] = {}
        self.group_timers: Dict[str, asyncio.Task] = {}
        
        logger.debug(f"Media group collector initialized with {timeout_seconds}s timeout")
    
    async def add_media(self, message: Message, callback: Callable) -> bool:
        """
        Add media message to collection.
        
        Args:
            message: Telegram message containing media
            callback: Callback function to call when group is complete
            
        Returns:
            True if this completes a group, False if waiting for more
        """
        media_group_id = message.media_group_id
        
        if not media_group_id:
            # Single media file, process immediately
            await callback([message])
            return True
        
        try:
            # Initialize group if not exists
            if media_group_id not in self.pending_groups:
                self.pending_groups[media_group_id] = {
                    'messages': [],
                    'callback': callback,
                    'start_time': datetime.now(timezone.utc)
                }
                
                # Set timer for group completion
                self.group_timers[media_group_id] = asyncio.create_task(
                    self._wait_for_group_completion(media_group_id)
                )
                
                logger.debug(f"Started new media group collection: {media_group_id}")
            
            # Add message to group
            self.pending_groups[media_group_id]['messages'].append(message)
            
            logger.debug(
                f"Added media to group {media_group_id}, "
                f"total: {len(self.pending_groups[media_group_id]['messages'])}"
            )
            
            return False  # Still waiting for more media
            
        except Exception as e:
            logger.error(f"Error processing media group {media_group_id}: {e}")
            # Process what we have
            if media_group_id in self.pending_groups:
                await callback(self.pending_groups[media_group_id]['messages'])
                self._cleanup_group(media_group_id)
            raise MediaProcessingError(f"Media group processing failed: {e}")
    
    async def _wait_for_group_completion(self, media_group_id: str) -> None:
        """
        Wait for media group completion and process when timeout reached.
        
        Args:
            media_group_id: ID of the media group
        """
        try:
            # Wait for timeout
            await asyncio.sleep(self.timeout_seconds)
            
            if media_group_id in self.pending_groups:
                group_data = self.pending_groups[media_group_id]
                messages = group_data['messages']
                callback = group_data['callback']
                
                logger.info(
                    f"Processing media group {media_group_id} with {len(messages)} items"
                )
                
                # Process the collected media
                await callback(messages)
                
                # Cleanup
                self._cleanup_group(media_group_id)
        
        except asyncio.CancelledError:
            logger.debug(f"Media group timer cancelled for {media_group_id}")
        except Exception as e:
            logger.error(f"Error in media group timer for {media_group_id}: {e}")
            self._cleanup_group(media_group_id)
    
    def _cleanup_group(self, media_group_id: str) -> None:
        """
        Clean up completed or failed media group.
        
        Args:
            media_group_id: ID of the group to clean up
        """
        try:
            # Cancel timer if still running
            if media_group_id in self.group_timers:
                timer = self.group_timers[media_group_id]
                if not timer.done():
                    timer.cancel()
                del self.group_timers[media_group_id]
            
            # Remove group data
            if media_group_id in self.pending_groups:
                del self.pending_groups[media_group_id]
            
            logger.debug(f"Cleaned up media group: {media_group_id}")
            
        except Exception as e:
            logger.error(f"Error cleaning up media group {media_group_id}: {e}")
    
    def get_pending_groups_count(self) -> int:
        """Get count of pending media groups."""
        return len(self.pending_groups)


class TelegramHandler:
    """
    Main Telegram bot handler class.
    
    This class provides comprehensive Telegram bot functionality including
    multimedia data processing, user interaction management, conversation
    handling, and integration with the decision engine system.
    """
    
    def __init__(self, config: HandlerConfig, state_manager: StateManager):
        """
        Initialize Telegram handler with configuration and dependencies.
        
        Args:
            config: Handler configuration
            state_manager: State management instance
            
        Raises:
            ConfigurationError: If configuration is invalid
            InitializationError: If initialization fails
        """
        try:
            self.config = config
            self.state_manager = state_manager
            self.app_config = get_config()
            
            # Initialize components
            self.bot: Optional[Bot] = None
            self.application: Optional[Application] = None
            self.media_validator = MediaValidator(config.allowed_mime_types, config.max_file_size_mb)
            self.input_validator = UserInputValidator()
            self.file_manager = FileManager()
            self.media_collector = MediaGroupCollector()
            
            # User session management
            self.active_sessions: Dict[int, UserSession] = {}
            self.rate_limiters: Dict[int, List[datetime]] = defaultdict(list)
            
            # Processing queue
            self.processing_queue: asyncio.Queue = asyncio.Queue()
            self.processing_workers: List[asyncio.Task] = []
            
            # Conversation states
            self.COLLECTING_DATA = 1
            self.WAITING_CONFIRMATION = 2
            self.PROCESSING_DECISION = 3
            
            logger.info("Telegram handler initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Telegram handler: {e}")
            raise ConfigurationError(f"Handler initialization failed: {e}")
    
    async def initialize(self) -> None:
        """
        Initialize the Telegram bot application and set up handlers.
        
        This method sets up the bot instance, registers all message handlers,
        and prepares the application for receiving updates.
        
        Raises:
            TelegramHandlerError: If initialization fails
        """
        try:
            # Initialize bot application
            bot_token = self.app_config.telegram.bot_token
            if not bot_token:
                raise ConfigurationError("Telegram bot token not configured")
            
            self.application = Application.builder().token(bot_token).build()
            self.bot = self.application.bot
            
            # Setup command handlers
            await self._setup_command_handlers()
            
            # Setup message handlers
            await self._setup_message_handlers()
            
            # Setup conversation handlers
            await self._setup_conversation_handlers()
            
            # Setup callback query handlers
            await self._setup_callback_handlers()
            
            # Setup error handlers
            await self._setup_error_handlers()
            
            # Set bot commands
            await self._setup_bot_commands()
            
            # Start processing workers
            await self._start_processing_workers()
            
            logger.info("Telegram bot application initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Telegram application: {e}")
            raise TelegramHandlerError(f"Application initialization failed: {e}")
    
    async def _setup_command_handlers(self) -> None:
        """Setup command handlers for bot commands."""
        command_handlers = [
            CommandHandler("start", self.handle_start_command),
            CommandHandler("help", self.handle_help_command),
            CommandHandler("status", self.handle_status_command),
            CommandHandler("cancel", self.handle_cancel_command),
            CommandHandler("clear", self.handle_clear_command),
            CommandHandler("settings", self.handle_settings_command),
        ]
        
        for handler in command_handlers:
            self.application.add_handler(handler)
        
        logger.debug("Command handlers registered")
    
    async def _setup_message_handlers(self) -> None:
        """Setup message handlers for different media types."""
        # Media handlers
        self.application.add_handler(
            MessageHandler(filters.PHOTO, self.handle_photo_message)
        )
        self.application.add_handler(
            MessageHandler(filters.VIDEO, self.handle_video_message)
        )
        self.application.add_handler(
            MessageHandler(filters.DOCUMENT, self.handle_document_message)
        )
        
        # Text message handler
        self.application.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_text_message)
        )
        
        # Voice and audio handlers
        self.application.add_handler(
            MessageHandler(filters.VOICE | filters.AUDIO, self.handle_audio_message)
        )
        
        logger.debug("Message handlers registered")
    
    async def _setup_conversation_handlers(self) -> None:
        """Setup conversation handlers for multi-step interactions."""
        conversation_handler = ConversationHandler(
            entry_points=[CommandHandler("collect", self.start_data_collection)],
            states={
                self.COLLECTING_DATA: [
                    MessageHandler(
                        filters.PHOTO | filters.VIDEO | filters.DOCUMENT | filters.TEXT,
                        self.handle_data_collection
                    ),
                    CallbackQueryHandler(self.handle_collection_callback, pattern="^collection_")
                ],
                self.WAITING_CONFIRMATION: [
                    CallbackQueryHandler(self.handle_confirmation_callback, pattern="^confirm_")
                ],
                self.PROCESSING_DECISION: [
                    CallbackQueryHandler(self.handle_decision_callback, pattern="^decision_")
                ]
            },
            fallbacks=[
                CommandHandler("cancel", self.handle_cancel_command),
                CallbackQueryHandler(self.handle_cancel_callback, pattern="^cancel")
            ],
            allow_reentry=True,
            conversation_timeout=self.config.session_timeout_minutes * 60
        )
        
        self.application.add_handler(conversation_handler)
        logger.debug("Conversation handlers registered")
    
    async def _setup_callback_handlers(self) -> None:
        """Setup callback query handlers for inline keyboards."""
        self.application.add_handler(
            CallbackQueryHandler(self.handle_generic_callback)
        )
        logger.debug("Callback handlers registered")
    
    async def _setup_error_handlers(self) -> None:
        """Setup error handlers for graceful error handling."""
        self.application.add_error_handler(self.handle_telegram_error)
        logger.debug("Error handlers registered")
    
    async def _setup_bot_commands(self) -> None:
        """Setup bot command menu."""
        commands = [
            BotCommand("start", "Start interaction with the bot"),
            BotCommand("collect", "Start data collection process"),
            BotCommand("status", "Check current processing status"),
            BotCommand("help", "Get help information"),
            BotCommand("cancel", "Cancel current operation"),
            BotCommand("clear", "Clear session data"),
            BotCommand("settings", "View bot settings"),
        ]
        
        try:
            await self.bot.set_my_commands(commands)
            logger.info("Bot commands set successfully")
        except Exception as e:
            logger.warning(f"Failed to set bot commands: {e}")
    
    async def _start_processing_workers(self, worker_count: int = 3) -> None:
        """
        Start background workers for processing tasks.
        
        Args:
            worker_count: Number of worker tasks to start
        """
        for i in range(worker_count):
            worker = asyncio.create_task(self._processing_worker(f"worker-{i}"))
            self.processing_workers.append(worker)
        
        logger.info(f"Started {worker_count} processing workers")
    
    async def _processing_worker(self, worker_name: str) -> None:
        """
        Background worker for processing tasks.
        
        Args:
            worker_name: Name identifier for the worker
        """
        logger.info(f"Processing worker {worker_name} started")
        
        while True:
            try:
                # Get task from queue
                task = await self.processing_queue.get()
                
                if task is None:  # Shutdown signal
                    break
                
                # Set logging context
                set_logging_context(
                    request_id=task.id,
                    user_id=task.user_id
                )
                
                logger.info(f"Worker {worker_name} processing task {task.id}")
                
                # Process the task
                await self._process_task(task)
                
                # Mark task as done
                self.processing_queue.task_done()
                
            except asyncio.CancelledError:
                logger.info(f"Worker {worker_name} cancelled")
                break
            except Exception as e:
                logger.error(f"Worker {worker_name} error: {e}")
                # Continue processing other tasks
    
    @perf_logger.log_function_performance("process_task")
    async def _process_task(self, task: ProcessingTask) -> None:
        """
        Process a single task.
        
        Args:
            task: Processing task to execute
        """
        try:
            # Update task status
            task.status = TaskStatus.PROCESSING
            task.started_at = datetime.now(timezone.utc)
            
            # Save initial state
            await self.state_manager.save_task_state(task)
            
            # Process based on task type
            if task.task_type == "media_processing":
                await self._process_media_task(task)
            elif task.task_type == "text_processing":
                await self._process_text_task(task)
            elif task.task_type == "decision_making":
                await self._process_decision_task(task)
            else:
                raise ValueError(f"Unknown task type: {task.task_type}")
            
            # Mark as completed
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now(timezone.utc)
            
            logger.info(f"Task {task.id} completed successfully")
            
        except Exception as e:
            # Mark as failed
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
            task.failed_at = datetime.now(timezone.utc)
            
            logger.error(f"Task {task.id} failed: {e}")
            
            # Notify user of failure
            await self._notify_task_failure(task, e)
        
        finally:
            # Save final state
            await self.state_manager.save_task_state(task)
    
    async def _process_media_task(self, task: ProcessingTask) -> None:
        """Process media-related task."""
        # Implementation will be connected to LLM processor
        logger.info(f"Processing media task {task.id}")
        # TODO: Connect to LLM processor for media analysis
    
    async def _process_text_task(self, task: ProcessingTask) -> None:
        """Process text-related task."""
        # Implementation will be connected to LLM processor
        logger.info(f"Processing text task {task.id}")
        # TODO: Connect to LLM processor for text analysis
    
    async def _process_decision_task(self, task: ProcessingTask) -> None:
        """Process decision-making task."""
        # Implementation will be connected to decision engine
        logger.info(f"Processing decision task {task.id}")
        # TODO: Connect to decision engine
    
    async def _notify_task_failure(self, task: ProcessingTask, error: Exception) -> None:
        """
        Notify user of task failure.
        
        Args:
            task: Failed task
            error: Error that occurred
        """
        try:
            error_message = (
                f"❌ Task failed: {task.task_type}\n"
                f"Error: {str(error)}\n"
                f"Task ID: {task.id}\n"
                f"Please try again or contact support."
            )
            
            await self.bot.send_message(
                chat_id=task.user_id,
                text=error_message
            )
            
        except Exception as e:
            logger.error(f"Failed to notify user of task failure: {e}")
    
    # Message handler methods
    
    async def handle_start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Handle /start command.
        
        Args:
            update: Telegram update object
            context: Bot context
        """
        user = update.effective_user
        chat_id = update.effective_chat.id
        
        # Set logging context
        set_logging_context(user_id=user.id)
        
        logger.info(f"User {user.id} ({user.username}) started the bot")
        
        try:
            # Create or update user session
            session = await self._get_or_create_session(user.id, user)
            
            welcome_message = (
                f"🤖 Welcome to the LLM Decision Engine, {user.first_name}!\n\n"
                f"I can help you process and analyze various types of data including:\n"
                f"📷 Images and photos\n"
                f"🎥 Videos\n"
                f"📄 Documents and files\n"
                f"💬 Text messages\n\n"
                f"Use /collect to start data collection or send me any media directly!\n"
                f"Use /help for more information."
            )
            
            # Create inline keyboard
            keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton("🚀 Start Collection", callback_data="start_collection")],
                [InlineKeyboardButton("ℹ️ Help", callback_data="show_help")],
                [InlineKeyboardButton("⚙️ Settings", callback_data="show_settings")]
            ])
            
            await update.message.reply_text(
                welcome_message,
                reply_markup=keyboard
            )
            
        except Exception as e:
            logger.error(f"Error in start command handler: {e}")
            await update.message.reply_text(
                "❌ An error occurred while starting. Please try again."
            )
    
    async def handle_help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /help command."""
        help_text = (
            "🤖 **LLM Decision Engine Bot Help**\n\n"
            "**Commands:**\n"
            "• /start - Start the bot\n"
            "• /collect - Begin data collection\n"
            "• /status - Check processing status\n"
            "• /cancel - Cancel current operation\n"
            "• /clear - Clear session data\n"
            "• /settings - View settings\n\n"
            "**Supported Media:**\n"
            "📷 Images: JPG, PNG, GIF, WebP\n"
            "🎥 Videos: MP4, MOV, AVI\n"
            "📄 Documents: PDF, TXT, DOC, DOCX\n\n"
            "**Features:**\n"
            "• Multi-file processing\n"
            "• Album/media group support\n"
            "• Intelligent decision making\n"
            "• WooCommerce integration\n"
            "• Excel database updates\n"
            "• Smart responses\n\n"
            "Simply send me your files or use /collect to get started!"
        )
        
        await update.message.reply_text(help_text, parse_mode='Markdown')
    
    async def handle_status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /status command."""
        user_id = update.effective_user.id
        session = self.active_sessions.get(user_id)
        
        if not session:
            await update.message.reply_text("No active session found.")
            return
        
        status_text = (
            f"📊 **Session Status**\n\n"
            f"User ID: {user_id}\n"
            f"Session Started: {session.created_at.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Files Collected: {len(session.collected_media)}\n"
            f"Messages: {len(session.conversation_history)}\n"
            f"Current State: {session.current_state}\n"
            f"Pending Groups: {self.media_collector.get_pending_groups_count()}"
        )
        
        await update.message.reply_text(status_text, parse_mode='Markdown')
    
    async def handle_cancel_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /cancel command."""
        user_id = update.effective_user.id
        
        # Clean up user session
        if user_id in self.active_sessions:
            del self.active_sessions[user_id]
        
        await update.message.reply_text(
            "❌ Operation cancelled. Session cleared.\nUse /start to begin again."
        )
        
        return ConversationHandler.END
    
    async def handle_clear_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /clear command."""
        user_id = update.effective_user.id
        
        # Clear session data
        if user_id in self.active_sessions:
            self.active_sessions[user_id].collected_media.clear()
            self.active_sessions[user_id].conversation_history.clear()
            
            await update.message.reply_text("✅ Session data cleared.")
        else:
            await update.message.reply_text("No active session to clear.")
    
    async def handle_settings_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /settings command."""
        settings_text = (
            f"⚙️ **Bot Settings**\n\n"
            f"Max File Size: {self.config.max_file_size_mb} MB\n"
            f"Session Timeout: {self.config.session_timeout_minutes} minutes\n"
            f"Max Files per Request: {self.config.max_files_per_request}\n"
            f"Rate Limit: {self.config.max_messages_per_minute} msg/min\n"
            f"Typing Indicator: {'✅' if self.config.enable_typing_indicator else '❌'}\n"
            f"Progress Updates: {'✅' if self.config.enable_progress_updates else '❌'}"
        )
        
        await update.message.reply_text(settings_text, parse_mode='Markdown')
    
    # Media handler methods
    
    async def handle_photo_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle photo messages."""
        await self._handle_media_message(update, context, "photo")
    
    async def handle_video_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle video messages."""
        await self._handle_media_message(update, context, "video")
    
    async def handle_document_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle document messages."""
        await self._handle_media_message(update, context, "document")
    
    async def handle_audio_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle audio/voice messages."""
        await update.message.reply_text(
            "🎵 Audio processing is not yet supported. Please send images, videos, or documents."
        )
    
    async def _handle_media_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE, media_type: str) -> None:
        """
        Handle media messages with comprehensive processing.
        
        Args:
            update: Telegram update
            context: Bot context
            media_type: Type of media (photo, video, document)
        """
        user_id = update.effective_user.id
        message = update.message
        
        # Set logging context
        set_logging_context(user_id=user_id)
        
        logger.info(f"Received {media_type} from user {user_id}")
        
        try:
            # Check rate limiting
            if not await self._check_rate_limit(user_id):
                await message.reply_text(
                    "⏳ Please slow down. You're sending messages too quickly."
                )
                return
            
            # Get or create session
            session = await self._get_or_create_session(user_id, update.effective_user)
            
            # Show typing indicator
            if self.config.enable_typing_indicator:
                await context.bot.send_chat_action(
                    chat_id=update.effective_chat.id,
                    action="upload_document"
                )
            
            # Handle media group collection
            await self.media_collector.add_media(
                message,
                lambda messages: self._process_media_group(messages, session)
            )
            
        except Exception as e:
            logger.error(f"Error handling {media_type} message: {e}")
            await message.reply_text(
                f"❌ Error processing {media_type}: {str(e)}\n"
                "Please try again or contact support."
            )
    
    async def _process_media_group(self, messages: List[Message], session: UserSession) -> None:
        """
        Process collected media messages.
        
        Args:
            messages: List of media messages
            session: User session
        """
        try:
            logger.info(f"Processing media group with {len(messages)} items for user {session.user_id}")
            
            media_files = []
            
            for message in messages:
                # Extract media file information
                media_file = await self._extract_media_info(message)
                
                if media_file:
                    # Validate media file
                    validation_result = await self.media_validator.validate_file(media_file)
                    
                    if not validation_result.is_valid:
                        logger.warning(f"Invalid media file: {validation_result.error}")
                        await self.bot.send_message(
                            chat_id=session.user_id,
                            text=f"❌ Invalid file: {validation_result.error}"
                        )
                        continue
                    
                    # Download and store file
                    downloaded_file = await self._download_media_file(media_file)
                    media_files.append(downloaded_file)
            
            if not media_files:
                await self.bot.send_message(
                    chat_id=session.user_id,
                    text="❌ No valid files found to process."
                )
                return
            
            # Add to session
            session.collected_media.extend(media_files)
            
            # Create processing task
            task = ProcessingTask(
                id=generate_unique_id(),
                user_id=session.user_id,
                task_type="media_processing",
                data={"media_files": [file.to_dict() for file in media_files]},
                priority=Priority.NORMAL,
                created_at=datetime.now(timezone.utc)
            )
            
            # Queue for processing
            await self.processing_queue.put(task)
            
            # Notify user
            await self.bot.send_message(
                chat_id=session.user_id,
                text=(
                    f"✅ Received {len(media_files)} files\n"
                    f"📊 Total files in session: {len(session.collected_media)}\n"
                    f"🔄 Processing started..."
                )
            )
            
        except Exception as e:
            logger.error(f"Error processing media group: {e}")
            await self.bot.send_message(
                chat_id=session.user_id,
                text=f"❌ Error processing media: {str(e)}"
            )
    
    async def _extract_media_info(self, message: Message) -> Optional[MediaFile]:
        """
        Extract media file information from message.
        
        Args:
            message: Telegram message
            
        Returns:
            MediaFile object or None
        """
        try:
            if message.photo:
                # Get highest resolution photo
                photo = message.photo[-1]
                return MediaFile(
                    file_id=photo.file_id,
                    file_unique_id=photo.file_unique_id,
                    file_type="photo",
                    file_size=photo.file_size,
                    width=photo.width,
                    height=photo.height,
                    caption=message.caption,
                    mime_type="image/jpeg"
                )
            
            elif message.video:
                video = message.video
                return MediaFile(
                    file_id=video.file_id,
                    file_unique_id=video.file_unique_id,
                    file_type="video",
                    file_size=video.file_size,
                    width=video.width,
                    height=video.height,
                    duration=video.duration,
                    caption=message.caption,
                    mime_type=video.mime_type or "video/mp4"
                )
            
            elif message.document:
                document = message.document
                return MediaFile(
                    file_id=document.file_id,
                    file_unique_id=document.file_unique_id,
                    file_type="document",
                    file_size=document.file_size,
                    file_name=document.file_name,
                    caption=message.caption,
                    mime_type=document.mime_type or "application/octet-stream"
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error extracting media info: {e}")
            return None
    
    async def _download_media_file(self, media_file: MediaFile) -> MediaFile:
        """
        Download media file from Telegram servers.
        
        Args:
            media_file: Media file information
            
        Returns:
            Updated MediaFile with local path
            
        Raises:
            StorageError: If download fails
        """
        try:
            # Get file from Telegram
            file = await self.bot.get_file(media_file.file_id)
            
            # Generate local file path
            file_extension = self._get_file_extension(media_file)
            safe_filename = sanitize_filename(
                media_file.file_name or f"{media_file.file_unique_id}{file_extension}"
            )
            
            local_path = Path(self.app_config.downloads_dir) / safe_filename
            
            # Download file
            await file.download_to_drive(str(local_path))
            
            # Update media file with local path
            media_file.local_path = str(local_path)
            media_file.file_hash = await self._calculate_file_hash(local_path)
            media_file.downloaded_at = datetime.now(timezone.utc)
            
            logger.info(f"Downloaded file: {safe_filename} ({media_file.file_size} bytes)")
            
            return media_file
            
        except Exception as e:
            logger.error(f"Error downloading file {media_file.file_id}: {e}")
            raise StorageError(f"Failed to download file: {e}")
    
    def _get_file_extension(self, media_file: MediaFile) -> str:
        """Get appropriate file extension for media file."""
        if media_file.file_name:
            return Path(media_file.file_name).suffix
        
        # Determine extension from MIME type
        extension = mimetypes.guess_extension(media_file.mime_type)
        return extension or ".bin"
    
    async def _calculate_file_hash(self, file_path: Path) -> str:
        """
        Calculate SHA-256 hash of file.
        
        Args:
            file_path: Path to file
            
        Returns:
            Hexadecimal hash string
        """
        hash_sha256 = hashlib.sha256()
        
        async with aiofiles.open(file_path, 'rb') as f:
            async for chunk in f:
                hash_sha256.update(chunk)
        
        return hash_sha256.hexdigest()
    
    async def handle_text_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle text messages."""
        user_id = update.effective_user.id
        message_text = update.message.text
        
        # Set logging context
        set_logging_context(user_id=user_id)
        
        logger.info(f"Received text message from user {user_id}")
        
        try:
            # Validate text input
            validation_result = self.input_validator.validate_text(message_text)
            
            if not validation_result.is_valid:
                await update.message.reply_text(
                    f"❌ Invalid input: {validation_result.error}"
                )
                return
            
            # Get or create session
            session = await self._get_or_create_session(user_id, update.effective_user)
            
            # Add to conversation history
            session.add_message(
                TelegramMessage(
                    message_id=update.message.message_id,
                    from_user=TelegramUser.from_telegram_user(update.effective_user),
                    text=message_text,
                    timestamp=datetime.now(timezone.utc)
                )
            )
            
            # Create text processing task
            task = ProcessingTask(
                id=generate_unique_id(),
                user_id=user_id,
                task_type="text_processing",
                data={"text": message_text},
                priority=Priority.NORMAL,
                created_at=datetime.now(timezone.utc)
            )
            
            # Queue for processing
            await self.processing_queue.put(task)
            
            # Show typing indicator
            if self.config.enable_typing_indicator:
                await context.bot.send_chat_action(
                    chat_id=update.effective_chat.id,
                    action="typing"
                )
                await asyncio.sleep(self.config.typing_delay_seconds)
            
            await update.message.reply_text(
                "✅ Message received and queued for processing...",
                reply_to_message_id=update.message.message_id
            )
            
        except Exception as e:
            logger.error(f"Error handling text message: {e}")
            await update.message.reply_text(
                f"❌ Error processing message: {str(e)}"
            )
    
    # Conversation handler methods
    
    async def start_data_collection(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Start data collection conversation."""
        user_id = update.effective_user.id
        
        # Set logging context
        set_logging_context(user_id=user_id)
        
        logger.info(f"Starting data collection for user {user_id}")
        
        # Get or create session
        session = await self._get_or_create_session(user_id, update.effective_user)
        session.current_state = ConversationState.COLLECTING
        
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("📸 Send Photos", callback_data="collection_photos")],
            [InlineKeyboardButton("🎥 Send Videos", callback_data="collection_videos")],
            [InlineKeyboardButton("📄 Send Documents", callback_data="collection_documents")],
            [InlineKeyboardButton("✅ Finish Collection", callback_data="collection_finish")],
            [InlineKeyboardButton("❌ Cancel", callback_data="cancel")]
        ])
        
        await update.message.reply_text(
            "🚀 **Data Collection Started**\n\n"
            "Please send me your data:\n"
            "• Photos, images, screenshots\n"
            "• Videos, screen recordings\n"
            "• Documents, PDFs, text files\n"
            "• Text messages\n\n"
            "You can send multiple files at once (albums are supported).\n"
            "Click 'Finish Collection' when done.",
            reply_markup=keyboard,
            parse_mode='Markdown'
        )
        
        return self.COLLECTING_DATA
    
    async def handle_data_collection(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Handle data collection during conversation."""
        # Process the data using existing handlers
        if update.message.photo:
            await self.handle_photo_message(update, context)
        elif update.message.video:
            await self.handle_video_message(update, context)
        elif update.message.document:
            await self.handle_document_message(update, context)
        elif update.message.text:
            await self.handle_text_message(update, context)
        
        return self.COLLECTING_DATA
    
    # Callback handler methods
    
    async def handle_collection_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Handle collection-related callbacks."""
        query = update.callback_query
        await query.answer()
        
        if query.data == "collection_finish":
            return await self._finish_collection(update, context)
        elif query.data == "collection_photos":
            await query.edit_message_text("📸 Please send your photos now...")
        elif query.data == "collection_videos":
            await query.edit_message_text("🎥 Please send your videos now...")
        elif query.data == "collection_documents":
            await query.edit_message_text("📄 Please send your documents now...")
        
        return self.COLLECTING_DATA
    
    async def _finish_collection(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Finish data collection and proceed to confirmation."""
        query = update.callback_query
        user_id = update.effective_user.id
        session = self.active_sessions.get(user_id)
        
        if not session or not session.collected_media:
            await query.edit_message_text(
                "❌ No data collected. Please send some files first."
            )
            return self.COLLECTING_DATA
        
        # Show collection summary
        summary = self._generate_collection_summary(session)
        
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("✅ Process Data", callback_data="confirm_process")],
            [InlineKeyboardButton("📝 Add More", callback_data="confirm_add_more")],
            [InlineKeyboardButton("❌ Cancel", callback_data="cancel")]
        ])
        
        await query.edit_message_text(
            f"📋 **Collection Summary**\n\n{summary}\n\n"
            "Ready to process this data?",
            reply_markup=keyboard,
            parse_mode='Markdown'
        )
        
        return self.WAITING_CONFIRMATION
    
    def _generate_collection_summary(self, session: UserSession) -> str:
        """Generate summary of collected data."""
        media_count = len(session.collected_media)
        message_count = len(session.conversation_history)
        
        # Count by type
        type_counts = defaultdict(int)
        total_size = 0
        
        for media in session.collected_media:
            type_counts[media.file_type] += 1
            total_size += media.file_size or 0
        
        summary = f"📊 **Files:** {media_count} files\n"
        summary += f"💬 **Messages:** {message_count} messages\n"
        summary += f"💾 **Size:** {total_size / 1024 / 1024:.1f} MB\n\n"
        
        if type_counts:
            summary += "**File Types:**\n"
            for file_type, count in type_counts.items():
                emoji = {"photo": "📸", "video": "🎥", "document": "📄"}.get(file_type, "📄")
                summary += f"{emoji} {file_type.title()}: {count}\n"
        
        return summary
    
    async def handle_confirmation_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Handle confirmation callbacks."""
        query = update.callback_query
        await query.answer()
        
        if query.data == "confirm_process":
            return await self._start_processing(update, context)
        elif query.data == "confirm_add_more":
            await query.edit_message_text("📸 Please send additional files...")
            return self.COLLECTING_DATA
        
        return self.WAITING_CONFIRMATION
    
    async def _start_processing(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Start data processing."""
        query = update.callback_query
        user_id = update.effective_user.id
        session = self.active_sessions.get(user_id)
        
        if not session:
            await query.edit_message_text("❌ Session expired. Please start over.")
            return ConversationHandler.END
        
        # Create comprehensive processing task
        task = ProcessingTask(
            id=generate_unique_id(),
            user_id=user_id,
            task_type="decision_making",
            data={
                "media_files": [file.to_dict() for file in session.collected_media],
                "messages": [msg.to_dict() for msg in session.conversation_history],
                "session_data": session.to_dict()
            },
            priority=Priority.HIGH,
            created_at=datetime.now(timezone.utc)
        )
        
        # Queue for processing
        await self.processing_queue.put(task)
        
        await query.edit_message_text(
            "🔄 **Processing Started**\n\n"
            "Your data is being analyzed and processed...\n"
            "I'll notify you with the results shortly.\n\n"
            f"Task ID: `{task.id}`",
            parse_mode='Markdown'
        )
        
        return ConversationHandler.END
    
    async def handle_decision_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Handle decision-related callbacks."""
        # This will be connected to the decision engine
        query = update.callback_query
        await query.answer("Decision processing...")
        return ConversationHandler.END
    
    async def handle_cancel_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Handle cancel callback."""
        query = update.callback_query
        await query.answer()
        await query.edit_message_text("❌ Operation cancelled.")
        return ConversationHandler.END
    
    async def handle_generic_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle generic callbacks."""
        query = update.callback_query
        await query.answer()
        
        if query.data == "start_collection":
            await self.start_data_collection(update, context)
        elif query.data == "show_help":
            await self.handle_help_command(update, context)
        elif query.data == "show_settings":
            await self.handle_settings_command(update, context)
    
    async def handle_telegram_error(self, update: Optional[Update], context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle Telegram API errors."""
        error = context.error
        
        logger.error(f"Telegram error: {error}")
        
        if update and update.effective_message:
            try:
                await update.effective_message.reply_text(
                    "❌ An error occurred. Please try again later."
                )
            except Exception as e:
                logger.error(f"Failed to send error message: {e}")
    
    # Utility methods
    
    async def _get_or_create_session(self, user_id: int, telegram_user) -> UserSession:
        """Get existing session or create new one."""
        if user_id not in self.active_sessions:
            user = TelegramUser.from_telegram_user(telegram_user)
            session = UserSession(
                user_id=user_id,
                user=user,
                created_at=datetime.now(timezone.utc)
            )
            self.active_sessions[user_id] = session
            logger.info(f"Created new session for user {user_id}")
        
        return self.active_sessions[user_id]
    
    async def _check_rate_limit(self, user_id: int) -> bool:
        """
        Check if user is within rate limits.
        
        Args:
            user_id: User ID to check
            
        Returns:
            True if within limits, False otherwise
        """
        now = datetime.now(timezone.utc)
        user_messages = self.rate_limiters[user_id]
        
        # Remove old messages
        cutoff = now.timestamp() - 60  # 1 minute window
        self.rate_limiters[user_id] = [
            msg_time for msg_time in user_messages 
            if msg_time.timestamp() > cutoff
        ]
        
        # Check limit
        if len(self.rate_limiters[user_id]) >= self.config.max_messages_per_minute:
            return False
        
        # Add current message
        self.rate_limiters[user_id].append(now)
        return True
    
    async def start_polling(self) -> None:
        """Start the bot with polling."""
        try:
            logger.info("Starting Telegram bot with polling...")
            await self.application.initialize()
            await self.application.start()
            await self.application.updater.start_polling()
            
            logger.info("Bot started successfully")
            
            # Keep running
            await self.application.updater.idle()
            
        except Exception as e:
            logger.error(f"Error starting bot: {e}")
            raise
        finally:
            await self.stop()
    
    async def stop(self) -> None:
        """Stop the bot and cleanup resources."""
        logger.info("Stopping Telegram bot...")
        
        try:
            # Stop processing workers
            for _ in self.processing_workers:
                await self.processing_queue.put(None)  # Shutdown signal
            
            # Wait for workers to finish
            if self.processing_workers:
                await asyncio.gather(*self.processing_workers, return_exceptions=True)
            
            # Stop application
            if self.application:
                await self.application.stop()
                await self.application.shutdown()
            
            logger.info("Bot stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping bot: {e}")


# Export main class
__all__ = ['TelegramHandler', 'HandlerConfig']