"""
Base Handler Module

Defines abstract base classes and common interfaces for all handlers
(e.g., TelegramHandler, DecisionEngine, LLMProcessor).

Author: Development Team
Date: 2025-09-16
"""

import logging
from abc import ABC, abstractmethod


class BaseHandler(ABC):
    """
    Abstract Base Class for all application handlers.
    Provides common lifecycle and utility methods.
    """

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    async def initialize(self):
        """Initialize the handler (e.g., connect resources)."""
        pass

    @abstractmethod
    async def shutdown(self):
        """Cleanly shutdown the handler and release resources."""
        pass

    def log_debug(self, message: str, **kwargs):
        """Convenience debug logging method."""
        self.logger.debug(message, extra=kwargs)

    def log_info(self, message: str, **kwargs):
        """Convenience info logging method."""
        self.logger.info(message, extra=kwargs)

    def log_warning(self, message: str, **kwargs):
        """Convenience warning logging method."""
        self.logger.warning(message, extra=kwargs)

    def log_error(self, message: str, **kwargs):
        """Convenience error logging method."""
        self.logger.error(message, extra=kwargs)
