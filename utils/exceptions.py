"""
Custom Exception Classes

Defines application-specific exceptions for better error handling.

Author: Development Team
Date: 2025-09-16
"""


class TelegramHandlerError(Exception):
    pass


class MediaProcessingError(Exception):
    pass


class ValidationError(Exception):
    pass


class StorageError(Exception):
    pass


class ConfigurationError(Exception):
    pass


class LLMProcessingError(Exception):
    pass


class TokenLimitExceededError(LLMProcessingError):
    pass


class ProviderUnavailableError(LLMProcessingError):
    pass


class InvalidRequestError(LLMProcessingError):
    pass


class RateLimitExceededError(LLMProcessingError):
    pass


class DecisionEngineError(Exception):
    pass


class ApplicationError(Exception):
    pass
