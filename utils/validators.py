"""
Validation Utilities

Provides reusable validators for media files, user inputs, and other data structures.

Author: GG
Date: 2025-09-16
"""

import mimetypes
from typing import Optional, Set


class ValidationResult:
    def __init__(self, is_valid: bool, error: Optional[str] = None):
        self.is_valid = is_valid
        self.error = error


class MediaValidator:
    def __init__(self, allowed_mime_types: Set[str], max_file_size_mb: int):
        self.allowed_mime_types = allowed_mime_types
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024

    def validate_file(self, media_file) -> ValidationResult:
        mime_type = (
            media_file.mime_type or mimetypes.guess_type(media_file.file_name or "")[0]
        )
        if mime_type not in self.allowed_mime_types:
            return ValidationResult(False, f"Unsupported file type: {mime_type}")

        if media_file.file_size and media_file.file_size > self.max_file_size_bytes:
            return ValidationResult(
                False, f"File size exceeds limit ({self.max_file_size_bytes} bytes)"
            )

        return ValidationResult(True)


class UserInputValidator:
    def __init__(self, max_text_length: int = 4096):
        self.max_text_length = max_text_length

    def validate_text(self, text: str) -> ValidationResult:
        if not text:
            return ValidationResult(False, "Empty text input")

        if len(text) > self.max_text_length:
            return ValidationResult(
                False, f"Input text too long (max {self.max_text_length} characters)"
            )

        return ValidationResult(True)
