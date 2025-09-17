"""
File Manager Utilities

Includes async functions for downloading, storing, and managing files locally.
Handles safe filename generation and file integrity checks.

Author: GG
Date: 2025-09-16
"""

import hashlib
import os
from pathlib import Path
from typing import Optional

import aiofiles


class FileManager:
    """
    Async file management utility class.
    """

    async def save_file(self, path: Path, content: bytes) -> None:
        """
        Save bytes content to specified path asynchronously.
        Creates parent dirs if needed.

        Args:
            path: Destination file path.
            content: Data bytes to write.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(path, "wb") as f:
            await f.write(content)

    async def calculate_sha256(self, path: Path) -> str:
        """
        Calculate SHA256 hash of a file asynchronously.

        Args:
            path: File path.

        Returns:
            Hexadecimal SHA256 hash string.
        """
        hash_sha256 = hashlib.sha256()
        async with aiofiles.open(path, "rb") as f:
            while True:
                chunk = await f.read(8192)
                if not chunk:
                    break
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()

    def sanitize_filename(self, filename: str) -> str:
        """
        Sanitize filename to remove unsafe characters.

        Args:
            filename: Original filename.

        Returns:
            Sanitized safe filename string.
        """
        keepcharacters = (" ", ".", "_", "-")
        return "".join(
            c for c in filename if c.isalnum() or c in keepcharacters
        ).strip()

    def generate_unique_filename(
        self, original_filename: str, suffix: Optional[str] = None
    ) -> str:
        """
        Generate a unique filename preserving extension.

        Args:
            original_filename: Source filename.
            suffix: Optional string suffix.

        Returns:
            Unique filename string.
        """
        base, ext = os.path.splitext(original_filename)
        suffix_str = f"_{suffix}" if suffix else ""
        unique_id = os.urandom(8).hex()
        safe_base = self.sanitize_filename(base)
        return f"{safe_base}{suffix_str}_{unique_id}{ext}"
