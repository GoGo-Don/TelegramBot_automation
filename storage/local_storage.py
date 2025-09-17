"""
Local Storage Module

Provides utilities for managing files and directories on the local filesystem,
including saving, moving, and cleaning up media and temporary files.

Author: GG
Date: 2025-09-16
"""

import logging
import os
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


class LocalStorage:
    """
    Handles local file system storage operations.
    """

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        if not self.base_dir.exists():
            self.base_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"LocalStorage initialized at {self.base_dir}")

    def save_file(self, src_path: Path, sub_dir: str = "uploads") -> Path:
        """
        Move a file into an organized subdirectory.

        Args:
            src_path: The source file path.
            sub_dir: Subdirectory under base_dir to store the file.

        Returns:
            Target file path.
        """
        target_dir = self.base_dir / sub_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / src_path.name
        shutil.move(str(src_path), target_path)
        logger.debug(f"Moved file {src_path} to {target_path}")
        return target_path

    def cleanup_files(self, sub_dir: str = "temp", max_age_seconds: int = 86400) -> int:
        """
        Delete files older than max_age_seconds in specified subdirectory.

        Args:
            sub_dir: Subdirectory relative to base_dir.
            max_age_seconds: Maximum age in seconds to keep files.

        Returns:
            Number of deleted files.
        """
        deleted_count = 0
        cleanup_dir = self.base_dir / sub_dir
        if not cleanup_dir.exists():
            return deleted_count
        now = os.stat_result.st_mtime
        for file_path in cleanup_dir.iterdir():
            if file_path.is_file():
                age = Path().stat().st_mtime - file_path.stat().st_mtime
                if age > max_age_seconds:
                    try:
                        file_path.unlink()
                        deleted_count += 1
                        logger.debug(f"Deleted old file: {file_path}")
                    except Exception as e:
                        logger.error(f"Error deleting file {file_path}: {e}")
        logger.info(
            f"Cleanup complete, deleted {deleted_count} files from {cleanup_dir}"
        )
        return deleted_count
