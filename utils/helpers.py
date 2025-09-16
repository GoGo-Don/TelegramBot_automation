"""
General Helper Utilities

Common utility functions used across the project.

Author: Development Team
Date: 2025-09-16
"""

import uuid


def generate_unique_id() -> str:
    """
    Generate a unique ID string using UUID4.

    Returns:
        Unique string identifier.
    """
    return str(uuid.uuid4())


def truncate_text(text: str, max_tokens: int) -> str:
    """
    Truncates the text to approximately max_tokens length, assuming average token length.

    Args:
        text: Input text.
        max_tokens: Max token count.

    Returns:
        Truncated text string.
    """
    # Rough approximation: 1.3 characters per token
    max_length = int(max_tokens * 1.3)
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def sanitize_filename(filename: str) -> str:
    """
    Sanitize the filename to be safe for file systems.

    Args:
        filename: Original filename.

    Returns:
        Sanitized filename string.
    """
    keepchars = (" ", ".", "_", "-")
    return "".join(c for c in filename if c.isalnum() or c in keepchars).strip()


def estimate_tokens(text: str, model: str) -> int:
    """
    Roughly estimate number of tokens for a given text and model.

    Args:
        text: Input prompt text.
        model: Model name (for future extension).

    Returns:
        Estimated token count.
    """
    # Placeholder: real tokenizers like tiktoken can be used
    return len(text) // 4  # Approximate 4 chars per token


async def retry_with_backoff(
    coro_func, *args, max_retries=3, delay=1.0, backoff=2.0, **kwargs
):
    """
    Retry asynchronous function with exponential backoff.

    Args:
        coro_func: Async function to retry.
        max_retries: Maximum attempts.
        delay: Initial delay seconds.
        backoff: Exponential backoff multiplier.
        *args: Positional args for coro_func.
        **kwargs: Keyword args for coro_func.

    Returns:
        coro_func result.

    Raises:
        Last exception if all retries fail.
    """
    attempt = 0
    current_delay = delay
    while attempt < max_retries:
        try:
            return await coro_func(*args, **kwargs)
        except Exception as e:
            attempt += 1
            if attempt == max_retries:
                raise
            await asyncio.sleep(current_delay)
            current_delay *= backoff
