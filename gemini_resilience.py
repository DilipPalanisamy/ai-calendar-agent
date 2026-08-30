import time
import asyncio
import logging
from typing import Any, Callable

try:
    import google.api_core.exceptions
    ResourceExhausted = google.api_core.exceptions.ResourceExhausted
except ImportError:
    ResourceExhausted = Exception

logger = logging.getLogger(__name__)


def is_rate_limit_error(exc: Exception) -> bool:
    """Detect if an exception is caused by a 429 Rate Limit or ResourceExhausted quota error."""
    if isinstance(exc, ResourceExhausted):
        return True
    err_str = str(exc).lower()
    return any(indicator in err_str for indicator in [
        "429",
        "resource_exhausted",
        "resourceexhausted",
        "quota",
        "rate limit",
        "rate_limit",
        "quota exceeded",
        "too many requests",
        "503",
        "unavailable",
        "high demand",
    ])


def call_gemini_with_retry(
    model_or_func: Any,
    prompt_or_arg: Any = None,
    max_retries: int = 5,
    initial_delay: float = 2.0,
    backoff_factor: float = 2.0,
    **kwargs
) -> Any:
    """
    Synchronous Exponential Backoff wrapper for Google Gemini API calls.
    Supports:
      1. call_gemini_with_retry(model, prompt, max_retries=5, initial_delay=2)
      2. call_gemini_with_retry(func, *args, max_retries=5, initial_delay=2, **kwargs)
    """
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            if hasattr(model_or_func, "generate_content"):
                return model_or_func.generate_content(prompt_or_arg, **kwargs)
            elif callable(model_or_func):
                if prompt_or_arg is not None:
                    return model_or_func(prompt_or_arg, **kwargs)
                return model_or_func(**kwargs)
            else:
                raise ValueError("model_or_func must be a callable or have a generate_content method.")
        except Exception as e:
            if is_rate_limit_error(e):
                if attempt == max_retries - 1:
                    logger.error(f"Failed after {max_retries} attempts due to rate limits: {e}")
                    raise e

                logger.warning(
                    f"Rate limit hit (Attempt {attempt + 1}/{max_retries}). Retrying in {delay} seconds..."
                )
                time.sleep(delay)
                delay *= backoff_factor
            else:
                raise e


async def acall_gemini_with_retry(
    async_func_or_model: Any,
    prompt_or_arg: Any = None,
    max_retries: int = 5,
    initial_delay: float = 2.0,
    backoff_factor: float = 2.0,
    **kwargs
) -> Any:
    """
    Asynchronous Exponential Backoff wrapper for Google Gemini API calls.
    Supports:
      1. acall_gemini_with_retry(async_func, prompt_or_arg, max_retries=5, initial_delay=2)
      2. acall_gemini_with_retry(model.ainvoke, messages)
    """
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            if callable(async_func_or_model):
                if prompt_or_arg is not None:
                    return await async_func_or_model(prompt_or_arg, **kwargs)
                return await async_func_or_model(**kwargs)
            elif hasattr(async_func_or_model, "ainvoke"):
                return await async_func_or_model.ainvoke(prompt_or_arg, **kwargs)
            elif hasattr(async_func_or_model, "agenerate_content"):
                return await async_func_or_model.agenerate_content(prompt_or_arg, **kwargs)
            else:
                raise ValueError("async_func_or_model must be an async callable or have an ainvoke/agenerate_content method.")
        except Exception as e:
            if is_rate_limit_error(e):
                if attempt == max_retries - 1:
                    logger.error(f"Failed after {max_retries} attempts due to rate limits: {e}")
                    raise e

                logger.warning(
                    f"Rate limit hit (Attempt {attempt + 1}/{max_retries}). Retrying in {delay} seconds..."
                )
                await asyncio.sleep(delay)
                delay *= backoff_factor
            else:
                raise e
