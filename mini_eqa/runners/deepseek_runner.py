from __future__ import annotations

import os
import random
import sys
import time
from typing import Any

_RETRYABLE_HTTP_STATUSES = frozenset({429, 500, 502, 503, 504})
_FATAL_HTTP_STATUSES = frozenset({401, 403})


def _build_fallback_prompt(question: str, retrieved: list[dict]) -> str:
    evidence_lines = "\n".join(
        f"- {item['frame_id']}: {item['caption']}"
        for item in retrieved
    )
    return (
        "Answer the question using only the retrieved evidence.\n\n"
        f"Question:\n{question}\n\n"
        f"Retrieved evidence:\n{evidence_lines}\n\n"
        "Give a short answer."
    )


def _extract_text_from_response(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    choices = getattr(response, "choices", None)
    if choices:
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", "")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            text_parts = []
            for item in content:
                text = getattr(item, "text", None)
                if text:
                    text_parts.append(text)
                elif isinstance(item, dict) and item.get("text"):
                    text_parts.append(item["text"])
            if text_parts:
                return "".join(text_parts).strip()

    output = getattr(response, "output", None)
    if output:
        text_parts = []
        for item in output:
            content = getattr(item, "content", None) or item.get("content", [])
            for block in content:
                text = getattr(block, "text", None)
                if text:
                    text_parts.append(text)
                elif isinstance(block, dict) and block.get("text"):
                    text_parts.append(block["text"])
        if text_parts:
            return "".join(text_parts).strip()

    raise RuntimeError("deepseek response did not contain readable text output.")


def _http_status_from_exc(exc: Exception) -> int | None:
    """Extract HTTP status code from an exception if present."""
    for attr in ("response", "http_response"):
        response = getattr(exc, attr, None)
        if response is not None:
            code = getattr(response, "status_code", None)
            if isinstance(code, int):
                return code
    code = getattr(exc, "status_code", None)
    if isinstance(code, int):
        return code
    return None


def _is_retryable(exc: Exception) -> bool:
    """Return True if the exception represents a transient failure worth retrying."""
    try:
        import requests.exceptions as req_exc

        if isinstance(exc, (req_exc.SSLError, req_exc.ConnectionError, req_exc.Timeout)):
            return True
    except ImportError:
        pass

    status = _http_status_from_exc(exc)
    if status is not None:
        return status in _RETRYABLE_HTTP_STATUSES

    return False


def _is_fatal(exc: Exception) -> bool:
    """Return True if the exception should never be retried (auth errors)."""
    status = _http_status_from_exc(exc)
    if status is not None:
        return status in _FATAL_HTTP_STATUSES
    return False


def _call_with_retry(
    fn,
    max_retries: int,
    initial_sleep: float,
    multiplier: float = 2.0,
    jitter_fraction: float = 0.1,
) -> Any:
    """Call fn(), retrying on transient errors with exponential backoff + jitter.

    Non-retryable exceptions (auth errors, logic errors) are re-raised immediately.
    After max_retries exhausted, raises RuntimeError with the last exception.
    """
    sleep_time = initial_sleep
    last_exc: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as exc:
            if _is_fatal(exc):
                status = _http_status_from_exc(exc)
                raise RuntimeError(
                    f"deepseek authentication error (HTTP {status}). "
                    "Check your DEEPSEEK_API_KEY. Not retrying."
                ) from exc

            if not _is_retryable(exc):
                raise

            last_exc = exc
            if attempt < max_retries:
                jitter = sleep_time * jitter_fraction * random.random()
                actual_sleep = sleep_time + jitter
                print(
                    f"[deepseek retry {attempt + 1}/{max_retries}] "
                    f"{type(exc).__name__}: {exc} — "
                    f"sleeping {actual_sleep:.1f}s before next attempt.",
                    file=sys.stderr,
                )
                time.sleep(actual_sleep)
                sleep_time *= multiplier

    raise RuntimeError(
        f"deepseek API request failed after {max_retries + 1} attempts. "
        f"Last error: {type(last_exc).__name__}: {last_exc}"
    ) from last_exc


def deepseek_answer(
    question: str,
    retrieved: list[dict],
    prompt: str | None = None,
    model: str = "gpt-4o-mini",
    max_output_tokens: int = 128,
    max_retries: int = 5,
    retry_initial_sleep: float = 3.0,
) -> str:
    try:
        from dotenv import load_dotenv
    except ImportError as exc:
        raise ImportError(
            "The python-dotenv package is not installed. "
            "Install dependencies from requirements.txt before using this runner."
        ) from exc

    load_dotenv()

    api_key = os.getenv("deepseek_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing deepseek_API_KEY environment variable. "
            "Set it in your shell or .env before using --runner deepseek."
        )

    try:
        import deepseek
    except ImportError as exc:
        raise ImportError(
            "The deepseek package is not installed. "
            "Install dependencies from requirements.txt before using this runner."
        ) from exc

    prompt_text = prompt if prompt is not None else _build_fallback_prompt(question, retrieved)
    resolved_model = model
    if model == "gpt-4o-mini":
        resolved_model = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")

    # Client construction makes no network call — do it outside the retry loop.
    try:
        if hasattr(deepseek, "DeepSeekAPI"):
            client = deepseek.DeepSeekAPI(api_key=api_key)
        elif hasattr(deepseek, "DeepSeek"):
            client = deepseek.DeepSeek(api_key=api_key)
        elif hasattr(deepseek, "Client"):
            client = deepseek.Client(api_key=api_key)
        else:
            raise RuntimeError(
                "Unsupported deepseek SDK interface. Expected DeepSeekAPI, DeepSeek, or Client."
            )
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"Failed to initialize deepseek client: {exc}") from exc

    def _do_call() -> str:
        if hasattr(client, "chat_completion"):
            return client.chat_completion(
                prompt=prompt_text,
                model=resolved_model,
                max_tokens=max_output_tokens,
            ).strip()

        if hasattr(client, "responses"):
            response = client.responses.create(
                model=resolved_model,
                input=prompt_text,
                max_output_tokens=max_output_tokens,
            )
            return _extract_text_from_response(response)

        if hasattr(client, "chat") and hasattr(client.chat, "completions"):
            response = client.chat.completions.create(
                model=resolved_model,
                messages=[{"role": "user", "content": prompt_text}],
                max_tokens=max_output_tokens,
            )
            return _extract_text_from_response(response)

        raise RuntimeError(
            "Unsupported deepseek client interface. Expected responses or chat.completions."
        )

    return _call_with_retry(
        _do_call,
        max_retries=max_retries,
        initial_sleep=retry_initial_sleep,
    )
