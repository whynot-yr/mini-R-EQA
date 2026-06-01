from __future__ import annotations

import json
import re
import sys
import time
from typing import Any

_JUDGE_PROMPT_TEMPLATE = """\
You are evaluating an answer for an embodied question answering task.

Question:
{question}

Gold answer:
{gold_answer}

Predicted answer:
{predicted_answer}

Judge whether the predicted answer semantically answers the question correctly compared with the gold answer.

Scoring rules:
- 1.0: The predicted answer is semantically correct or equivalent to the gold answer.
- 0.5: The predicted answer is partially correct, incomplete, or contains the correct answer along with some extra but non-contradictory information.
- 0.0: The predicted answer is wrong, contradicts the gold answer, says the evidence is insufficient, or only mentions the gold answer in a negated/irrelevant way.

Important:
- Do not reward an answer just because it contains a word from the gold answer.
- If the predicted answer says the answer cannot be determined, and the gold answer is specific, score 0.0.
- Ignore minor wording differences.
- Accept synonyms if they clearly refer to the same object/place/relation.

Return JSON only:
{{
  "score": 0.0,
  "label": "correct|partial|incorrect",
  "rationale": "brief reason"
}}"""

_VALID_LABELS = {"correct", "partial", "incorrect", "judge_parse_error"}
_RETRYABLE_HTTP_STATUSES = frozenset({429, 500, 502, 503, 504})
_FATAL_HTTP_STATUSES = frozenset({401, 403})


def _build_judge_prompt(*, question: str, gold_answer: str, predicted_answer: str) -> str:
    return _JUDGE_PROMPT_TEMPLATE.format(
        question=question,
        gold_answer=gold_answer,
        predicted_answer=predicted_answer,
    )


def _clamp_score(score: float) -> float:
    if score >= 0.75:
        return 1.0
    if score >= 0.25:
        return 0.5
    return 0.0


def _label_from_score(score: float) -> str:
    if score == 1.0:
        return "correct"
    if score == 0.5:
        return "partial"
    return "incorrect"


def _parse_judge_response(raw: str) -> dict[str, Any]:
    """Parse a judge JSON response robustly, returning a validated dict."""

    def _validate(data: dict) -> dict:
        score = _clamp_score(float(data.get("score", 0.0)))
        label = str(data.get("label", "incorrect")).strip().lower()
        if label not in _VALID_LABELS:
            label = _label_from_score(score)
        rationale = str(data.get("rationale", "")).strip()
        return {
            "score": score,
            "label": label,
            "rationale": rationale,
            "raw_response": raw,
        }

    # Step 1: direct JSON parse
    try:
        data = json.loads(raw.strip())
        if isinstance(data, dict):
            return _validate(data)
    except (json.JSONDecodeError, ValueError):
        pass

    # Step 2: extract first {...} block (handles markdown fences and preamble text)
    match = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, dict):
                return _validate(data)
        except (json.JSONDecodeError, ValueError):
            pass

    # Step 3: graceful fallback
    return {
        "score": 0.0,
        "label": "judge_parse_error",
        "rationale": "Failed to parse judge response as JSON.",
        "raw_response": raw,
    }


def _is_judge_retryable(exc: Exception) -> bool:
    # OpenAI SDK exceptions
    try:
        import openai

        if isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError)):
            return True
        if isinstance(exc, openai.RateLimitError):
            return True
        if isinstance(exc, openai.InternalServerError):
            return True
        if isinstance(exc, openai.APIStatusError):
            return exc.status_code in _RETRYABLE_HTTP_STATUSES
    except ImportError:
        pass

    # requests exceptions (fallback for SDK internals that bubble up)
    try:
        import requests.exceptions

        if isinstance(exc, (
            requests.exceptions.SSLError,
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
        )):
            return True
    except ImportError:
        pass

    return False


def _is_judge_fatal(exc: Exception) -> bool:
    try:
        import openai

        if isinstance(exc, (openai.AuthenticationError, openai.PermissionDeniedError)):
            return True
        if isinstance(exc, openai.APIStatusError):
            return exc.status_code in _FATAL_HTTP_STATUSES
    except ImportError:
        pass
    return False


def _judge_call_with_retry(
    fn,
    max_retries: int,
    initial_sleep: float,
    multiplier: float = 2.0,
    jitter_fraction: float = 0.1,
) -> Any:
    import random

    sleep_time = initial_sleep
    last_exc: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as exc:
            if _is_judge_fatal(exc):
                raise RuntimeError(
                    f"DeepSeek judge authentication error: {exc}. "
                    "Check your DEEPSEEK_API_KEY. Not retrying."
                ) from exc
            if not _is_judge_retryable(exc):
                raise
            last_exc = exc
            if attempt < max_retries:
                jitter = sleep_time * jitter_fraction * random.random()
                actual_sleep = sleep_time + jitter
                print(
                    f"[judge retry {attempt + 1}/{max_retries}] "
                    f"{type(exc).__name__}: {exc} — "
                    f"sleeping {actual_sleep:.1f}s before next attempt.",
                    file=sys.stderr,
                )
                time.sleep(actual_sleep)
                sleep_time *= multiplier

    raise RuntimeError(
        f"DeepSeek judge failed after {max_retries + 1} attempts. "
        f"Last error: {type(last_exc).__name__}: {last_exc}"
    ) from last_exc


def judge_answer_with_deepseek(
    *,
    question: str,
    gold_answer: str,
    predicted_answer: str,
    model: str = "deepseek-chat",
    max_output_tokens: int = 128,
    temperature: float = 0.0,
    max_retries: int = 5,
    retry_initial_sleep: float = 3.0,
    api_key: str | None = None,
    base_url: str = "https://api.deepseek.com",
) -> dict[str, Any]:
    """Call DeepSeek as a semantic judge and return a structured result.

    Returns a JSON-serializable dict:
        {
          "score": 0.0 | 0.5 | 1.0,
          "label": "correct" | "partial" | "incorrect" | "judge_parse_error",
          "rationale": "...",
          "raw_response": "..."
        }
    """
    import os

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "The openai package is required for DeepSeek judge. "
            "Install it with: pip install openai"
        ) from exc

    resolved_key = api_key or os.getenv("DEEPSEEK_API_KEY") or os.getenv("deepseek_API_KEY")
    if not resolved_key:
        raise RuntimeError(
            "Missing DEEPSEEK_API_KEY environment variable. "
            "Set it before using DeepSeek judge."
        )

    client = OpenAI(api_key=resolved_key, base_url=base_url)
    prompt = _build_judge_prompt(
        question=question,
        gold_answer=gold_answer,
        predicted_answer=predicted_answer,
    )

    def _call() -> dict[str, Any]:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_output_tokens,
        )
        content = response.choices[0].message.content or ""
        return _parse_judge_response(content)

    return _judge_call_with_retry(
        _call,
        max_retries=max_retries,
        initial_sleep=retry_initial_sleep,
    )
