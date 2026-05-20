from __future__ import annotations

import os
from typing import Any


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


def deepseek_answer(
    question: str,
    retrieved: list[dict],
    prompt: str | None = None,
    model: str = "gpt-4o-mini",
    max_output_tokens: int = 128,
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
    except Exception as exc:
        raise RuntimeError(f"deepseek API request failed: {exc}") from exc
