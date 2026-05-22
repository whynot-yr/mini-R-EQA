from __future__ import annotations

import os


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


def openai_compatible_answer(
    question: str,
    retrieved: list[dict],
    prompt: str | None = None,
    model: str = "meta-llama/Llama-3.1-70B-Instruct",
    base_url: str = "http://localhost:8000/v1",
    api_key: str = "EMPTY",
    max_output_tokens: int = 256,
    temperature: float = 0.2,
) -> str:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "The openai package is not installed. "
            "Install dependencies from requirements.txt or requirements-llama.txt."
        ) from exc

    resolved_api_key = os.getenv("LOCAL_LLM_API_KEY", api_key or "EMPTY")
    prompt_text = prompt if prompt is not None else _build_fallback_prompt(question, retrieved)

    try:
        client = OpenAI(base_url=base_url, api_key=resolved_api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt_text}],
            max_tokens=max_output_tokens,
            temperature=temperature,
        )
    except Exception as exc:
        raise RuntimeError(
            "OpenAI-compatible local runner request failed. "
            f"base_url={base_url}, model={model}. Error: {exc}"
        ) from exc

    if not response.choices:
        raise RuntimeError("OpenAI-compatible local runner returned no choices.")

    content = response.choices[0].message.content
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("text"):
                text_parts.append(item["text"])
            elif hasattr(item, "text") and item.text:
                text_parts.append(item.text)
        answer = "".join(text_parts).strip()
    else:
        answer = (content or "").strip()

    if not answer:
        raise RuntimeError("OpenAI-compatible local runner returned empty text.")

    return answer
