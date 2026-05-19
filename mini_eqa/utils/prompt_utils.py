from pathlib import Path
from typing import Any


DEFAULT_PROMPT_DIR = Path(__file__).resolve().parents[2] / "prompts"

PROMPT_NAME_TO_PATH = {
    "mini_rag": DEFAULT_PROMPT_DIR / "mini_rag.txt",
}


def load_prompt(name: str) -> str:
    if name not in PROMPT_NAME_TO_PATH:
        valid_names = ", ".join(PROMPT_NAME_TO_PATH.keys())
        raise ValueError(f"Invalid prompt name: {name}. Valid names: {valid_names}")

    path = PROMPT_NAME_TO_PATH[name]
    with path.open("r", encoding="utf-8") as f:
        return f.read().strip()


def format_evidence(retrieved: list[dict[str, Any]]) -> str:
    return "\n".join(
        f"- {item['frame_id']}: {item['caption']}"
        for item in retrieved
    )


def build_prompt(question: str, retrieved: list[dict[str, Any]], prompt_name: str = "mini_rag") -> str:
    template = load_prompt(prompt_name)
    evidence_text = format_evidence(retrieved)

    return template.format(
        question=question,
        evidence_text=evidence_text,
    )