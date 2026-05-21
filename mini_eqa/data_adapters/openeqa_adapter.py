from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from mini_eqa.utils.io_utils import load_json, save_json

QUESTION_KEYS = ("question", "query", "question_text", "prompt")
QUESTION_ID_KEYS = ("question_id", "id", "uid", "qid")
ANSWER_KEYS = ("answer", "gt_answer", "gold_answer", "target")
ANSWER_LIST_KEYS = ("answers", "gt_answers", "gold_answers")
EPISODE_HISTORY_KEYS = ("episode_history", "episode_id", "episode", "history")
SCENE_ID_KEYS = ("scene_id", "scene", "scan_id", "environment_id")


def _first_present(raw_item: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in raw_item and raw_item[key] is not None:
            return raw_item[key]
    return None


def _normalize_answer(raw_answer: Any) -> str | None:
    if raw_answer is None:
        return None
    if isinstance(raw_answer, str):
        answer = raw_answer.strip()
        return answer or None
    if isinstance(raw_answer, dict):
        nested = _first_present(raw_answer, ANSWER_KEYS + ANSWER_LIST_KEYS)
        return _normalize_answer(nested)
    if isinstance(raw_answer, list):
        normalized_items = []
        for item in raw_answer:
            normalized_item = _normalize_answer(item)
            if normalized_item:
                normalized_items.append(normalized_item)
        if not normalized_items:
            return None
        return " | ".join(normalized_items)
    return str(raw_answer)


def extract_question_item(raw_item: dict[str, Any]) -> dict[str, Any]:
    question = _first_present(raw_item, QUESTION_KEYS)
    if question is None:
        raise KeyError(
            "Unable to extract question text from raw item. "
            f"Tried keys: {QUESTION_KEYS}."
        )

    return {
        "question_id": _first_present(raw_item, QUESTION_ID_KEYS),
        "question": str(question).strip(),
        "answer": _normalize_answer(
            _first_present(raw_item, ANSWER_KEYS)
            if _first_present(raw_item, ANSWER_KEYS) is not None
            else _first_present(raw_item, ANSWER_LIST_KEYS)
        ),
        "episode_history": _first_present(raw_item, EPISODE_HISTORY_KEYS),
        "scene_id": _first_present(raw_item, SCENE_ID_KEYS),
        "raw_item": raw_item,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert OpenEQA-style QA JSON into mini-R-EQA questions.json format."
    )
    parser.add_argument("--qa_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--episode_id", type=str, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load_items(qa_file: Path) -> list[dict[str, Any]]:
    data = load_json(qa_file)

    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        candidate = None
        for key in ("questions", "data", "items", "qa_pairs", "qas"):
            value = data.get(key)
            if isinstance(value, list):
                candidate = value
                break
        if candidate is None:
            raise ValueError(
                f"Unsupported QA JSON structure in {qa_file}. "
                "Expected a top-level list or a dict containing a list field such as "
                "'questions', 'data', 'items', 'qa_pairs', or 'qas'."
            )
        items = candidate
    else:
        raise ValueError(
            f"Unsupported QA JSON type in {qa_file}: {type(data).__name__}."
        )

    if not all(isinstance(item, dict) for item in items):
        raise ValueError(f"Expected every QA item in {qa_file} to be a JSON object.")

    return items


def _matches_episode(raw_item: dict[str, Any], episode_id: str) -> bool:
    candidates = [
        _first_present(raw_item, EPISODE_HISTORY_KEYS),
        raw_item.get("episode"),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        if str(candidate) == episode_id:
            return True
        if isinstance(candidate, dict):
            nested_id = _first_present(candidate, ("episode_id", "id", "name"))
            if nested_id is not None and str(nested_id) == episode_id:
                return True
    return False


def convert_items(
    raw_items: list[dict[str, Any]],
    start: int = 0,
    limit: int | None = None,
    episode_id: str | None = None,
) -> list[dict[str, Any]]:
    if start < 0:
        raise ValueError(f"start must be non-negative, got {start}")
    if limit is not None and limit <= 0:
        raise ValueError(f"limit must be positive when provided, got {limit}")

    filtered_items = raw_items
    if episode_id is not None:
        filtered_items = [item for item in raw_items if _matches_episode(item, episode_id)]

    sliced_items = filtered_items[start:]
    if limit is not None:
        sliced_items = sliced_items[:limit]

    questions = []
    for offset, raw_item in enumerate(sliced_items, start=start):
        question_item = extract_question_item(raw_item)
        if not question_item["question_id"]:
            question_item["question_id"] = f"q{offset + 1:06d}"
        questions.append(question_item)

    return questions


def main() -> None:
    args = parse_args()

    qa_file = Path(args.qa_file)
    output_dir = Path(args.output_dir)

    if not qa_file.exists():
        raise FileNotFoundError(f"QA file does not exist: {qa_file}")

    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory {output_dir} is not empty. Use --overwrite to replace files."
        )

    raw_items = _load_items(qa_file)
    questions = convert_items(
        raw_items=raw_items,
        start=args.start,
        limit=args.limit,
        episode_id=args.episode_id,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    save_json(questions, output_dir / "questions.json")
    save_json(
        {
            "qa_file": str(qa_file),
            "output_dir": str(output_dir),
            "episode_id": args.episode_id,
            "start": args.start,
            "limit": args.limit,
            "num_questions": len(questions),
        },
        output_dir / "adapter_meta.json",
    )

    print("=" * 80)
    print("OpenEQA Adapter")
    print("=" * 80)
    print(f"QA file: {qa_file}")
    print(f"Output dir: {output_dir}")
    print(f"Episode filter: {args.episode_id}")
    print(f"Start: {args.start}")
    print(f"Limit: {args.limit}")
    print(f"Number of questions: {len(questions)}")
    print(f"Saved: {output_dir / 'questions.json'}")
    print(f"Saved: {output_dir / 'adapter_meta.json'}")


if __name__ == "__main__":
    main()
