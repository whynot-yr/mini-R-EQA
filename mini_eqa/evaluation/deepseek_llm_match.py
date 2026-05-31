from __future__ import annotations

import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from mini_eqa.exporters.official_results import (
    load_dataset_items,
    load_official_results_items,
    order_items_by_dataset,
)
from mini_eqa.utils.io_utils import load_json, save_json


SCORE_REGEX = re.compile(r"Your mark:\s*([1-5])\b", re.IGNORECASE)
FALLBACK_INTEGER_REGEX = re.compile(r"\b([1-5])\b")
DEFAULT_PROGRESS_EVERY = 10


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_judge_score(text: str) -> int | None:
    match = SCORE_REGEX.search(text)
    if match:
        return int(match.group(1))

    fallback_match = FALLBACK_INTEGER_REGEX.search(text)
    if fallback_match:
        return int(fallback_match.group(1))

    return None


def map_score_to_100(score_1_to_5: int) -> float:
    clipped_score = min(max(score_1_to_5, 1), 5)
    return 100.0 * (clipped_score - 1) / 4.0


def build_judge_prompt(
    *,
    question: str,
    gold_answer: str,
    extra_answers: list[str],
    prediction: str,
) -> str:
    extra_answers_text = "\n".join(f"- {answer}" for answer in extra_answers) or "- None"
    return (
        "You are grading a visual question answering prediction.\n"
        "Judge semantic correctness, not exact string overlap.\n"
        "Accept synonyms, paraphrases, and equivalent descriptions.\n"
        "Treat the extra answers as valid alternative references.\n"
        "Penalize contradictions, hallucinations, and answers that miss key facts.\n\n"
        "Score rubric:\n"
        "5 = fully correct or semantically equivalent\n"
        "4 = mostly correct with only a minor omission or imprecision\n"
        "3 = partially correct\n"
        "2 = mostly incorrect but somewhat related\n"
        "1 = incorrect, irrelevant, contradictory, or no answer\n\n"
        f"Question:\n{question}\n\n"
        f"Gold answer:\n{gold_answer}\n\n"
        f"Extra valid answers:\n{extra_answers_text}\n\n"
        f"Model prediction:\n{prediction}\n\n"
        "Return only one line in this exact format:\n"
        "Your mark: X"
    )


def extract_response_text(response: Any) -> str:
    if not getattr(response, "choices", None):
        raise RuntimeError("Judge response did not contain choices.")

    content = response.choices[0].message.content
    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("text"):
                text_parts.append(str(item["text"]))
            elif hasattr(item, "text") and item.text:
                text_parts.append(str(item.text))
        text = "".join(text_parts).strip()
    else:
        text = str(content or "").strip()

    if not text:
        raise RuntimeError("Judge response did not contain readable text.")

    return text


def build_output_payload(
    *,
    created_at: str,
    dataset_path: str | Path,
    results_path: str | Path,
    model: str,
    base_url: str,
    total_dataset_questions: int,
    per_question_entries: list[dict[str, Any]],
) -> dict[str, Any]:
    completed_entries = [
        entry
        for entry in per_question_entries
        if entry.get("status") == "completed"
        and entry.get("mapped_score_0_to_100") is not None
    ]
    failed_entries = [
        entry for entry in per_question_entries if entry.get("status") == "failed"
    ]
    dry_run_entries = [
        entry for entry in per_question_entries if entry.get("status") == "dry_run"
    ]
    overall_mapped_score = (
        mean(entry["mapped_score_0_to_100"] for entry in completed_entries)
        if completed_entries
        else None
    )

    metadata = {
        "evaluator_name": "deepseek_llm_match",
        "note": (
            "This is a DeepSeek-based OpenEQA-style LLM judge. "
            "It is not the official OpenEQA GPT-4 LLM-Match score."
        ),
        "dataset_path": str(dataset_path),
        "results_path": str(results_path),
        "model": model,
        "base_url": base_url,
        "created_at": created_at,
        "updated_at": utc_now_iso(),
        "total_dataset_questions": total_dataset_questions,
        "processed_count": len(per_question_entries),
        "evaluated_count": len(completed_entries),
        "failed_count": len(failed_entries),
        "dry_run_count": len(dry_run_entries),
        "overall_mapped_score": overall_mapped_score,
    }
    summary = {
        "overall_mapped_score": overall_mapped_score,
        "evaluated_count": len(completed_entries),
        "failed_count": len(failed_entries),
        "dry_run_count": len(dry_run_entries),
        "total_dataset_questions": total_dataset_questions,
    }
    return {
        "metadata": metadata,
        "summary": summary,
        "per_question": per_question_entries,
    }


def _completed_question_ids(existing_entries: list[dict[str, Any]]) -> set[str]:
    completed_ids: set[str] = set()
    for entry in existing_entries:
        if (
            entry.get("status") == "completed"
            and entry.get("parsed_score_1_to_5") is not None
            and entry.get("question_id")
        ):
            completed_ids.add(str(entry["question_id"]))
    return completed_ids


def _normalize_extra_answers(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _ordered_entries_from_map(
    dataset_items: list[dict[str, Any]],
    entry_map: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    ordered_entries: list[dict[str, Any]] = []
    for dataset_item in dataset_items:
        question_id = str(dataset_item["question_id"])
        if question_id in entry_map:
            ordered_entries.append(entry_map[question_id])
    return ordered_entries


def _require_openai() -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "The openai package is not installed. Install project dependencies before "
            "running DeepSeek LLM-Match evaluation."
        ) from exc
    return OpenAI


def run_deepseek_llm_match(
    *,
    dataset_path: str | Path,
    results_path: str | Path,
    output_path: str | Path,
    model: str = "deepseek-chat",
    base_url: str = "https://api.deepseek.com",
    api_key_env: str = "DEEPSEEK_API_KEY",
    limit: int | None = None,
    sleep_seconds: float = 0.0,
    max_retries: int = 2,
    timeout: float = 60.0,
    resume: bool = False,
    dry_run: bool = False,
    verbose: bool = False,
    progress_every: int = DEFAULT_PROGRESS_EVERY,
) -> dict[str, Any]:
    if limit is not None and limit <= 0:
        raise ValueError(f"limit must be positive when provided, got {limit}")
    if max_retries < 0:
        raise ValueError(f"max_retries must be >= 0, got {max_retries}")
    if timeout <= 0:
        raise ValueError(f"timeout must be positive, got {timeout}")
    if sleep_seconds < 0:
        raise ValueError(f"sleep_seconds must be >= 0, got {sleep_seconds}")

    dataset_items = load_dataset_items(dataset_path)
    result_items = load_official_results_items(results_path)
    ordered_results, _ = order_items_by_dataset(
        dataset_items,
        result_items,
        observed_label="results",
    )
    ordered_pairs = list(zip(dataset_items, ordered_results))
    if limit is not None:
        ordered_pairs = ordered_pairs[:limit]

    output_path = Path(output_path)
    existing_output: dict[str, Any] | None = None
    if resume and output_path.exists():
        existing_output = load_json(output_path)
        if not isinstance(existing_output, dict):
            raise ValueError(f"Existing output {output_path} must be a JSON object.")

    existing_entries = []
    created_at = utc_now_iso()
    if existing_output is not None:
        metadata = existing_output.get("metadata")
        if isinstance(metadata, dict) and isinstance(metadata.get("created_at"), str):
            created_at = metadata["created_at"]
        existing_entries = existing_output.get("per_question", [])
        if not isinstance(existing_entries, list):
            raise ValueError(
                f"Existing output {output_path} has a non-list per_question field."
            )

    ordered_existing_map: dict[str, dict[str, Any]] = {}
    for entry in existing_entries:
        if not isinstance(entry, dict):
            raise ValueError(f"Existing output {output_path} contains a non-object entry.")
        question_id = entry.get("question_id")
        if question_id:
            ordered_existing_map[str(question_id)] = entry

    completed_ids = _completed_question_ids(existing_entries)

    client = None
    if not dry_run:
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Missing API key environment variable {api_key_env}. "
                "Set it before running DeepSeek LLM-Match evaluation."
            )
        OpenAI = _require_openai()
        client = OpenAI(api_key=api_key, base_url=base_url)

    total_to_process = len(ordered_pairs)
    for index, (dataset_item, result_item) in enumerate(ordered_pairs, start=1):
        question_id = str(dataset_item["question_id"])
        if resume and question_id in completed_ids:
            if verbose:
                print(f"[skip] {index}/{total_to_process} question_id={question_id}")
            continue

        question = str(dataset_item.get("question", ""))
        gold_answer = str(dataset_item.get("answer", ""))
        extra_answers = _normalize_extra_answers(dataset_item.get("extra_answers"))
        prediction = str(result_item.get("answer", ""))

        entry: dict[str, Any] = {
            "question_id": question_id,
            "question": question,
            "gold_answer": gold_answer,
            "extra_answers": extra_answers,
            "prediction": prediction,
            "raw_judge_response": None,
            "parsed_score_1_to_5": None,
            "mapped_score_0_to_100": None,
            "status": "pending",
            "error": None,
        }

        if dry_run:
            entry["status"] = "dry_run"
            ordered_existing_map[question_id] = entry
        else:
            prompt = build_judge_prompt(
                question=question,
                gold_answer=gold_answer,
                extra_answers=extra_answers,
                prediction=prediction,
            )
            final_error: str | None = None

            for attempt in range(max_retries + 1):
                try:
                    response = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0,
                        max_tokens=64,
                        timeout=timeout,
                    )
                    raw_response = extract_response_text(response)
                    entry["raw_judge_response"] = raw_response
                    parsed_score = parse_judge_score(raw_response)
                    if parsed_score is None:
                        raise RuntimeError(
                            "Could not parse judge score from response. "
                            "Expected 'Your mark: X'."
                        )

                    entry["parsed_score_1_to_5"] = parsed_score
                    entry["mapped_score_0_to_100"] = map_score_to_100(parsed_score)
                    entry["status"] = "completed"
                    entry["error"] = None
                    final_error = None
                    break
                except Exception as exc:
                    final_error = str(exc)
                    entry["error"] = final_error
                    if attempt >= max_retries:
                        entry["status"] = "failed"
                        break
                    if verbose:
                        print(
                            f"[retry] {index}/{total_to_process} question_id={question_id} "
                            f"attempt={attempt + 1} error={final_error}"
                        )
                    if sleep_seconds > 0:
                        time.sleep(sleep_seconds)

            if entry["status"] == "failed" and verbose:
                print(
                    f"[failed] {index}/{total_to_process} question_id={question_id} "
                    f"error={entry['error']}"
                )

            ordered_existing_map[question_id] = entry

        ordered_entries = _ordered_entries_from_map(dataset_items, ordered_existing_map)
        payload = build_output_payload(
            created_at=created_at,
            dataset_path=dataset_path,
            results_path=results_path,
            model=model,
            base_url=base_url,
            total_dataset_questions=len(dataset_items),
            per_question_entries=ordered_entries,
        )
        save_json(payload, output_path)

        if verbose or index % progress_every == 0 or index == total_to_process:
            print(
                f"[progress] {index}/{total_to_process} processed "
                f"(completed={payload['metadata']['evaluated_count']}, "
                f"failed={payload['metadata']['failed_count']}, "
                f"dry_run={payload['metadata']['dry_run_count']})"
            )

        if sleep_seconds > 0 and not dry_run:
            time.sleep(sleep_seconds)

    final_entries = _ordered_entries_from_map(dataset_items, ordered_existing_map)
    final_payload = build_output_payload(
        created_at=created_at,
        dataset_path=dataset_path,
        results_path=results_path,
        model=model,
        base_url=base_url,
        total_dataset_questions=len(dataset_items),
        per_question_entries=final_entries,
    )
    save_json(final_payload, output_path)
    return final_payload
