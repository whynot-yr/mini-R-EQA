from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mini_eqa.utils.io_utils import load_json, save_json


METHOD_TO_PATTERN = {
    "reqa": "predictions_reqa_*.json",
    "uniform": "predictions_uniform_*.json",
}


@dataclass(frozen=True)
class QuestionIdCoverage:
    expected_count: int
    observed_count: int
    missing_ids: list[str]
    extra_ids: list[str]
    duplicate_ids: list[str]

    @property
    def missing_count(self) -> int:
        return len(self.missing_ids)

    @property
    def extra_count(self) -> int:
        return len(self.extra_ids)

    @property
    def duplicate_count(self) -> int:
        return len(self.duplicate_ids)

    @property
    def is_exact_match(self) -> bool:
        return (
            self.missing_count == 0
            and self.extra_count == 0
            and self.duplicate_count == 0
        )


def _preview_ids(values: Sequence[str], limit: int = 5) -> str:
    if not values:
        return "[]"
    preview = ", ".join(repr(value) for value in values[:limit])
    suffix = "" if len(values) <= limit else ", ..."
    return f"[{preview}{suffix}]"


def _require_question_id(item: dict[str, Any], context: str) -> str:
    question_id = item.get("question_id")
    if not question_id or not isinstance(question_id, str):
        raise KeyError(f"{context} is missing a valid string question_id.")
    return question_id


def load_dataset_items(dataset_path: str | Path) -> list[dict[str, Any]]:
    dataset_path = Path(dataset_path)
    dataset = load_json(dataset_path)
    if not isinstance(dataset, list):
        raise ValueError(f"Dataset {dataset_path} must be a JSON list.")

    dataset_items: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    duplicate_ids: list[str] = []

    for index, item in enumerate(dataset):
        if not isinstance(item, dict):
            raise ValueError(
                f"Dataset {dataset_path} item at index {index} must be a JSON object."
            )
        question_id = _require_question_id(item, f"Dataset item at index {index}")
        if question_id in seen_ids and question_id not in duplicate_ids:
            duplicate_ids.append(question_id)
        seen_ids.add(question_id)
        dataset_items.append(item)

    if duplicate_ids:
        raise ValueError(
            f"Dataset {dataset_path} contains duplicate question_id values: "
            f"{_preview_ids(duplicate_ids)}"
        )

    return dataset_items


def load_official_results_items(results_path: str | Path) -> list[dict[str, Any]]:
    results_path = Path(results_path)
    results = load_json(results_path)
    if not isinstance(results, list):
        raise ValueError(f"Results file {results_path} must be a JSON list.")

    result_items: list[dict[str, Any]] = []
    for index, item in enumerate(results):
        if not isinstance(item, dict):
            raise ValueError(
                f"Results file {results_path} item at index {index} must be a JSON object."
            )
        _require_question_id(item, f"Results item at index {index}")
        if "answer" not in item:
            raise KeyError(
                f"Results item at index {index} ({item.get('question_id')}) is missing answer."
            )
        result_items.append(item)
    return result_items


def validate_question_id_coverage(
    expected_ids: Sequence[str],
    observed_ids: Sequence[str],
) -> QuestionIdCoverage:
    expected_set = set(expected_ids)
    observed_set: set[str] = set()
    duplicate_ids: list[str] = []
    for question_id in observed_ids:
        if question_id in observed_set and question_id not in duplicate_ids:
            duplicate_ids.append(question_id)
        observed_set.add(question_id)

    missing_ids = [question_id for question_id in expected_ids if question_id not in observed_set]
    extra_ids = [question_id for question_id in observed_ids if question_id not in expected_set]

    return QuestionIdCoverage(
        expected_count=len(expected_ids),
        observed_count=len(observed_ids),
        missing_ids=missing_ids,
        extra_ids=extra_ids,
        duplicate_ids=duplicate_ids,
    )


def raise_for_question_id_mismatch(
    coverage: QuestionIdCoverage,
    *,
    dataset_label: str = "dataset",
    observed_label: str = "predictions",
) -> None:
    if coverage.is_exact_match:
        return

    message_lines = [
        f"Question ID mismatch between {dataset_label} and {observed_label}.",
        f"{dataset_label} count: {coverage.expected_count}",
        f"{observed_label} count: {coverage.observed_count}",
        f"Missing IDs: {coverage.missing_count} {_preview_ids(coverage.missing_ids)}",
        f"Extra IDs: {coverage.extra_count} {_preview_ids(coverage.extra_ids)}",
        f"Duplicate IDs: {coverage.duplicate_count} {_preview_ids(coverage.duplicate_ids)}",
    ]
    raise ValueError("\n".join(message_lines))


def order_items_by_dataset(
    dataset_items: Sequence[dict[str, Any]],
    observed_items: Sequence[dict[str, Any]],
    *,
    observed_label: str,
) -> tuple[list[dict[str, Any]], QuestionIdCoverage]:
    expected_ids = [_require_question_id(item, "Dataset item") for item in dataset_items]
    observed_ids = [_require_question_id(item, f"{observed_label} item") for item in observed_items]

    coverage = validate_question_id_coverage(expected_ids, observed_ids)
    raise_for_question_id_mismatch(
        coverage,
        dataset_label="dataset",
        observed_label=observed_label,
    )

    observed_map = {
        _require_question_id(item, f"{observed_label} item"): item for item in observed_items
    }
    ordered_items = [observed_map[question_id] for question_id in expected_ids]
    return ordered_items, coverage


def _extract_prediction_answer(item: dict[str, Any]) -> str:
    answer = item.get("predicted_answer")
    if answer is None:
        answer = item.get("prediction")
    if answer is None:
        raise KeyError(
            f"Prediction item {item.get('question_id')} is missing predicted_answer/prediction."
        )
    if not isinstance(answer, str):
        answer = str(answer)
    return answer


def _build_export_item(
    *,
    dataset_item: dict[str, Any],
    prediction_item: dict[str, Any],
    report: dict[str, Any],
    method: str,
    source_prediction_file: Path,
    include_debug_fields: bool,
) -> dict[str, Any]:
    export_item: dict[str, Any] = {
        "question_id": _require_question_id(prediction_item, "Prediction item"),
        "answer": _extract_prediction_answer(prediction_item),
    }

    if include_debug_fields:
        export_item["question"] = prediction_item.get("question", dataset_item.get("question"))
        export_item["gold_answer"] = prediction_item.get(
            "gold_answer",
            dataset_item.get("answer"),
        )
        export_item["episode_history"] = prediction_item.get(
            "episode_history",
            dataset_item.get("episode_history"),
        )
        export_item["scene_id"] = prediction_item.get("scene_id", dataset_item.get("scene_id"))
        export_item["method"] = method
        export_item["source_prediction_file"] = str(source_prediction_file)
        export_item["retrieved"] = prediction_item.get("retrieved")
        export_item["prompt"] = prediction_item.get("prompt")
        export_item["runner"] = report.get("runner")
        export_item["model"] = report.get("model")

    return export_item


def export_official_results(
    *,
    pred_dir: str | Path,
    method: str,
    dataset_path: str | Path,
    output_path: str | Path,
    include_debug_fields: bool = False,
) -> dict[str, Any]:
    if method not in METHOD_TO_PATTERN:
        raise ValueError(
            f"Unsupported method {method!r}. Expected one of: {sorted(METHOD_TO_PATTERN)}"
        )

    pred_dir = Path(pred_dir)
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)

    dataset_items = load_dataset_items(dataset_path)
    prediction_files = sorted(pred_dir.glob(METHOD_TO_PATTERN[method]))
    if not prediction_files:
        raise FileNotFoundError(
            f"No prediction files found in {pred_dir} matching {METHOD_TO_PATTERN[method]!r}."
        )

    aggregated_items: list[dict[str, Any]] = []
    for prediction_file in prediction_files:
        report = load_json(prediction_file)
        if not isinstance(report, dict):
            raise ValueError(f"Prediction report {prediction_file} must be a JSON object.")
        prediction_items = report.get("predictions")
        if not isinstance(prediction_items, list):
            raise ValueError(
                f"Prediction report {prediction_file} has a non-list predictions field."
            )

        for item in prediction_items:
            if not isinstance(item, dict):
                raise ValueError(
                    f"Prediction report {prediction_file} contains a non-object prediction item."
                )
            aggregated_items.append(
                {
                    "question_id": _require_question_id(item, "Prediction item"),
                    "_dataset_prediction_item": item,
                    "_source_report": report,
                    "_source_prediction_file": prediction_file,
                }
            )

    ordered_wrapped_items, coverage = order_items_by_dataset(
        dataset_items,
        aggregated_items,
        observed_label="predictions",
    )

    dataset_by_id = {
        _require_question_id(item, "Dataset item"): item for item in dataset_items
    }
    exported_items = [
        _build_export_item(
            dataset_item=dataset_by_id[item["question_id"]],
            prediction_item=item["_dataset_prediction_item"],
            report=item["_source_report"],
            method=method,
            source_prediction_file=item["_source_prediction_file"],
            include_debug_fields=include_debug_fields,
        )
        for item in ordered_wrapped_items
    ]

    save_json(exported_items, output_path)
    return {
        "prediction_files": [str(path) for path in prediction_files],
        "num_prediction_files": len(prediction_files),
        "num_dataset_questions": len(dataset_items),
        "num_exported_predictions": len(exported_items),
        "missing_count": coverage.missing_count,
        "extra_count": coverage.extra_count,
        "duplicate_count": coverage.duplicate_count,
        "method": method,
        "dataset_path": str(dataset_path),
        "output_path": str(output_path),
        "include_debug_fields": include_debug_fields,
        "results": exported_items,
    }
