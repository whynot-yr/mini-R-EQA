from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from mini_eqa.utils.io_utils import load_json, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export internal mini-R-EQA predictions to an OpenEQA-compatible JSON list."
    )
    parser.add_argument("--predictions", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--include_debug_fields", action="store_true")
    return parser.parse_args()


def build_export_item(
    item: dict[str, Any],
    report: dict[str, Any],
    include_debug_fields: bool,
) -> dict[str, Any]:
    export_item = {
        "question_id": item.get("question_id"),
        "question": item.get("question"),
        "answer": item.get("gold_answer"),
        "prediction": item.get("predicted_answer"),
        "episode_history": item.get("episode_history", report.get("episode_history")),
        "scene_id": item.get("scene_id"),
    }

    if include_debug_fields:
        export_item["retrieved"] = item.get("retrieved")
        export_item["prompt"] = item.get("prompt")
        export_item["retriever"] = report.get("retriever")
        export_item["runner"] = report.get("runner")
        export_item["model"] = report.get("model")

    return export_item


def export_predictions(
    predictions_path: str | Path,
    output_path: str | Path,
    include_debug_fields: bool = False,
) -> list[dict[str, Any]]:
    predictions_path = Path(predictions_path)
    report = load_json(predictions_path)

    if not isinstance(report, dict):
        raise ValueError(
            f"Prediction report {predictions_path} must be a JSON object."
        )
    if "predictions" not in report:
        raise KeyError(
            f"Prediction report {predictions_path} does not contain a predictions field."
        )

    prediction_items = report["predictions"]
    if not isinstance(prediction_items, list):
        raise ValueError(
            f"Prediction report {predictions_path} has a non-list predictions field."
        )

    exported_items = [
        build_export_item(
            item=item,
            report=report,
            include_debug_fields=include_debug_fields,
        )
        for item in prediction_items
    ]

    save_json(exported_items, output_path)
    return exported_items


def main() -> None:
    args = parse_args()
    exported_items = export_predictions(
        predictions_path=args.predictions,
        output_path=args.output,
        include_debug_fields=args.include_debug_fields,
    )

    print("=" * 80)
    print("OpenEQA-Compatible Prediction Export")
    print("=" * 80)
    print(f"Input predictions: {args.predictions}")
    print(f"Output: {args.output}")
    print(f"Include debug fields: {args.include_debug_fields}")
    print(f"Num exported items: {len(exported_items)}")


if __name__ == "__main__":
    main()
