from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mini_eqa.exporters.official_results import export_official_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export per-episode mini-R-EQA prediction reports into official-style OpenEQA results."
    )
    parser.add_argument("--pred_dir", type=str, required=True)
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["reqa", "uniform"],
    )
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--include_debug_fields", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = export_official_results(
        pred_dir=args.pred_dir,
        method=args.method,
        dataset_path=args.dataset,
        output_path=args.output,
        include_debug_fields=args.include_debug_fields,
    )

    print("=" * 80)
    print("Official-Style OpenEQA Results Export")
    print("=" * 80)
    print(f"Prediction files: {summary['num_prediction_files']}")
    print(f"Dataset questions: {summary['num_dataset_questions']}")
    print(f"Exported predictions: {summary['num_exported_predictions']}")
    print(f"Missing count: {summary['missing_count']}")
    print(f"Extra count: {summary['extra_count']}")
    print(f"Duplicate count: {summary['duplicate_count']}")
    print(f"Output path: {summary['output_path']}")


if __name__ == "__main__":
    main()
