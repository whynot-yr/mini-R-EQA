from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mini_eqa.exporters.openeqa_predictions import export_predictions
from mini_eqa.utils.io_utils import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export internal predictions and optionally call official OpenEQA evaluation."
    )
    parser.add_argument("--internal_predictions", type=str, required=True)
    parser.add_argument("--exported_predictions", type=str, required=True)
    parser.add_argument("--openeqa_eval_script", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--include_debug_fields", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    internal_predictions = Path(args.internal_predictions)
    exported_predictions = Path(args.exported_predictions)
    eval_script = Path(args.openeqa_eval_script)

    if not internal_predictions.exists():
        raise FileNotFoundError(
            f"Internal predictions file does not exist: {internal_predictions}"
        )

    exported_items = export_predictions(
        predictions_path=internal_predictions,
        output_path=exported_predictions,
        include_debug_fields=args.include_debug_fields,
    )

    log = {
        "internal_predictions": str(internal_predictions),
        "exported_predictions": str(exported_predictions),
        "openeqa_eval_script": str(eval_script),
        "include_debug_fields": args.include_debug_fields,
        "dry_run": args.dry_run,
        "num_exported_items": len(exported_items),
        "status": "exported",
        "message": None,
        "command": None,
        "returncode": None,
        "stdout": None,
        "stderr": None,
    }

    if args.dry_run:
        log["message"] = "Dry run: exported predictions only; official evaluator not called."
    elif not eval_script.exists():
        log["status"] = "missing_eval_script"
        log["message"] = (
            "Official OpenEQA evaluate-predictions.py script not found. "
            "Provide a valid --openeqa_eval_script path to run official evaluation."
        )
    else:
        command = [sys.executable, str(eval_script), str(exported_predictions)]
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
        log["command"] = command
        log["returncode"] = completed.returncode
        log["stdout"] = completed.stdout
        log["stderr"] = completed.stderr
        if completed.returncode == 0:
            log["status"] = "evaluated"
            log["message"] = "Official evaluator completed."
        else:
            log["status"] = "eval_failed"
            log["message"] = (
                "Official evaluator returned a non-zero exit code. "
                "Inspect stdout/stderr in this wrapper log."
            )

    save_json(log, args.output)

    print("=" * 80)
    print("OpenEQA Evaluation Wrapper")
    print("=" * 80)
    print(f"Internal predictions: {internal_predictions}")
    print(f"Exported predictions: {exported_predictions}")
    print(f"Evaluator script: {eval_script}")
    print(f"Dry run: {args.dry_run}")
    print(f"Status: {log['status']}")
    print(f"Message: {log['message']}")
    print(f"Saved log to: {args.output}")


if __name__ == "__main__":
    main()
