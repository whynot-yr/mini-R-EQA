from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mini_eqa.data_adapters.openeqa_adapter import _load_items, convert_items
from mini_eqa.data_adapters.openeqa_paths import resolve_episode_frames_dir
from mini_eqa.utils.io_utils import save_json
from scripts._prepared_episode_utils import count_image_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate OpenEQA metadata and episode frame availability."
    )
    parser.add_argument("--qa_file", type=str, required=True)
    parser.add_argument("--frames_root", type=str, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--output", type=str, default="reports/validation_openeqa.json"
    )
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    qa_file = Path(args.qa_file)
    frames_root = Path(args.frames_root)

    if not qa_file.exists():
        raise FileNotFoundError(f"QA file does not exist: {qa_file}")
    if not frames_root.exists():
        raise FileNotFoundError(f"Frames root does not exist: {frames_root}")

    raw_items = _load_items(qa_file)
    questions = convert_items(raw_items=raw_items, limit=args.limit)

    items = []
    num_valid = 0
    num_missing_frames_dir = 0
    num_empty_frame_dirs = 0
    missing_fields_count = 0
    severe_messages = []

    for question_item in questions:
        item_result = {
            "question_id": question_item.get("question_id"),
            "question": question_item.get("question"),
            "episode_history": question_item.get("episode_history"),
            "answer": question_item.get("answer"),
            "has_question": bool(question_item.get("question")),
            "has_answer": question_item.get("answer") is not None,
            "has_episode_history": bool(question_item.get("episode_history")),
            "frames_dir": None,
            "frame_count": 0,
            "status": "valid",
            "message": None,
        }

        if not item_result["has_question"] or not item_result["has_answer"] or not item_result["has_episode_history"]:
            item_result["status"] = "missing_fields"
            item_result["message"] = "Missing required question, answer, or episode_history field."
            missing_fields_count += 1
            items.append(item_result)
            severe_messages.append(item_result["message"])
            continue

        try:
            frames_dir = resolve_episode_frames_dir(
                frames_root=frames_root,
                episode_history=str(question_item["episode_history"]),
            )
            item_result["frames_dir"] = str(frames_dir)
            item_result["frame_count"] = count_image_files(frames_dir)
        except FileNotFoundError as exc:
            item_result["status"] = "missing_frames_dir"
            item_result["message"] = str(exc)
            num_missing_frames_dir += 1
            items.append(item_result)
            severe_messages.append(item_result["message"])
            continue

        if item_result["frame_count"] == 0:
            item_result["status"] = "empty_frames_dir"
            item_result["message"] = "Resolved frames_dir contains no image files."
            num_empty_frame_dirs += 1
            items.append(item_result)
            severe_messages.append(item_result["message"])
            continue

        num_valid += 1
        items.append(item_result)

    report = {
        "qa_file": str(qa_file),
        "frames_root": str(frames_root),
        "num_checked": len(questions),
        "num_valid": num_valid,
        "num_missing_fields": missing_fields_count,
        "num_missing_frames_dir": num_missing_frames_dir,
        "num_empty_frame_dirs": num_empty_frame_dirs,
        "items": items,
    }
    save_json(report, args.output)

    print("=" * 80)
    print("OpenEQA Data Validation")
    print("=" * 80)
    print(f"QA file: {qa_file}")
    print(f"Frames root: {frames_root}")
    print(f"Checked items: {report['num_checked']}")
    print(f"Valid items: {report['num_valid']}")
    print(f"Missing fields: {report['num_missing_fields']}")
    print(f"Missing frames dirs: {report['num_missing_frames_dir']}")
    print(f"Empty frames dirs: {report['num_empty_frame_dirs']}")
    print(f"Saved report to: {args.output}")

    if args.strict and severe_messages:
        raise RuntimeError(
            "Strict validation failed. "
            f"Found {len(severe_messages)} issues. See {args.output} for details."
        )


if __name__ == "__main__":
    main()
