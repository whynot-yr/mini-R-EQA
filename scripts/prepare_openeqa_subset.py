from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mini_eqa.data_adapters.openeqa_adapter import _load_items, convert_items
from mini_eqa.data_adapters.openeqa_paths import (
    make_episode_output_dir,
    resolve_episode_frames_dir,
)
from mini_eqa.utils.io_utils import save_json


UNKNOWN_EPISODE_HISTORY = "__missing_episode_history__"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a small OpenEQA subset grouped by episode_history."
    )
    parser.add_argument("--qa_file", type=str, required=True)
    parser.add_argument("--frames_root", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--strict_frames", action="store_true")
    return parser.parse_args()


def group_questions_by_episode(
    questions: list[dict],
) -> dict[str, list[dict]]:
    grouped_questions: dict[str, list[dict]] = {}
    for question_item in questions:
        episode_history = question_item.get("episode_history") or UNKNOWN_EPISODE_HISTORY
        grouped_questions.setdefault(str(episode_history), []).append(question_item)
    return grouped_questions


def unique_scene_ids(question_items: list[dict]) -> list[str]:
    seen = set()
    scene_ids = []

    for item in question_items:
        scene_id = item.get("scene_id")
        if scene_id is None or scene_id in seen:
            continue
        seen.add(scene_id)
        scene_ids.append(scene_id)

    return scene_ids


def prepare_subset(
    qa_file: str | Path,
    frames_root: str | Path,
    output_root: str | Path,
    start: int = 0,
    limit: int | None = None,
    overwrite: bool = False,
    strict_frames: bool = False,
) -> list[dict]:
    qa_file = Path(qa_file)
    frames_root = Path(frames_root)
    output_root = Path(output_root)

    if not qa_file.exists():
        raise FileNotFoundError(f"QA file does not exist: {qa_file}")
    if not frames_root.exists():
        raise FileNotFoundError(f"Frames root does not exist: {frames_root}")
    if output_root.exists() and any(output_root.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output root {output_root} is not empty. Use --overwrite to replace files."
        )

    raw_items = _load_items(qa_file)
    questions = convert_items(
        raw_items=raw_items,
        start=start,
        limit=limit,
    )
    grouped_questions = group_questions_by_episode(questions)
    summaries = []

    for episode_history, episode_questions in grouped_questions.items():
        episode_output_dir = make_episode_output_dir(output_root, episode_history)
        episode_output_dir.mkdir(parents=True, exist_ok=True)

        frames_dir_str = None
        try:
            frames_dir = resolve_episode_frames_dir(frames_root, episode_history)
            frames_dir_str = str(frames_dir)
        except FileNotFoundError as exc:
            if strict_frames:
                raise
            print(f"Warning: {exc}")

        save_json(episode_questions, episode_output_dir / "questions.json")
        episode_meta = {
            "episode_history": episode_history,
            "scene_id_list": unique_scene_ids(episode_questions),
            "num_questions": len(episode_questions),
            "frames_dir": frames_dir_str,
        }
        save_json(episode_meta, episode_output_dir / "episode_meta.json")

        summaries.append(
            {
                "episode_history": episode_history,
                "output_dir": str(episode_output_dir),
                "num_questions": len(episode_questions),
                "frames_dir": frames_dir_str,
            }
        )

    return summaries


def main() -> None:
    args = parse_args()
    summaries = prepare_subset(
        qa_file=args.qa_file,
        frames_root=args.frames_root,
        output_root=args.output_root,
        start=args.start,
        limit=args.limit,
        overwrite=args.overwrite,
        strict_frames=args.strict_frames,
    )

    print("=" * 80)
    print("OpenEQA Small Subset Prepared")
    print("=" * 80)
    print(f"QA file: {args.qa_file}")
    print(f"Frames root: {args.frames_root}")
    print(f"Output root: {args.output_root}")
    print(f"Num episodes: {len(summaries)}")
    for summary in summaries:
        print("-" * 80)
        print(f"Episode history: {summary['episode_history']}")
        print(f"Output dir: {summary['output_dir']}")
        print(f"Num questions: {summary['num_questions']}")
        print(f"Frames dir: {summary['frames_dir']}")


if __name__ == "__main__":
    main()
