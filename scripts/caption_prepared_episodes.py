from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mini_eqa.captioning.caption_frames import build_captions, collect_frame_paths
from mini_eqa.captioning.backends import get_captioner
from mini_eqa.utils.io_utils import save_json
from scripts._prepared_episode_utils import (
    list_prepared_episode_dirs,
    load_episode_meta,
    slice_episode_dirs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-caption prepared OpenEQA episodes."
    )
    parser.add_argument("--prepared_root", type=str, required=True)
    parser.add_argument("--backend", type=str, default="filename_stub")
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--torch_dtype", type=str, default="auto")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--limit_episodes", type=int, default=None)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=None)
    parser.add_argument("--shard_id", type=int, default=None)
    parser.add_argument("--num_shards", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    prepared_root = Path(args.prepared_root)
    if not prepared_root.exists():
        raise FileNotFoundError(f"Prepared root does not exist: {prepared_root}")

    episode_dirs = slice_episode_dirs(
        list_prepared_episode_dirs(prepared_root),
        start_index=args.start_index,
        end_index=args.end_index,
        limit_episodes=args.limit_episodes,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
    )

    captioner = None
    if not args.dry_run:
        captioner = get_captioner(
            args.backend,
            model_name=args.model_name,
            device=args.device,
            torch_dtype=args.torch_dtype,
            max_new_tokens=args.max_new_tokens,
        )

    rows = []
    for episode_dir in episode_dirs:
        captions_path = episode_dir / "captions.json"
        row = {
            "episode": episode_dir.name,
            "episode_dir": str(episode_dir),
            "captions_path": str(captions_path),
            "status": "ok",
            "message": None,
        }

        if captions_path.exists() and not args.overwrite:
            row["status"] = "skipped_existing"
            row["message"] = "captions.json already exists."
            rows.append(row)
            continue

        episode_meta_path = episode_dir / "episode_meta.json"
        if not episode_meta_path.exists():
            row["status"] = "missing_episode_meta"
            row["message"] = "episode_meta.json is missing."
            rows.append(row)
            continue

        episode_meta = load_episode_meta(episode_dir)
        frames_dir_value = episode_meta.get("frames_dir")
        if not frames_dir_value:
            row["status"] = "missing_frames_dir"
            row["message"] = "episode_meta.json does not contain a usable frames_dir."
            rows.append(row)
            continue

        frames_dir = Path(frames_dir_value)
        if not frames_dir.exists():
            row["status"] = "missing_frames_dir"
            row["message"] = f"frames_dir does not exist: {frames_dir}"
            rows.append(row)
            continue

        try:
            frame_paths = collect_frame_paths(frames_dir, "*.png", None)
            if not frame_paths:
                frame_paths = collect_frame_paths(frames_dir, "*.jpg", None)
            if not frame_paths:
                frame_paths = collect_frame_paths(frames_dir, "*.jpeg", None)
            if not frame_paths:
                raise ValueError(
                    f"No images found in {frames_dir} matching .png/.jpg/.jpeg."
                )
        except Exception as exc:
            row["status"] = "invalid_frames_dir"
            row["message"] = str(exc)
            rows.append(row)
            continue

        if args.dry_run:
            row["status"] = "dry_run"
            row["message"] = f"Would caption {len(frame_paths)} frames."
            rows.append(row)
            continue

        try:
            captions = build_captions(frame_paths, captioner)
            save_json(captions, captions_path)
            row["message"] = f"Captioned {len(captions)} frames."
        except Exception as exc:
            row["status"] = "failed"
            row["message"] = str(exc)

        rows.append(row)

    status_path = prepared_root / "caption_status.json"
    save_json(
        {
            "prepared_root": str(prepared_root),
            "backend": args.backend,
            "dry_run": args.dry_run,
            "rows": rows,
        },
        status_path,
    )

    print("=" * 80)
    print("Caption Prepared Episodes")
    print("=" * 80)
    print(f"Prepared root: {prepared_root}")
    print(f"Backend: {args.backend}")
    print(f"Dry run: {args.dry_run}")
    print(f"Num episodes: {len(rows)}")
    print(f"Saved status to: {status_path}")


if __name__ == "__main__":
    main()
