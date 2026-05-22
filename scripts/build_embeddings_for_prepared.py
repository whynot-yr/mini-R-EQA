from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from mini_eqa.preprocess.build_caption_embeddings import (
    DEFAULT_MODEL_NAME,
    build_metadata,
    load_model,
    resolve_output_dir,
)
from mini_eqa.utils.io_utils import load_json, save_json
from scripts._prepared_episode_utils import (
    expected_embedding_paths,
    list_prepared_episode_dirs,
    slice_episode_dirs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-build caption embeddings for prepared episodes."
    )
    parser.add_argument("--prepared_root", type=str, required=True)
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--device", type=str, default=None)
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

    model = None
    rows = []
    for episode_dir in episode_dirs:
        captions_path = episode_dir / "captions.json"
        embeddings_path, metadata_path = expected_embedding_paths(
            episode_dir, args.model_name
        )
        row = {
            "episode": episode_dir.name,
            "episode_dir": str(episode_dir),
            "embeddings_path": str(embeddings_path),
            "metadata_path": str(metadata_path),
            "status": "ok",
            "message": None,
        }

        if not captions_path.exists():
            row["status"] = "missing_captions"
            row["message"] = "captions.json is missing."
            rows.append(row)
            continue

        if embeddings_path.exists() and metadata_path.exists() and not args.overwrite:
            row["status"] = "skipped_existing"
            row["message"] = "Embedding cache already exists."
            rows.append(row)
            continue

        if args.dry_run:
            row["status"] = "dry_run"
            row["message"] = "Would build caption embeddings."
            rows.append(row)
            continue

        if model is None:
            model = load_model(args.model_name, args.device)

        try:
            captions = load_json(captions_path)
            caption_texts = [item["caption"] for item in captions]
            caption_embeddings = model.encode(
                caption_texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                batch_size=32,
                show_progress_bar=True,
            )

            output_dir = resolve_output_dir(episode_dir, args.model_name, None)
            output_dir.mkdir(parents=True, exist_ok=True)
            np.save(embeddings_path, caption_embeddings)
            metadata = build_metadata(
                captions=captions,
                model_name=args.model_name,
                captions_path=captions_path,
                caption_embeddings=caption_embeddings,
            )
            save_json(metadata, metadata_path)
            row["message"] = f"Built embeddings for {len(captions)} captions."
        except Exception as exc:
            row["status"] = "failed"
            row["message"] = str(exc)

        rows.append(row)

    status_path = prepared_root / "embedding_status.json"
    save_json(
        {
            "prepared_root": str(prepared_root),
            "model_name": args.model_name,
            "dry_run": args.dry_run,
            "rows": rows,
        },
        status_path,
    )

    print("=" * 80)
    print("Build Embeddings For Prepared Episodes")
    print("=" * 80)
    print(f"Prepared root: {prepared_root}")
    print(f"Model name: {args.model_name}")
    print(f"Dry run: {args.dry_run}")
    print(f"Num episodes: {len(rows)}")
    print(f"Saved status to: {status_path}")


if __name__ == "__main__":
    main()
