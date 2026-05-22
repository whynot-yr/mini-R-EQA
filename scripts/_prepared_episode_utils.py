from __future__ import annotations

from pathlib import Path

from mini_eqa.preprocess.build_caption_embeddings import resolve_output_dir
from mini_eqa.utils.io_utils import load_json

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def list_prepared_episode_dirs(prepared_root: Path) -> list[Path]:
    return sorted(path for path in prepared_root.iterdir() if path.is_dir())


def slice_episode_dirs(
    episode_dirs: list[Path],
    start_index: int = 0,
    end_index: int | None = None,
    limit_episodes: int | None = None,
    shard_id: int | None = None,
    num_shards: int | None = None,
) -> list[Path]:
    if start_index < 0:
        raise ValueError(f"start_index must be non-negative, got {start_index}")
    if end_index is not None and end_index < start_index:
        raise ValueError(
            f"end_index must be >= start_index, got start_index={start_index}, end_index={end_index}"
        )
    if limit_episodes is not None and limit_episodes <= 0:
        raise ValueError(
            f"limit_episodes must be positive when provided, got {limit_episodes}"
        )
    if (shard_id is None) ^ (num_shards is None):
        raise ValueError("shard_id and num_shards must be provided together.")
    if num_shards is not None:
        if num_shards <= 0:
            raise ValueError(f"num_shards must be positive, got {num_shards}")
        if shard_id is None or shard_id < 0 or shard_id >= num_shards:
            raise ValueError(
                f"shard_id must be in [0, {num_shards}), got {shard_id}"
            )

    selected = episode_dirs[start_index:end_index]
    if num_shards is not None and shard_id is not None:
        selected = [
            episode_dir
            for idx, episode_dir in enumerate(selected)
            if idx % num_shards == shard_id
        ]
    if limit_episodes is not None:
        selected = selected[:limit_episodes]
    return selected


def count_image_files(frames_dir: Path) -> int:
    return sum(
        1
        for path in frames_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def load_episode_meta(episode_dir: Path) -> dict:
    return load_json(episode_dir / "episode_meta.json")


def expected_embedding_paths(episode_dir: Path, model_name: str) -> tuple[Path, Path]:
    output_dir = resolve_output_dir(episode_dir, model_name, None)
    return (
        output_dir / "caption_embeddings.npy",
        output_dir / "caption_embedding_meta.json",
    )
