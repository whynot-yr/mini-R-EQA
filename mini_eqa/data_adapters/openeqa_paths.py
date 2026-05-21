from __future__ import annotations

from pathlib import Path


def resolve_episode_frames_dir(
    frames_root: str | Path,
    episode_history: str,
) -> Path:
    frames_root = Path(frames_root)
    frames_dir = frames_root / episode_history

    if not frames_dir.exists():
        raise FileNotFoundError(
            f"Could not resolve frames directory for episode_history={episode_history!r}. "
            f"Expected path: {frames_dir}"
        )

    return frames_dir


def make_episode_output_dir(
    output_root: str | Path,
    episode_history: str,
) -> Path:
    output_root = Path(output_root)
    safe_episode_name = episode_history.replace("/", "__")
    return output_root / safe_episode_name
