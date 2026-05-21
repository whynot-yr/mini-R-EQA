from __future__ import annotations

import argparse
import re
from pathlib import Path

from mini_eqa.captioning.backends import get_captioner
from mini_eqa.utils.io_utils import save_json


FRAME_STEM_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate captions.json from a directory of episode frame images."
    )
    parser.add_argument("--frames_dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--backend", type=str, default="filename_stub")
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--glob", type=str, default="*.png")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def infer_frame_id(image_path: Path, index: int) -> str:
    stem = image_path.stem.strip()
    if stem and FRAME_STEM_PATTERN.fullmatch(stem):
        return stem
    return f"frame_{index:06d}"


def collect_frame_paths(frames_dir: Path, pattern: str, limit: int | None) -> list[Path]:
    frame_paths = sorted(path for path in frames_dir.glob(pattern) if path.is_file())
    if limit is not None:
        if limit <= 0:
            raise ValueError(f"limit must be positive when provided, got {limit}")
        frame_paths = frame_paths[:limit]
    return frame_paths


def build_captions(
    frame_paths: list[Path],
    captioner,
) -> list[dict[str, str]]:
    captions = []
    for index, image_path in enumerate(frame_paths, start=1):
        captions.append(
            {
                "frame_id": infer_frame_id(image_path, index),
                "image_path": str(image_path),
                "caption": captioner(image_path),
            }
        )
    return captions


def main() -> None:
    args = parse_args()

    frames_dir = Path(args.frames_dir)
    output_path = Path(args.output)

    if not frames_dir.exists():
        raise FileNotFoundError(f"Frames directory does not exist: {frames_dir}")
    if not frames_dir.is_dir():
        raise NotADirectoryError(f"Frames path is not a directory: {frames_dir}")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists at {output_path}. Use --overwrite to replace it."
        )

    frame_paths = collect_frame_paths(
        frames_dir=frames_dir,
        pattern=args.glob,
        limit=args.limit,
    )
    if not frame_paths:
        raise ValueError(
            f"No images found in {frames_dir} matching glob pattern {args.glob!r}."
        )

    try:
        captioner = get_captioner(
            args.backend,
            model_name=args.model_name,
            device=args.device,
        )
    except (ImportError, NotImplementedError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    captions = build_captions(frame_paths=frame_paths, captioner=captioner)
    save_json(captions, output_path)

    print("=" * 80)
    print("Frame Captioning Complete")
    print("=" * 80)
    print(f"Frames dir: {frames_dir}")
    print(f"Backend: {args.backend}")
    print(f"Model name: {args.model_name}")
    print(f"Device: {args.device}")
    print(f"Num frames: {len(captions)}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
