from __future__ import annotations

from pathlib import Path


def caption_with_filename_stub(image_path: Path) -> str:
    return f"Frame image {image_path.stem} shows an unknown scene."
