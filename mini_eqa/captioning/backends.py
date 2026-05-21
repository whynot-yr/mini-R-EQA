from __future__ import annotations

from pathlib import Path


def caption_with_filename_stub(image_path: Path) -> str:
    return f"Frame image {image_path.stem} shows an unknown scene."


def get_captioner(name: str):
    if name == "filename_stub":
        return caption_with_filename_stub

    if name in {"qwen_vl", "blip"}:
        raise NotImplementedError(
            "This backend is reserved for future VLM captioning. "
            "Use filename_stub for pipeline testing."
        )

    raise ValueError(f"Unknown captioning backend: {name}")
