from __future__ import annotations

from mini_eqa.captioning.backends.filename_stub import caption_with_filename_stub
from mini_eqa.captioning.backends.qwen_vl import QwenVLCaptioner


def get_captioner(
    name: str,
    model_name: str | None = None,
    device: str | None = None,
    **kwargs,
):
    if name == "filename_stub":
        return caption_with_filename_stub

    if name == "qwen_vl":
        return QwenVLCaptioner(
            model_name=model_name or "Qwen/Qwen2.5-VL-7B-Instruct",
            device=device,
            **kwargs,
        )

    if name == "blip":
        raise NotImplementedError(
            "BLIP backend is reserved for future VLM captioning. "
            "Use filename_stub for pipeline testing."
        )

    raise ValueError(f"Unknown captioning backend: {name}")


__all__ = [
    "QwenVLCaptioner",
    "caption_with_filename_stub",
    "get_captioner",
]
