from __future__ import annotations

from pathlib import Path


DEFAULT_QWEN_VL_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_QWEN_VL_PROMPT = (
    "Describe the scene in this image for embodied question answering. "
    "Mention objects, room type, spatial relations, and visible landmarks."
)


class QwenVLCaptioner:
    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name or DEFAULT_QWEN_VL_MODEL
        self.device = device
        self.prompt = DEFAULT_QWEN_VL_PROMPT

        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
            import qwen_vl_utils  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "Qwen-VL backend requires transformers, torch, qwen-vl-utils, "
                "and a GPU-capable environment."
            ) from exc

    def __call__(self, image_path: Path) -> str:
        raise NotImplementedError(
            "Qwen-VL captioner skeleton is defined, but real inference is not "
            "implemented yet. Do not auto-download models; wire local model loading "
            "and generation in a future step."
        )
