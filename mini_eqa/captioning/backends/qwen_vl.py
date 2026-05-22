from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any


DEFAULT_QWEN_VL_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_QWEN_VL_PROMPT = (
    "Describe the scene in this image for embodied question answering. "
    "Mention the room type, visible objects, spatial relations, and useful "
    "landmarks. Be concise but specific."
)
DEFAULT_MAX_NEW_TOKENS = 128


def _missing_dependency_error() -> ImportError:
    return ImportError(
        "Qwen-VL backend requires torch, transformers, qwen-vl-utils, and pillow."
    )


def _resolve_torch_dtype(torch_module: Any, torch_dtype: str):
    if torch_dtype == "auto":
        return "auto"

    dtype_name = torch_dtype.lower()
    dtype_map = {
        "float16": torch_module.float16,
        "fp16": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
        "bf16": torch_module.bfloat16,
        "float32": torch_module.float32,
        "fp32": torch_module.float32,
    }
    if dtype_name not in dtype_map:
        raise ValueError(
            f"Unsupported torch_dtype: {torch_dtype}. "
            "Use auto, float16, bfloat16, or float32."
        )
    return dtype_map[dtype_name]


def _resolve_device(torch_module: Any, device: str | None) -> str:
    if device is not None:
        return device
    return "cuda" if torch_module.cuda.is_available() else "cpu"


@lru_cache(maxsize=4)
def _load_qwen_vl_components(
    model_name: str,
    device: str,
    torch_dtype: str,
    attn_implementation: str | None,
    device_map: str | None,
):
    try:
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    except ImportError as exc:
        raise _missing_dependency_error() from exc

    resolved_dtype = _resolve_torch_dtype(torch, torch_dtype)

    try:
        model_kwargs = {"torch_dtype": resolved_dtype}
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation
        if device_map is not None:
            model_kwargs["device_map"] = device_map

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            **model_kwargs,
        )
        processor = AutoProcessor.from_pretrained(model_name)
    except Exception as exc:
        raise RuntimeError(
            "Failed to load Qwen-VL model or processor. "
            "Ensure the model is accessible via the HuggingFace cache or network."
        ) from exc

    if device_map is None:
        try:
            model = model.to(device)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to move Qwen-VL model to device={device!r}."
            ) from exc

    model.eval()
    return model, processor


class QwenVLCaptioner:
    def __init__(
        self,
        model_name: str = DEFAULT_QWEN_VL_MODEL,
        device: str | None = None,
        torch_dtype: str = "auto",
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        prompt: str | None = None,
        attn_implementation: str | None = None,
        device_map: str | None = None,
    ) -> None:
        try:
            import torch
            from PIL import Image  # noqa: F401
            from qwen_vl_utils import process_vision_info  # noqa: F401
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration  # noqa: F401
        except ImportError as exc:
            raise _missing_dependency_error() from exc

        if max_new_tokens <= 0:
            raise ValueError(
                f"max_new_tokens must be positive, got {max_new_tokens}"
            )

        self.model_name = model_name
        self.device = _resolve_device(torch, device)
        self.torch_dtype = torch_dtype
        self.max_new_tokens = max_new_tokens
        self.prompt = prompt or DEFAULT_QWEN_VL_PROMPT
        self.attn_implementation = attn_implementation
        self.device_map = device_map
        self._model, self._processor = _load_qwen_vl_components(
            model_name=self.model_name,
            device=self.device,
            torch_dtype=self.torch_dtype,
            attn_implementation=self.attn_implementation,
            device_map=self.device_map,
        )

    def __call__(self, image_path: str | Path) -> str:
        try:
            import torch
            from PIL import Image
            from qwen_vl_utils import process_vision_info
        except ImportError as exc:
            raise _missing_dependency_error() from exc

        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file does not exist: {image_path}")

        try:
            with Image.open(image_path) as img:
                img.verify()
        except Exception as exc:
            raise RuntimeError(f"Failed to open image for Qwen-VL: {image_path}") from exc

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": self.prompt},
                ],
            }
        ]

        try:
            chat_text = self._processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            image_inputs, video_inputs = process_vision_info(messages)
            model_inputs = self._processor(
                text=[chat_text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            model_inputs = model_inputs.to(self.device)
            generated_ids = self._model.generate(
                **model_inputs,
                max_new_tokens=self.max_new_tokens,
            )
            generated_trimmed = [
                output_ids[len(input_ids):]
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            output_text = self._processor.batch_decode(
                generated_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Qwen-VL caption generation failed for image: {image_path}"
            ) from exc

        caption = output_text[0].strip() if output_text else ""
        if not caption:
            raise RuntimeError(
                f"Qwen-VL returned an empty caption for image: {image_path}"
            )

        return caption
